from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
from pydantic import BaseModel
from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
import schedule
import threading
import time
from db.database import init_db, save_conversation, get_conversation_history, save_reminder, get_reminders
from langdetect import detect, DetectorFactory
import logging
import json
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Handle preflight OPTIONS requests
@app.options("/{rest_of_path:path}", include_in_schema=False)
async def preflight_handler(rest_of_path: str, request: Request):
    return JSONResponse(content={}, status_code=200)

llm = Ollama(model="nous-hermes2:10.7b-solar-q3_k_m", temperature=0.3)
memory = ConversationBufferMemory(return_messages=True)

# Ensure consistent language detection
DetectorFactory.seed = 0

# Initialize database
init_db()

# Pydantic model for structured LLM output
class ResponseOutput(BaseModel):
    type: str  # "greeting", "vat", "profit", or "general"
    name: str = None
    amount: float = None
    rate: float = None
    revenue: float = None
    expenses: float = None
    tax_rate: float = None
    result: float = None
    response: str  # REQUIRED

# Enhanced prompt template
response_parser = PydanticOutputParser(pydantic_object=ResponseOutput)
prompt_template = PromptTemplate(
    input_variables=["history", "input", "language"],
    template="You are a professional tax assistant for Tunisia. Respond EXCLUSIVELY in {language}, using accurate, concise, and professional tax terminology. "
             "For Arabic, use conversational Tunisian Arabic with local terms (e.g., 'شنو' for 'what', 'كيفاش' for 'how', 'مرحبا' for greetings, 'تڤا' for VAT). "
             "Avoid formal Modern Standard Arabic. Do not switch languages. "
             "Analyze the input to determine the response type: "
             "- If the user introduces their name (e.g., 'my name is Ahmed', 'je m'appelle Ahmed', 'people call me Ahmed', 'انا احمد'), extract the name and return a personalized greeting. "
             "- If the user asks for a VAT calculation (e.g., 'Calculate VAT for 100 TND at 7%'), compute the VAT (amount * rate / 100) and return the result. "
             "- If the user asks for a net profit calculation (e.g., 'Calculate net profit for revenue 1000 TND and expenses 400 TND'), compute the profit ((revenue - expenses) * (1 - tax_rate/100)) and return the result. "
             "- For other queries, provide a clear and professional tax-related response. "
             "Return a SINGLE JSON object with: {{type}} ('greeting', 'vat', 'profit', or 'general'), {{name}} (if greeting), {{amount}}/{{rate}} (if VAT), {{revenue}}/{{expenses}}/{{tax_rate}}/{{result}} (if profit), and {{response}} (the natural language answer in {language}). The {{response}} field is REQUIRED. "
             "Example for input 'people call me Ahmed' in English: {{\"type\": \"greeting\", \"name\": \"Ahmed\", \"response\": \"Hello Ahmed! I’m your tax assistant for Tunisia. How can I assist you today?\"}} "
             "Example for input 'je m'appelle Ahmed' in French: {{\"type\": \"greeting\", \"name\": \"Ahmed\", \"response\": \"Bonjour Ahmed ! Je suis votre assistant fiscal pour la Tunisie. Comment puis-je vous aider aujourd'hui ?\"}} "
             "Conversation history:\n{history}\nUser: {input}\n\n{format_instructions}",
    partial_variables={"format_instructions": response_parser.get_format_instructions()}
)

# Fallback prompt template (simpler, no JSON requirement)
fallback_prompt_template = PromptTemplate(
    input_variables=["history", "input", "language"],
    template="You are a professional tax assistant for Tunisia. Respond EXCLUSIVELY in {language}, using accurate, concise, and professional tax terminology. "
             "For Arabic, use conversational Tunisian Arabic with local terms (e.g., 'شنو' for 'what', 'كيفاش' for 'how', 'مرحبا' for greetings, 'تڤا' for VAT). "
             "Avoid formal Modern Standard Arabic. Do not switch languages. "
             "For name introductions (e.g., 'my name is Ahmed', 'je m'appelle Ahmed', 'people call me Ahmed', 'انا احمد'), extract the name and respond with a personalized greeting. "
             "For other queries, provide a clear and professional tax-related response. "
             "Conversation history:\n{history}\nUser: {input}"
)

# Pydantic models for input
class ChatInput(BaseModel):
    message: str
    user_id: str = "default_user"
    language: str = None

class ReminderInput(BaseModel):
    declaration_type: str
    deadline: str
    user_id: str = "default_user"

# Language detection
def detect_language(message: str, specified_language: str) -> str:
    if specified_language in ["fr", "en", "ar"]:
        logger.info(f"Using specified language: {specified_language}")
        return specified_language
    
    normalized_message = " ".join(message.lower().strip().split())
    french_keywords = ["je", "m'appelle", "bonjour", "salut", "tva", "impôt", "société", "entrepreneur", "suis", "appelle"]
    english_keywords = ["my", "name", "is", "hello", "hi", "what", "how", "tax", "vat", "profit", "people", "call", "me"]
    arabic_keywords = ["مرحبا", "شنو", "ضرائب", "تكلفة", "اسمي", "احمد"]
    
    french_score = sum(1 for keyword in french_keywords if keyword in normalized_message)
    english_score = sum(1 for keyword in english_keywords if keyword in normalized_message)
    arabic_score = sum(1 for keyword in arabic_keywords if keyword in normalized_message) or any(char in normalized_message for char in "ابتثجحخدذرزسشص")

    if english_score >= max(french_score, arabic_score):
        logger.info(f"Detected English via keywords: {normalized_message}")
        return "en"
    elif french_score >= max(english_score, arabic_score):
        logger.info(f"Detected French via keywords: {normalized_message}")
        return "fr"
    elif arabic_score > 0:
        logger.info(f"Detected Arabic via keywords/characters: {normalized_message}")
        return "ar"
    
    try:
        lang = detect(normalized_message)
        logger.info(f"Langdetect result: {lang} for message: {normalized_message}")
        if lang in ["fr", "en", "ar"]:
            return lang
        if lang.startswith("ar"):
            return "ar"
        return "en" if "people" in normalized_message or "call" in normalized_message else "fr"
    except Exception as e:
        logger.error(f"Langdetect error: {e}, defaulting to English if 'people' or 'call' present, else French")
        return "en" if "people" in normalized_message or "call" in normalized_message else "fr"

# Profile detection
def detect_profile(message: str) -> str:
    profiles = ["auto-entrepreneur", "société", "particulier"]
    for profile in profiles:
        if profile in message.lower():
            return profile
    return None

# Extract JSON from output
def extract_json_output(raw_output: str, user_input: str) -> dict:
    # Look for JSON object matching the input
    json_pattern = r'\{[^}]*"type"\s*:\s*"[^"]*"\s*,\s*"[^"]*"\s*:\s*[^}]*\}'
    matches = re.findall(json_pattern, raw_output)
    for match in matches:
        try:
            parsed = json.loads(match)
            if parsed.get("type") == "greeting":
                name = parsed.get("name", "").lower()
                if any(word.lower() in user_input.lower() for word in name.split()):
                    return parsed
            if parsed.get("type") in ["vat", "profit", "general"]:
                return parsed
        except json.JSONDecodeError:
            continue
    # Fallback: Try parsing entire output
    try:
        parsed = json.loads(raw_output.strip())
        return parsed
    except json.JSONDecodeError:
        logger.warning(f"No valid JSON found in output: {raw_output}")
        return None

# Reminders scheduler
def run_scheduler():
    while True:
        schedule.run_pending()
        time.sleep(60)

threading.Thread(target=run_scheduler, daemon=True).start()

@app.post("/chat")
async def chat(input: ChatInput):
    user_input = input.message
    user_id = input.user_id
    language = detect_language(user_input, input.language)
    profile = detect_profile(user_input)

    # Load conversation history
    history = get_conversation_history(user_id)
    history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])

    # Save user input
    save_conversation(user_id, "user", user_input, language)

    # LLM response with structured output
    chain = prompt_template | llm | response_parser
    try:
        # Format prompt
        formatted_prompt = prompt_template.format(history=history_text, input=user_input, language=language)
        # Get raw LLM output
        raw_output = llm.invoke(formatted_prompt)
        logger.info(f"Raw LLM output: {raw_output}")
        
        # Preprocess output
        raw_output = raw_output.strip()
        try:
            parsed_output = json.loads(raw_output) if isinstance(raw_output, str) else raw_output
        except json.JSONDecodeError:
            # Try extracting JSON
            parsed_output = extract_json_output(raw_output, user_input)
            if not parsed_output:
                logger.error(f"JSON decode error: No valid JSON found, raw output: {raw_output}")
                raise ValueError("Invalid JSON output from LLM")

        output = response_parser.parse(json.dumps(parsed_output))
        response = output.response
        # Generate response if missing
        if not response:
            if output.type == "greeting" and output.name:
                response = {
                    "en": f"Hello {output.name}! I’m your tax assistant for Tunisia. How can I assist you today?",
                    "fr": f"Bonjour {output.name} ! Je suis votre assistant fiscal pour la Tunisie. Comment puis-je vous aider aujourd'hui ?",
                    "ar": f"مرحبا {output.name}، أنا مساعدك الضريبي في تونس. كيفاش نقدر نساعدك اليوم؟"
                }[language]
            elif output.type == "vat" and output.amount is not None and output.rate is not None:
                result = output.result if output.result is not None else output.amount * output.rate / 100
                response = {
                    "fr": f"La TVA pour {output.amount} TND à {output.rate}% est {result:.2f} TND.",
                    "en": f"The VAT for {output.amount} TND at {output.rate}% is {result:.2f} TND.",
                    "ar": f"الضريبة على القيمة المضافة لـ {output.amount} دينار تونسي بنسبة {output.rate}% هي {result:.2f} دينار تونسي."
                }[language]
            elif output.type == "profit" and output.revenue is not None and output.expenses is not None:
                result = output.result if output.result is not None else (output.revenue - output.expenses) * (1 - (output.tax_rate or 15) / 100)
                response = {
                    "fr": f"Le profit net pour un revenu de {output.revenue} TND et dépenses de {output.expenses} TND est {result:.2f} TND.",
                    "en": f"The net profit for revenue of {output.revenue} TND and expenses of {output.expenses} TND is {result:.2f} TND.",
                    "ar": f"الربح الصافي لإيرادات {output.revenue} دينار تونسي ونفقات {output.expenses} دينار تونسي هو {result:.2f} دينار تونسي."
                }[language]
            else:
                response = {
                    "en": "I’m your tax assistant for Tunisia. How can I assist you today?",
                    "fr": "Je suis votre assistant fiscal pour la Tunisie. Comment puis-je vous aider aujourd'hui ?",
                    "ar": "أنا مساعدك الضريبي في تونس. كيفاش نقدر نساعدك اليوم؟"
                }[language]
        if profile and output.type != "greeting":
            response += {
                "fr": f" (Profil détecté : {profile})",
                "en": f" (Profile detected: {profile})",
                "ar": f" (الملف الشخصي المكتشف: {profile})"
            }[language]
    except Exception as e:
        logger.error(f"LLM processing error: {e}, input: {user_input}, language: {language}")
        # Fallback to simpler prompt
        chain = fallback_prompt_template | llm
        try:
            response = chain.invoke({"history": history_text, "input": user_input, "language": language})
            # Extract name for greeting in fallback
            name_pattern = r"(?:my name is|je m'appelle|people call me|انا)\s+([a-zA-Z\s]+)"
            name_match = re.search(name_pattern, user_input.lower())
            if name_match:
                name = name_match.group(1).title()
                response = {
                    "en": f"Hello {name}! I’m your tax assistant for Tunisia. How can I assist you today?",
                    "fr": f"Bonjour {name} ! Je suis votre assistant fiscal pour la Tunisie. Comment puis-je vous aider aujourd'hui ?",
                    "ar": f"مرحبا {name}، أنا مساعدك الضريبي في تونس. كيفاش نقدر نساعدك اليوم؟"
                }[language]
            if profile:
                response += {
                    "fr": f" (Profil détecté : {profile})",
                    "en": f" (Profile detected: {profile})",
                    "ar": f" (الملف الشخصي المكتشف: {profile})"
                }[language]
        except Exception as e2:
            logger.error(f"Fallback LLM error: {e2}, using default response")
            response = {
                "en": "I’m your tax assistant for Tunisia. How can I assist you today?",
                "fr": "Je suis votre assistant fiscal pour la Tunisie. Comment puis-je vous aider aujourd'hui ?",
                "ar": "أنا مساعدك الضريبي في تونس. كيفاش نقدر نساعدك اليوم؟"
            }[language]
            if profile:
                response += {
                    "fr": f" (Profil détecté : {profile})",
                    "en": f" (Profile detected: {profile})",
                    "ar": f" (الملف الشخصي المكتشف: {profile})"
                }[language]

    # Save bot response
    save_conversation(user_id, "assistant", response, language)
    memory.save_context({"input": user_input}, {"output": response})

    return {"response": response}

@app.post("/set_reminder")
async def set_reminder_endpoint(input: ReminderInput):
    save_reminder(input.user_id, input.declaration_type, input.deadline)
    schedule.every().day.at("08:00").do(lambda: print(f"Reminder: {input.declaration_type} due on {input.deadline} for user {input.user_id}"))
    return {"message": f"Reminder set for {input.declaration_type} on {input.deadline}"}

@app.get("/reminders/{user_id}")
async def get_reminders_endpoint(user_id: str):
    return {"reminders": get_reminders(user_id)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)