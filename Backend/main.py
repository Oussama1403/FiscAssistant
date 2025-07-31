from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
from pydantic import BaseModel
from langchain_ollama import OllamaLLM
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
import schedule
import threading
import time
from db.database import init_db, save_conversation, get_conversation_history, save_reminder, get_reminders
from langdetect import detect, DetectorFactory
import logging
import json
import re
from langchain_community.document_loaders import JSONLoader
from pathlib import Path
from sentence_transformers import SentenceTransformer, util

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

# Initialize LLM with Ollama
try:
    llm = OllamaLLM(
        model="nous-hermes2:10.7b-solar-q3_k_m",
        temperature=0.3,
        num_gpu=20,      # Offload 20 layers to GPU (tune based on VRAM)
    )
    logger.info("LLM initialized successfully with Ollama")
except Exception as e:
    logger.error(f"Failed to initialize LLM: {e}")
    raise

# Initialize ChromaDB
embedding_function = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")

vectorstore = Chroma(
    collection_name="fiscal_data",
    persist_directory="data/chromadb",
    embedding_function=embedding_function
)
logger.info("ChromaDB initialized successfully")

memory = ConversationBufferMemory(return_messages=True)

# Ensure consistent language detection
DetectorFactory.seed = 0

# Initialize database
init_db()

# Pydantic model for structured LLM output
class ResponseOutput(BaseModel):
    type: str  # "greeting", "vat", "profit", "deadline", "general"
    name: str = None
    amount: float = None
    rate: float = None
    revenue: float = None
    expenses: float = None
    tax_rate: float = None
    result: float = None
    deadline: str = None
    response: str  # REQUIRED

# Enhanced prompt template with tools
response_parser = PydanticOutputParser(pydantic_object=ResponseOutput)
prompt_template = PromptTemplate(
    input_variables=["history", "input", "language", "context"],
    template="You are a professional tax assistant for Tunisia. Respond EXCLUSIVELY in {language}, using accurate, concise, and professional tax terminology. "
             "For Arabic, use conversational Tunisian Arabic with local terms (e.g., 'شنو' for 'what', 'كيفاش' for 'how', 'مرحبا' for greetings, 'تڤا' for VAT). "
             "Avoid formal Modern Standard Arabic. Do not switch languages. "
             "Use the provided context from the fiscal knowledge base to inform your response. "
             "Available tools: "
             "- `calculate_vat(amount, rate)`: Computes VAT as amount * rate / 100. "
             "- `get_tax_deadline(declaration_type, language)`: Returns deadline for declaration_type (e.g., 'vat_monthly', 'vat_quarterly', 'income_tax_annual'). "
             "Analyze the input to determine the response type: "
             "- If the user introduces their name (e.g., 'my name is Ahmed', 'je m'appelle Ahmed', 'people call me Ahmed', 'انا احمد'), extract the name and return a personalized greeting. "
             "- If the user asks for a VAT calculation (e.g., 'Calculate VAT for 100 TND at 7%'), use `calculate_vat` and return the result. "
             "- If the user asks for a net profit calculation (e.g., 'Calculate net profit for revenue 1000 TND and expenses 400 TND'), compute profit as (revenue - expenses) * (1 - tax_rate/100) and return the result. "
             "- If the user asks for a tax deadline (e.g., 'When is the VAT deadline?'), use `get_tax_deadline` and return the result. "
             "- For other queries, provide a clear tax-related response, incorporating context if relevant. "
             "Return a SINGLE JSON object with: {type} ('greeting', 'vat', 'profit', 'deadline', 'general'), {name} (if greeting), {amount}/{rate}/{result} (if VAT), {revenue}/{expenses}/{tax_rate}/{result} (if profit), {deadline} (if deadline), and {response} (the natural language answer in {language}). The {response} field is REQUIRED. "
             "Context from fiscal knowledge base:\n{context}\n"
             "Conversation history:\n{history}\nUser: {input}\n\n{format_instructions}",
    partial_variables={"format_instructions": response_parser.get_format_instructions()}
)

# Fallback prompt (no JSON requirement)
fallback_prompt_template = PromptTemplate(
    input_variables=["history", "input", "language", "context"],
    template="You are a professional tax assistant for Tunisia. Respond EXCLUSIVELY in {language}, using accurate, conversational tax terminology. "
             "For Arabic, use Tunisian Arabic with local terms (e.g., 'شنو' for 'what', 'كيفاش' for 'how', 'مرحبا' for greetings, 'تڤا' for VAT). "
             "Avoid formal Modern Standard Arabic. Do not switch languages. "
             "Use the provided context from the fiscal knowledge base to inform your response. "
             "Available tools: "
             "- `calculate_vat(amount, rate)`: Computes VAT as amount * rate / 100. "
             "- `get_tax_deadline(declaration_type, language)`: Returns deadline for declaration_type (e.g., 'vat_monthly', 'vat_quarterly', 'income_tax_annual'). "
             "For name introductions (e.g., 'my name is Ahmed', 'je m'appelle Ahmed', 'people call me Ahmed', 'انا احمد'), respond with a personalized greeting. "
             "- For VAT calculations, use `calculate_vat`. "
             "- For tax deadlines, use `get_tax_deadline`. "
             "- For other queries, provide a clear tax-related response. "
             "Context from fiscal knowledge base:\n{context}\n"
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

def load_dialogue_dataset():
    dataset_path = "C:/Users/oussa/Documents/WORK/AI/FiscAssistant/Backend/data/dialogue_dataset.json"
    if not Path(dataset_path).exists():
        logger.error(f"Dataset not found: {dataset_path}")
        return
    try:
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        documents = []
        for entry in data:
            entry_id = entry.get("id", "")
            category = entry.get("category", "")
            intent = entry.get("intent", "")
            for lang in ["en", "fr", "ar", "tn"]:
                question = entry.get("question", {}).get(lang, "")
                answer = entry.get("answer", {}).get(lang, "")
                if question and answer:
                    doc_content = f"Question: {question}\nAnswer: {answer}"
                    documents.append(
                        Document(
                            page_content=doc_content,
                            metadata={
                                "id": entry_id,
                                "language": lang,
                                "category": category,
                                "intent": intent
                            }
                        )
                    )
        vectorstore.add_documents(documents)
        logger.info(f"Loaded {len(documents)} documents into ChromaDB from dialogue_dataset.json")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")

# Call after Chroma initialization
load_dialogue_dataset()


def calculate_vat(amount: float, rate: float) -> float:
    """Calculate VAT: amount * rate / 100."""
    return amount * rate / 100

def get_tax_deadline(declaration_type: str, language: str = "en") -> str:
    """Return tax deadline for declaration_type."""
    deadlines = {
        "vat_monthly": "15th of next month",
        "vat_quarterly": "20th of April, July, October, January",
        "income_tax_annual": "April 30th"
    }
    translations = {
        "en": deadlines,
        "fr": {
            "vat_monthly": "15 du mois suivant",
            "vat_quarterly": "20 avril, juillet, octobre, janvier",
            "income_tax_annual": "30 avril"
        },
        "ar": {
            "vat_monthly": "15 من الشهر القادم",
            "vat_quarterly": "20 أفريل، جويلية، أكتوبر، جانفي",
            "income_tax_annual": "30 أفريل"
        }
    }
    return translations.get(language, deadlines).get(declaration_type, "Unknown deadline")    

# Precompute reference embeddings at startup
REFERENCE_PHRASES = {
    "en": [
        "Hello, how can I help you with taxes?",
        "What is the VAT rate in Tunisia?",
        "My name is John, I need tax advice."
    ],
    "fr": [
        "Bonjour, comment puis-je vous aider avec les impôts ?",
        "Quel est le taux de TVA en Tunisie ?",
        "Je m'appelle Marie, j'ai besoin de conseils fiscaux."
    ],
    "ar": [
        "مرحبا، كيف يمكنني مساعدتك في الضرائب؟",
        "ما هو معدل الضريبة على القيمة المضافة في تونس؟",
        "اسمي أحمد، أحتاج إلى نصيحة ضريبية."
    ],
    "tn": [
        "مرحبا، كيفاش نقدر نساعدك في الضرائب؟",
        "شنو معدل تڤا في تونس؟",
        "انا سميتي أحمد، محتاج نصيحة في الضرائب."
    ]
}

# Initialize SentenceTransformer for language detection
language_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
REFERENCE_EMBEDDINGS = {
    lang: language_model.encode(phrases, convert_to_tensor=True)
    for lang, phrases in REFERENCE_PHRASES.items()
}

def detect_language(message: str, specified_language: str) -> str:
    """
    Detect the language of the input message using embedding-based cosine similarity.
    Args:
        message: User input string.
        specified_language: Language specified by user (if any).
    Returns:
        Detected language code ('en', 'fr', 'ar', 'tn').
    """
    if specified_language in ["en", "fr", "ar", "tn"]:
        logger.info(f"Using specified language: {specified_language}")
        return specified_language

    try:
        # Encode user input
        input_embedding = language_model.encode([message], convert_to_tensor=True)
        
        # Compute cosine similarity with reference embeddings
        max_similarity = -1.0
        detected_language = "en"  # Default fallback
        for lang, ref_embeddings in REFERENCE_EMBEDDINGS.items():
            similarities = util.cos_sim(input_embedding, ref_embeddings)[0]
            avg_similarity = similarities.mean().item()
            logger.info(f"Language {lang} similarity: {avg_similarity:.4f}")
            if avg_similarity > max_similarity:
                max_similarity = avg_similarity
                detected_language = lang
        
        # Threshold to avoid misclassification
        if max_similarity < 0.3:  # Adjustable threshold
            logger.warning(f"Low similarity score ({max_similarity:.4f}), defaulting to 'en'")
            return "en"
        
        logger.info(f"Detected language: {detected_language} (score: {max_similarity:.4f})")
        return detected_language
    except Exception as e:
        logger.error(f"Embedding-based language detection error: {e}, defaulting to 'en'")
        return "en"

# Profile detection
def detect_profile(message: str) -> str:
    profiles = ["auto-entrepreneur", "société", "particulier"]
    for profile in profiles:
        if profile in message.lower():
            return profile
    return None

# Extract JSON from output
def extract_json_output(raw_output: str, user_input: str) -> dict:
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

    # Retrieve relevant context from ChromaDB
    context = ""
    try:
        results = vectorstore.similarity_search(user_input, k=3)
        context = "\n".join([f"Document {i+1}: {doc.page_content} (ID: {doc.metadata['id']})" for i, doc in enumerate(results)])
        logger.info(f"Retrieved context: {context}")
    except Exception as e:
        logger.error(f"ChromaDB retrieval error: {e}")
        context = "No relevant fiscal data found."

    # Save user input
    save_conversation(user_id, "user", user_input, language)

    # LLM response with structured output
    chain = prompt_template | llm | response_parser
    try:
        # Format prompt with context
        formatted_prompt = prompt_template.format(history=history_text, input=user_input, language=language, context=context)
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
                    "ar": f"تڤا لـ {output.amount} دينار تونسي بنسبة {output.rate}% هي {result:.2f} دينار تونسي."
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
            response = chain.invoke({"history": history_text, "input": user_input, "language": language, "context": context})
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