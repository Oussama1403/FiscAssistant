from fastapi import FastAPI, Request, UploadFile, File, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_ollama import OllamaLLM
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import structlog
import pandas as pd
import pdfplumber
import joblib
import json
import re
import os
import torch
import numpy as np
from collections import Counter
from db.database import init_db, save_conversation, get_conversation_history, save_reminder, get_reminders

# Initialize logging
structlog.configure(processors=[structlog.processors.JSONRenderer()])
logger = structlog.get_logger()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Handle preflight OPTIONS requests
@app.options("/{rest_of_path:path}", include_in_schema=False)
async def preflight_handler(rest_of_path: str, request: Request):
    return {"status": "OK"}

# Initialize LLM and ChromaDB
llm = OllamaLLM(model="nous-hermes2:10.7b-solar-q3_k_m", temperature=0.3, num_gpu=10)
embedding_function = HuggingFaceEmbeddings(model_name=r"C:\Users\oussa\Documents\WORK\AI\FiscAssistant\Backend\models\paraphrase-multilingual-MiniLM-L12-v2")
vectorstore = Chroma(collection_name="fiscal_data", persist_directory="data/chromadb", embedding_function=embedding_function)
memory = ConversationBufferMemory(return_messages=True)
scheduler = AsyncIOScheduler()
scheduler.start()
init_db()

# Initialize DistilBERT for language detection
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-multilingual-cased")
language_model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-multilingual-cased", num_labels=4)
label_encoder = LabelEncoder()
label_encoder.fit(["en", "fr", "ar", "tn"])

# Preprocessing for language and intent detection
def preprocess_text(text):
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    dialect_map = {
        "كيفاش": "كيف",
        "شنية": "شنو",
        "تڤا": "تفا",
        "cava": "ça va",
        "commentcava": "comment ça va",
        "comen": "comment",
        "tva": "tva"
    }
    for k, v in dialect_map.items():
        text = text.replace(k, v)
    return text

# Character set heuristic for fallback
def detect_script(text):
    if any(ord(c) >= 0x0600 and ord(c) <= 0x06FF for c in text):  # Arabic script
        return "ar"
    if re.search(r"[çéèêëàâîïôûù]", text, re.IGNORECASE):  # French-specific characters
        return "fr"
    return "en"  # Default to English for Latin script

# Task 4.1: Language Detection
def train_language_classifier():
    try:
        dataset_path = "data/dialogue_dataset.json"
        if not os.path.exists(dataset_path):
            logger.error(f"Dataset file not found: {dataset_path}")
            return None
        with open(dataset_path, "r", encoding="utf-8") as f:
            dialogue_data = json.load(f)
        texts, labels = [], []
        for entry in dialogue_data:
            if not isinstance(entry, dict) or "question" not in entry or "intent" not in entry:
                logger.warning(f"Invalid entry format: {entry}")
                continue
            for lang in ["en", "fr", "ar", "tn"]:
                question = entry.get("question", {}).get(lang, "")
                if question:
                    texts.append(preprocess_text(question))
                    labels.append(lang)
                    if len(question.split()) < 5:
                        variations = [
                            question + "?",
                            question + " " + ("please" if lang == "en" else "s'il vous plaît" if lang == "fr" else "من فضلك" if lang == "ar" else "بربي" if lang == "tn" else ""),
                            question.replace(" ", ""),
                            question.replace("hi", "hey").replace("hello", "hallo").replace("tva", "vat") if lang == "en" else
                            question.replace("ça", "ca").replace("comment", "comen") if lang == "fr" else
                            question.replace("سلام", "salaam") if lang in ["ar", "tn"] else question
                        ]
                        for var in variations:
                            texts.append(preprocess_text(var))
                            labels.append(lang)
        if not texts:
            logger.error("No valid data for language classifier training")
            return None
        counts = Counter(labels)
        logger.info(f"Language sample counts: {counts}")
        min_samples = min(counts.values())
        texts_balanced, labels_balanced = [], []
        lang_counts = {lang: 0 for lang in ["en", "fr", "ar", "tn"]}
        for text, lang in zip(texts, labels):
            if lang_counts[lang] < min_samples:
                texts_balanced.append(text)
                labels_balanced.append(lang)
                lang_counts[lang] += 1
        y = label_encoder.transform(labels_balanced)
        encodings = tokenizer(texts_balanced, truncation=True, padding=True, max_length=32, return_tensors="pt")
        dataset = torch.utils.data.TensorDataset(
            encodings["input_ids"],
            encodings["attention_mask"],
            torch.tensor(y, dtype=torch.long)
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
        optimizer = torch.optim.AdamW(language_model.parameters(), lr=5e-5)
        language_model.train()
        for epoch in range(3):
            for batch in loader:
                optimizer.zero_grad()
                input_ids, attention_mask, labels = batch
                outputs = language_model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
        language_model.eval()
        language_model.save_pretrained("language_classifier")
        joblib.dump(label_encoder, "language_label_encoder.joblib")
        logger.info(f"Language classifier trained with {len(texts_balanced)} balanced samples")
        return language_model, label_encoder
    except Exception as e:
        logger.error(f"Language classifier training error: {e}")
        return None

language_classifier, label_encoder = train_language_classifier() or (None, None)
if language_classifier is None and os.path.exists("language_classifier"):
    try:
        language_classifier = DistilBertForSequenceClassification.from_pretrained("language_classifier")
        label_encoder = joblib.load("language_label_encoder.joblib")
        logger.info("Loaded existing language classifier")
    except Exception as e:
        logger.error(f"Failed to load language classifier: {e}")

# Embedding cache for frequent inputs
embedding_cache = {}

def detect_language(message: str, specified_language: str) -> str:
    if specified_language in ["en", "fr", "ar", "tn"]:
        logger.info(f"Using specified language: {specified_language}")
        return specified_language
    try:
        processed_input = preprocess_text(message)
        if processed_input in embedding_cache:
            logits = embedding_cache[processed_input]
        else:
            encodings = tokenizer(processed_input, truncation=True, padding=True, max_length=32, return_tensors="pt")
            with torch.no_grad():
                logits = language_model(**encodings).logits
            embedding_cache[processed_input] = logits
        probs = torch.softmax(logits, dim=-1).numpy()[0]
        predicted_idx = probs.argmax()
        predicted_lang = label_encoder.inverse_transform([predicted_idx])[0]
        confidence = probs[predicted_idx]
        probs_dict = dict(zip(label_encoder.classes_, probs))
        logger.info(f"Language probabilities: {probs_dict}")
        if confidence < 0.5:
            history = get_conversation_history("default_user")
            if history:
                recent_langs = [msg["language"] for msg in history[-3:]]
                lang_counts = Counter(recent_langs)
                history_lang = max(lang_counts, key=lang_counts.get)
                logger.info(f"Low confidence ({confidence:.4f}), using history language: {history_lang}")
                return history_lang
            script_lang = detect_script(message)
            logger.info(f"Low confidence ({confidence:.4f}), defaulting to script-based '{script_lang}'")
            return script_lang
        logger.info(f"Detected language: {predicted_lang} (confidence: {confidence:.4f})")
        return predicted_lang
    except Exception as e:
        logger.error(f"Language detection error: {e}, defaulting to 'en'")
        return "en"

# Task 4.2: Intent Classification
def train_intent_classifier():
    try:
        dataset_path = "data/dialogue_dataset.json"
        if not os.path.exists(dataset_path):
            logger.error(f"Dataset file not found: {dataset_path}")
            return None
        with open(dataset_path, "r", encoding="utf-8") as f:
            dialogue_data = json.load(f)
        X, y = [], []
        for entry in dialogue_data:
            if not isinstance(entry, dict) or "question" not in entry or "intent" not in entry:
                logger.warning(f"Invalid entry format: {entry}")
                continue
            intent = entry.get("intent", "")
            if not intent:
                logger.warning(f"Empty intent in entry: {entry}")
                continue
            for lang in ["en", "fr", "ar", "tn"]:
                question = entry.get("question", {}).get(lang, "")
                if question:
                    X.append(embedding_function.embed_query(preprocess_text(question)))
                    y.append(intent)
                    if len(question.split()) < 5 and intent == "greeting":
                        variations = [
                            question + "?",
                            question + " " + ("please" if lang == "en" else "s'il vous plaît" if lang == "fr" else "من فضلك" if lang == "ar" else "بربي" if lang == "tn" else ""),
                            question.replace(" ", ""),
                            question.replace("hi", "hey").replace("hello", "hallo") if lang == "en" else question
                        ]
                        for var in variations:
                            X.append(embedding_function.embed_query(preprocess_text(var)))
                            y.append(intent)
        if not X:
            logger.error("No data for intent classifier training")
            return None
        clf = LogisticRegression(max_iter=2000, solver='lbfgs', C=1.0)
        clf.fit(X, y)
        joblib.dump(clf, "intent_classifier.joblib")
        logger.info(f"Intent classifier trained with {len(X)} samples")
        return clf
    except Exception as e:
        logger.error(f"Intent classifier training error: {e}")
        return None

intent_classifier = train_intent_classifier()
if intent_classifier is None and os.path.exists("intent_classifier.joblib"):
    try:
        intent_classifier = joblib.load("intent_classifier.joblib")
        logger.info("Loaded existing intent classifier")
    except Exception as e:
        logger.error(f"Failed to load intent classifier: {e}")

def detect_intent(message: str) -> str:
    try:
        processed_input = preprocess_text(message)
        embedding = embedding_function.embed_query(processed_input)
        if intent_classifier:
            intent = intent_classifier.predict([embedding])[0]
            logger.info(f"Detected intent: {intent}")
            return intent
        else:
            logger.error("Intent classifier not available, defaulting to 'general'")
            return "general"
    except Exception as e:
        logger.error(f"Intent detection error: {e}, defaulting to 'general'")
        return "general"

# Pydantic models
class ChatInput(BaseModel):
    message: str
    user_id: str = "default_user"
    language: str = None

class ReminderInput(BaseModel):
    declaration_type: str
    deadline: str
    user_id: str = "default_user"

# Load fiscal data
def load_fiscal_data():
    try:
        with open("data/fiscal_data.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load fiscal_data.json: {e}")
        return []

FISCAL_DATA = load_fiscal_data()

# Calculation functions
def calculate_vat(amount: float, rate: float) -> float:
    return amount * rate / 100

def get_tax_deadline(declaration_type: str, language: str) -> str:
    for entry in FISCAL_DATA:
        if entry.get("category") == "deadline" and entry.get("type").lower() == declaration_type.lower():
            return entry.get("description", {}).get(language, entry.get("value", "Unknown deadline"))
    return {
        "en": "Unknown deadline",
        "fr": "Date limite inconnue",
        "ar": "موعد نهائي غير معروف",
        "tn": "آخر أجل مش معروف"
    }[language]

# Extract numerical values from input
def extract_numbers(text: str) -> tuple:
    numbers = re.findall(r'\d+\.?\d*', text)
    amount = float(numbers[0]) if numbers else None
    rate = float(numbers[1]) if len(numbers) > 1 else 19.0  # Default VAT rate for Tunisia
    return amount, rate

# Validate response language
def validate_response_language(response: str, expected_lang: str) -> bool:
    if expected_lang in ["ar", "tn"]:
        return any(ord(c) >= 0x0600 and ord(c) <= 0x06FF for c in response) and \
               bool(re.search(r"\b(تڤا|ضريب|كيفاش)\b" if expected_lang == "tn" else r"\b(ضريبة|مرحب)\b", response))
    if expected_lang == "fr":
        return bool(re.search(r"[çéèêëàâîïôûù]|vous|tva", response.lower()))
    # Allow French loanwords in English for Tunisian tax terms
    return bool(re.search(r"\b(tax|fiscal|income|deduction|vat)\b", response.lower())) or \
           all(ord(c) < 128 or c in "éèêëàâîïôûù" for c in response if c.isalpha())

# Validate response intent
def validate_response_intent(response: str, intent: str) -> bool:
    if intent == "calculate_tax":
        return bool(re.search(r"\b(tva|vat)\b", response.lower())) or any(str(n) in response for n in range(10))
    if intent == "request_form_guidance":
        return bool(re.search(r"\b(formulaire|form|declaration|tax|fiscal|income|deduction)\b", response.lower()))
    if intent == "greeting":
        return bool(re.search(r"\b(hello|bonjour|مرحب|سلام)\b", response.lower()))
    return True  # General intent

# Enforce language post-processing
def enforce_language(response: str, expected_lang: str) -> bool:
    if expected_lang in ["ar", "tn"]:
        return any(ord(c) >= 0x0600 and ord(c) <= 0x06FF for c in response)
    if expected_lang == "fr":
        return bool(re.search(r"[çéèêëàâîïôûù]|vous|tva", response.lower()))
    # English: Allow loanwords but penalize French-specific terms
    return bool(re.search(r"\b(tax|fiscal|income|deduction|vat)\b", response.lower())) and \
           not bool(re.search(r"\b(vous|tva|bonjour)\b", response.lower()))

# Filter context by language
def filter_context_by_language(context: list, language: str) -> str:
    filtered = []
    for doc in context:
        if doc.metadata.get("language") == language:
            content = doc.page_content
            if content.startswith("Question:"):
                content = content.split("Answer:")[1].strip() if "Answer:" in content else content
            filtered.append(content)
    return "\n".join(filtered) if filtered else context[0].page_content  # Fallback to first document

# Prompt template with CoT, language lock, and jailbreak examples
prompt_template = PromptTemplate(
    input_variables=["input", "language", "context", "intent", "amount", "rate"],
    template="""### Role
You are a tax assistant for Tunisia. Respond strictly in {language}. For 'tn', use conversational Tunisian Arabic (e.g., 'تڤا', 'كيفاش'). Do NOT respond in any other language, especially French or Arabic if {language} is 'en', or English if {language} is 'fr'. Wrong-language responses are unacceptable.

### Instructions
Follow these steps to generate the response:
1. **Identify Language**: The response must be in {language}. Verify the input and context are in {language}.
2. **Determine Intent**: The intent is {intent}.
3. **Use Context**: Base your response on the provided context: {context}.
4. **Handle Specific Intents**:
   - For 'greeting', provide a short, friendly greeting in {language}.
   - For 'calculate_tax', compute VAT as {amount} * {rate} / 100 and format in {language}.
   - For 'request_form_guidance', provide fiscal guidance based on the context.
   - For other intents, answer based on the context and input.
5. **Ask for Clarification**: If the input lacks details, ask for clarification in {language}.
6. **Penalize Wrong Language**: Do NOT use French words (e.g., 'vous', 'tva') in English responses, or English words in French responses.

### Jailbreak Examples
1. **Incorrect Language (en requested, fr given)**:
   Input: I need help with fiscal information
   Incorrect Response: Bonjour ! Je suis là pour vous aider...
   Correct Response: To provide fiscal information, you need to supply your Fiscal Identifier / National ID Number...
2. **Incorrect Language (fr requested, ar given)**:
   Input: Comment obtenir le formulaire de déclaration ?
   Incorrect Response: مرحبًا، يمكنك الحصول على استمارة...
   Correct Response: Vous pouvez obtenir le formulaire de déclaration des revenus en Tunisie sur le site web du ministère des Finances tunisien...

### Few-Shot Examples
1. **Greeting (en)**:
   Input: Hello
   Response: Hello! I'm your tax assistant for Tunisia. How can I help you today?
2. **Calculate Tax (en)**:
   Input: Calculate VAT 1000
   Context: Question: How much is VAT for goods worth 500 TND at 19%? Answer: VAT = 500 × 19 / 100 = 95 TND.
   Response: The VAT calculation for 1000 TND at 19% is 190 TND.
3. **Request Form Guidance (en)**:
   Input: I need help with fiscal information
   Context: You need to provide your Fiscal Identifier / National ID Number, Fiscal Year, Net Global Income, Income by Category, Tax Due, Source Deductions, and optionally Common Deductions and Provisional Payments.
   Response: To provide fiscal information, you need to supply your Fiscal Identifier / National ID Number, Fiscal Year, Net Global Income, Income by Category (e.g., salary, business income, interest, dividends), Tax Due, Source Deductions, and optionally Common Deductions and Provisional Payments. Please provide these details for accurate assistance.
4. **Request Form Guidance (fr)**:
   Input: Comment obtenir le formulaire de déclaration ?
   Context: Vous pouvez obtenir le formulaire de déclaration des revenus en Tunisie sur le site web du ministère des Finances tunisien ou en demandant un exemplaire à votre banque locale.
   Response: Vous pouvez obtenir le formulaire de déclaration des revenus en Tunisie sur le site web du ministère des Finances tunisien ou en demandant un exemplaire à votre banque locale.
5. **Calculate Tax (tn)**:
   Input: كيفاش نحسب تڤا 1000؟
   Context: السؤال: كم تڤا لسلع بقيمة 500 دينار بـ 19%؟ الجواب: تڤا = 500 × 19 / 100 = 95 دينار.
   Response: حساب التڤا لـ 1000 دينار بـ 19% يعمل 190 دينار.

### Input
{input}

### Context
{context}
"""
)

# Initialize ChromaDB with datasets
def initialize_vectorstore():
    documents = []
    try:
        with open("data/fiscal_data.json", "r", encoding="utf-8") as f:
            fiscal_data = json.load(f)
            for entry in fiscal_data:
                for lang in ["en", "fr", "ar", "tn"]:
                    desc = entry.get("description", {}).get(lang, "")
                    if desc:
                        documents.append(Document(
                            page_content=desc,
                            metadata={"id": entry["id"], "category": entry["category"], "type": entry["type"], "language": lang, "intent": "none"}
                        ))
        with open("data/dialogue_dataset.json", "r", encoding="utf-8") as f:
            dialogue_data = json.load(f)
            for entry in dialogue_data:
                for lang in ["en", "fr", "ar", "tn"]:
                    question = entry.get("question", {}).get(lang, "")
                    answer = entry.get("answer", {}).get(lang, "")
                    if question and answer:
                        documents.append(Document(
                            page_content=f"Question: {question}\nAnswer: {answer}",
                            metadata={"id": entry["id"], "category": entry["category"], "intent": entry["intent"], "language": lang}
                        ))
        vectorstore.add_documents(documents)
        logger.info(f"Loaded {len(documents)} documents into ChromaDB")
    except Exception as e:
        logger.error(f"Vectorstore initialization error: {e}")

initialize_vectorstore()

# API endpoints
@app.post("/chat")
async def chat(input: ChatInput):
    language = detect_language(input.message, input.language)
    intent = detect_intent(input.message)
    amount, rate = extract_numbers(input.message) if intent == "calculate_tax" else (None, 19.0)

    # Use strict language and intent filter
    where_filter = {
        "$and": [
            {"language": language},
            {"$or": [
                {"intent": intent},
                {"intent": "none"}
            ]}
        ]
    }
    context = vectorstore.similarity_search(input.message, k=3, filter=where_filter)
    context_text = filter_context_by_language(context, language)
    logger.info(f"Filtered context for language {language}: {context_text}")

    # Handle clarification for calculate_tax
    if intent == "calculate_tax" and amount is None:
        clarification = {
            "en": "Please provide the amount for VAT calculation.",
            "fr": "Veuillez fournir le montant pour le calcul de la TVA.",
            "ar": "يرجى تقديم المبلغ لحساب الضريبة على القيمة المضافة.",
            "tn": "عطيني المبلغ باش نحسب التڤا."
        }
        response = clarification.get(language, clarification["en"])
        save_conversation(input.user_id, "assistant", response, language)
        memory.save_context({"input": input.message}, {"output": response})
        return {"response": response}

    # Handle greetings
    if intent == "greeting":
        greeting_responses = {
            "en": "Hello! I'm your tax assistant for Tunisia. How can I help you today?",
            "fr": "Bonjour ! Je suis votre assistant fiscal pour la Tunisie. Comment puis-je vous aider aujourd’hui ?",
            "ar": "مرحبًا! أنا مساعدك الضريبي في تونس. كيف يمكنني مساعدتك اليوم؟",
            "tn": "سلام! أنا مساعدك الضريبي في تونس. كيفاش نقدر نساعدك اليوم؟"
        }
        response = greeting_responses.get(language, greeting_responses["en"])
    elif intent == "calculate_tax":
        # Direct VAT calculation
        vat = calculate_vat(amount, rate)
        response_templates = {
            "en": f"The VAT calculation for {amount} TND at {rate}% is {vat} TND.",
            "fr": f"Le calcul de la TVA pour {amount} TND à {rate}% est {vat} TND.",
            "ar": f"حساب الضريبة على القيمة المضافة لـ {amount} دينار بمعدل {rate}% هو {vat} دينار.",
            "tn": f"حساب التڤا لـ {amount} دينار بـ {rate}% يعمل {vat} دينار."
        }
        response = response_templates.get(language, response_templates["en"])
    else:
        # Generate response with LLM
        prompt = prompt_template.format(
            input=input.message,
            language=language,
            context=context_text,
            intent=intent,
            amount=amount if amount else "unknown",
            rate=rate
        )
        logger.info(f"LLM prompt: {prompt}")
        max_retries = 3
        for attempt in range(max_retries):
            response = llm.invoke(prompt)
            logger.info(f"LLM response (attempt {attempt + 1}): {response}")
            lang_valid = validate_response_language(response, language)
            intent_valid = validate_response_intent(response, intent)
            enforce_valid = enforce_language(response, language)
            if lang_valid and intent_valid and enforce_valid:
                break
            logger.warning(f"Response validation failed (language_valid: {lang_valid}, intent_valid: {intent_valid}, enforce_valid: {enforce_valid}, attempt {attempt + 1}/{max_retries}).")
            prompt = f"{prompt}\n### Strict Instruction\nRespond ONLY in {language}. Address the intent '{intent}' using the context. Do NOT use French or Arabic if {language} is 'en'."
        if not (lang_valid and intent_valid and enforce_valid):
            logger.error(f"Response validation failed after {max_retries} retries. Rewriting prompt.")
            prompt = prompt_template.format(
                input=input.message,
                language=language,
                context=context_text,
                intent=intent,
                amount=amount if amount else "unknown",
                rate=rate
            ) + f"\n### Strict Instruction\nRespond ONLY in {language}. Address the intent '{intent}' using the context: {context_text}. Do NOT use French or Arabic if {language} is 'en'."
            response = llm.invoke(prompt)
            logger.info(f"Rewritten prompt response: {response}")
            enforce_valid = enforce_language(response, language)
            if not enforce_valid:
                logger.error(f"Rewritten response still in wrong language. Using fallback.")
                response = {
                    "en": "To provide fiscal information, please supply your Fiscal Identifier / National ID Number, Fiscal Year, Net Global Income, Income by Category, Tax Due, Source Deductions, and optionally Common Deductions and Provisional Payments.",
                    "fr": "Pour fournir des informations fiscales, veuillez fournir votre identifiant fiscal / numéro de carte d’identité nationale, l’exercice fiscal, le revenu global net, les revenus par catégorie, l’impôt dû, les retenues à la source, et éventuellement les déductions communes et les acomptes provisionnels.",
                    "ar": "لتزويد المعلومات الضريبية، يرجى تقديم معرفك الضريبي / رقم بطاقة الهوية الوطنية، السنة المالية، الدخل الإجمالي الصافي، الدخل حسب الفئة، الضريبة المستحقة، الخصومات عند المصدر، واختياريًا الخصومات المشتركة والدفعات المؤقتة.",
                    "tn": "باش نعطيك معلومات على الضرائب، عطيني المعرف الضريبي / نوميرو دالكارط ناسيونال، العام المالي، المدخول الصافي الكلي، المدخول حسب الصنف، الضريبة إلي لازم تدفعها، الخصومات عند المصدر، وإذا تحب الخصومات المشتركة والدفعات المؤقتة."
                }[language]

    save_conversation(input.user_id, "assistant", response, language)
    memory.save_context({"input": input.message}, {"output": response})
    return {"response": response}

@app.post("/set_reminder")
async def set_reminder(input: ReminderInput):
    save_reminder(input.user_id, input.declaration_type, input.deadline)
    scheduler.add_job(
        lambda: logger.info(f"Reminder: {input.declaration_type} due on {input.deadline} for user {input.user_id}"),
        'date',
        run_date=input.deadline,
        args=[input.user_id, input.declaration_type]
    )
    return {"message": f"Reminder set for {input.declaration_type}"}

@app.get("/reminders/{user_id}")
async def get_reminders_endpoint(user_id: str):
    return {"reminders": get_reminders(user_id)}

# Debug endpoint for LLM
@app.post("/debug_llm")
async def debug_llm(input: ChatInput):
    language = detect_language(input.message, input.language)
    intent = detect_intent(input.message)
    where_filter = {
        "$and": [
            {"language": language},
            {"$or": [
                {"intent": intent},
                {"intent": "none"}
            ]}
        ]
    }
    context = vectorstore.similarity_search(input.message, k=3, filter=where_filter)
    context_text = filter_context_by_language(context, language)
    context_metadata = [doc.metadata for doc in context]
    amount, rate = extract_numbers(input.message) if intent == "calculate_tax" else (None, 19.0)
    prompt = prompt_template.format(
        input=input.message,
        language=language,
        context=context_text,
        intent=intent,
        amount=amount if amount else "unknown",
        rate=rate
    )
    response = llm.invoke(prompt)
    lang_valid = validate_response_language(response, language)
    intent_valid = validate_response_intent(response, intent)
    enforce_valid = enforce_language(response, language)
    return {
        "prompt": prompt,
        "response": response,
        "language": language,
        "intent": intent,
        "context_text": context_text,
        "context_metadata": context_metadata,
        "language_valid": lang_valid,
        "intent_valid": intent_valid,
        "enforce_language_valid": enforce_valid
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)