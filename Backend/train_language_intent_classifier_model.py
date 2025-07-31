import json
import os
import torch
import structlog
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import joblib
from langchain_huggingface import HuggingFaceEmbeddings
from collections import Counter
import re
import nlpaug.augmenter.word as naw
import random

# Initialize logging
structlog.configure(processors=[structlog.processors.JSONRenderer()])
logger = structlog.get_logger()

# Initialize embedding function for fallback (optional)
embedding_function = HuggingFaceEmbeddings(model_name=r"C:\Users\oussa\Documents\WORK\AI\FiscAssistant\Backend\models\paraphrase-multilingual-MiniLM-L12-v2")

import nltk

# Ensure required NLTK resources are available for nlpaug
for resource in [
    'averaged_perceptron_tagger_eng',  # For English synonym augmentation
    'averaged_perceptron_tagger',      # For French synonym augmentation
    'wordnet',                         # For synonyms
    'omw-1.4'                          # For multilingual WordNet
]:
    try:
        nltk.data.find(f'taggers/{resource}') if "tagger" in resource else nltk.data.find(f'corpora/{resource}')
    except LookupError:
        nltk.download(resource)

# Preprocessing for language and intent detection
def preprocess_text(text):
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    dialect_map = {
        "كيفاش": "كيف",
        "شنية": "شنو",
        "تڤا": "تفا",
        "بربي": "بربي",
        "cava": "ça va",
        "commentcava": "comment ça va",
        "comen": "comment",
        "tva": "tva"
    }
    for k, v in dialect_map.items():
        text = text.replace(k, v)
    return text

# Data augmentation for intent classifier
def augment_text(text, lang, augmenter):
    try:
        if lang == "en":
            return augmenter.augment(text)[0]
        elif lang == "fr":
            return augmenter.augment(text)[0]
        elif lang in ["ar", "tn"]:
            # Limited augmentation for Arabic/Tunisian due to model constraints
            variations = [
                text + "؟",
                text + " من فضلك" if lang == "ar" else text + " بربي" if lang == "tn" else text,
                text.replace("سلام", "salaam")
            ]
            return random.choice(variations)
        return text
    except Exception as e:
        logger.warning(f"Augmentation failed for text: {text}, error: {e}")
        return text

# Train language classifier (unchanged, as it performs well)
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
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-multilingual-cased")
        language_model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-multilingual-cased", num_labels=4)
        label_encoder = LabelEncoder()
        label_encoder.fit(["en", "fr", "ar", "tn"])
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
        tokenizer.save_pretrained("language_classifier")
        joblib.dump(label_encoder, "language_label_encoder.joblib")
        logger.info(f"Language classifier trained with {len(texts_balanced)} balanced samples")
        return language_model, label_encoder
    except Exception as e:
        logger.error(f"Language classifier training error: {e}")
        return None

# Train intent classifier with transformer-based model and advanced augmentation
def train_intent_classifier():
    try:
        dataset_path = "data/dialogue_dataset.json"
        if not os.path.exists(dataset_path):
            logger.error(f"Dataset file not found: {dataset_path}")
            return None
        with open(dataset_path, "r", encoding="utf-8") as f:
            dialogue_data = json.load(f)

        texts, labels = [], []
        # Initialize augmenter for English and French
        augmenter = naw.SynonymAug(aug_src='wordnet')  # For English
        fr_augmenter = naw.SynonymAug(aug_src='wordnet', lang='fra')  # For French

        # Collect and augment data
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
                    # Original text
                    texts.append(preprocess_text(question))
                    labels.append(intent)
                    # Basic augmentation (as in original)
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
                            labels.append(intent)
                    # Advanced augmentation
                    for _ in range(3):  # Generate 3 augmented versions per question
                        aug_text = augment_text(question, lang, augmenter if lang == "en" else fr_augmenter if lang == "fr" else None)
                        texts.append(preprocess_text(aug_text))
                        labels.append(intent)

        if not texts:
            logger.error("No data for intent classifier training")
            return None

        # Balance dataset by intent
        counts = Counter(labels)
        logger.info(f"Intent sample counts: {counts}")
        min_samples = min(counts.values())
        texts_balanced, labels_balanced = [], []
        intent_counts = {intent: 0 for intent in counts.keys()}
        for text, intent in zip(texts, labels):
            if intent_counts[intent] < min_samples:
                texts_balanced.append(text)
                labels_balanced.append(intent)
                intent_counts[intent] += 1

        # Initialize transformer model
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-multilingual-cased")
        intent_model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-multilingual-cased",
            num_labels=len(set(labels_balanced))
        )
        label_encoder = LabelEncoder()
        label_encoder.fit(list(set(labels_balanced)))
        y = label_encoder.transform(labels_balanced)

        # Prepare dataset
        encodings = tokenizer(texts_balanced, truncation=True, padding=True, max_length=32, return_tensors="pt")
        dataset = torch.utils.data.TensorDataset(
            encodings["input_ids"],
            encodings["attention_mask"],
            torch.tensor(y, dtype=torch.long)
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

        # Train model
        optimizer = torch.optim.AdamW(intent_model.parameters(), lr=5e-5)
        intent_model.train()
        for epoch in range(3):
            for batch in loader:
                optimizer.zero_grad()
                input_ids, attention_mask, labels = batch
                outputs = intent_model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
        intent_model.eval()

        # Save model and encoder
        intent_model.save_pretrained("intent_classifier")
        tokenizer.save_pretrained("intent_classifier")
        joblib.dump(label_encoder, "intent_label_encoder.joblib")
        logger.info(f"Intent classifier trained with {len(texts_balanced)} balanced samples")
        return intent_model, label_encoder
    except Exception as e:
        logger.error(f"Intent classifier training error: {e}")
        return None

if __name__ == "__main__":
    language_model, language_label_encoder = train_language_classifier()
    intent_model, intent_label_encoder = train_intent_classifier()
    if language_model and language_label_encoder and intent_model and intent_label_encoder:
        logger.info("Training completed successfully. Models saved.")
    else:
        logger.error("Training failed. Check logs for details.")