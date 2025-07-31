import json
import chromadb
from sentence_transformers import SentenceTransformer

# Initialize ChromaDB client
client = chromadb.PersistentClient(path="data/chromadb")
try:
    client.delete_collection("fiscal_data")
except Exception as e:
    print("Delete collection exception:", e)
collection = client.create_collection("fiscal_data")

# Load SentenceTransformer model
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# Load fiscal_data.json
with open("data/fiscal_data.json", "r", encoding="utf-8") as f:
    fiscal_data = json.load(f)

# Prepare documents for ChromaDB
documents = []
metadatas = []
ids = []
for idx, entry in enumerate(fiscal_data):
    print(f"Processing entry {idx}: {entry.get('id', 'Unknown ID')}")
    # Combine all descriptions for embedding
    text = " ".join([entry.get("description", {}).get(lang, "") for lang in ["en", "fr", "ar", "tn"]])
    documents.append(text)
    meta = {
        "category": entry.get("category", ""),
        "type": entry.get("type", ""),
        "applicability": json.dumps(entry.get("applicability", {})),
        "value": str(entry.get("value", "")),
        "details": json.dumps(entry.get("details", {})) if entry.get("details") else "",
        "source": entry.get("source", ""),
        "last_updated": entry.get("last_updated", "")
    }
    # Check for None in metadata fields
    for k, v in meta.items():
        if v is None:
            print(f"None value found in metadata field '{k}' for entry {entry.get('id', 'Unknown ID')}")
            raise ValueError(f"None value found in metadata field '{k}' for entry {entry.get('id', 'Unknown ID')}")
        if isinstance(v, dict) or isinstance(v, list):
            print(f"Non-string value in metadata field '{k}' for entry {entry.get('id', 'Unknown ID')}: {v}")
            raise ValueError(f"Non-string value in metadata field '{k}' for entry {entry.get('id', 'Unknown ID')}")
    metadatas.append(meta)
    ids.append(entry.get("id", ""))

# Check for mismatched lengths
n_docs = len(documents)
n_meta = len(metadatas)
n_ids = len(ids)
print(f"Length check: documents={n_docs}, metadatas={n_meta}, ids={n_ids}")
if not (n_docs == n_meta == n_ids):
    raise ValueError(f"Length mismatch: documents={n_docs}, metadatas={n_meta}, ids={n_ids}")

# Check for duplicate or empty IDs
from collections import Counter
dupes = [item for item, count in Counter(ids).items() if count > 1]
if dupes:
    print(f"Duplicate IDs found: {dupes}")
    raise ValueError(f"Duplicate IDs found: {dupes}")
empty_ids = [i for i, id_ in enumerate(ids) if not id_]
if empty_ids:
    print(f"Empty IDs at indices: {empty_ids}")
    raise ValueError(f"Empty IDs at indices: {empty_ids}")

# Generate embeddings
embeddings = model.encode(documents, batch_size=10)
print("Embeddings generated for", len(documents), "documents")
import numpy as np
print("Embedding shape for first doc:", np.shape(embeddings[0]) if len(embeddings) > 0 else "No embeddings")
# Check embeddings length and shape
if len(embeddings) != n_docs:
    raise ValueError(f"Embeddings length mismatch: embeddings={len(embeddings)}, documents={n_docs}")
if any(len(e) != len(embeddings[0]) for e in embeddings):
    print("Inconsistent embedding sizes detected!")
    raise ValueError("Inconsistent embedding sizes detected!")

# Print sample data
print("Sample document:", documents[0])
print("Sample embedding length:", len(embeddings[0]))
print("Sample metadata:", metadatas[0])
print("Sample id:", ids[0])

# Try adding one document with simple metadata
#print("Trying to add one document with simple metadata...")
#try:
#    collection.add(
#        documents=[documents[0]],
#        embeddings=[embeddings[0]],
#        metadatas=[{"foo": "bar"}],
#        ids=[ids[0]]
#    )
#    print("Added one document with simple metadata.")
#except Exception as e:
#    print("Error during collection.add() with simple metadata:", e)

# Try adding one document with real metadata
#print("Trying to add one document with real metadata...")
#try:
#    collection.add(
#        documents=[documents[0]],
#        embeddings=[embeddings[0]],
#        metadatas=[metadatas[0]],
#        ids=[ids[0]]
#    )
#    print("Added one document with real metadata.")
#except Exception as e:
#    print("Error during collection.add() with real metadata:", e)

# Try adding first 5 documents
#print("Trying to add first 5 documents...")
#try:
#    collection.add(
#        documents=documents[:5],
#        embeddings=embeddings[:5],
#        metadatas=metadatas[:5],
#        ids=ids[:5]
#    )
#    print("Added first 5 documents.")
#except Exception as e:
#    print("Error during collection.add() with first 5 documents:", e)

# Try adding all documents
print("Trying to add all documents...")
try:
    collection.add(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )
    print("Added all documents.")
except Exception as e:
    print("Error during collection.add() with all documents:", e)