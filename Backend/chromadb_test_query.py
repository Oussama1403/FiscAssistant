import json
import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

#load collection
client = chromadb.PersistentClient(path="data/chromadb")
collection = client.get_collection("fiscal_data")
print("Collection loaded:", collection.name)
try:
    count = collection.count()
    print("Collection count:", count)
except Exception as e:
    print("Error getting collection count:", e)

# Test retrieval
queries = [
    "What’s the TVA rate for restaurants?",
    "كيفاش نحسب تڤا؟",
    "When is the CIT declaration due?"
]
for query in queries:
    query_embedding = model.encode([query])
    print(f"Query: {query}")
    try:
        results = collection.query(query_embeddings=query_embedding, n_results=3)
        print("Raw results:", results)
        # Print matched documents and metadata
        for i, doc in enumerate(results['documents'][0]):
            print(f"Result {i+1}:")
            print("Document:", doc)
            print("Metadata:", results['metadatas'][0][i])
            print("ID:", results['ids'][0][i])
    except Exception as e:
        print("Error during query:", e)