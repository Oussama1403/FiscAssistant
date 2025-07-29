import json

# Load dialogue dataset
with open("C:\\Users\\oussa\\Documents\\WORK\\AI\\FiscAssistant\\Backend\\data\\dialogue_data.json", "r", encoding="utf-8") as f:
    dialogue_data = json.load(f)

# Check volume
print(f"Total pairs: {len(dialogue_data)}")

# Check category distribution
categories = {}
intents = {}
for entry in dialogue_data:
    cat = entry["category"]
    intent = entry["intent"]
    categories[cat] = categories.get(cat, 0) + 1
    intents[intent] = intents.get(intent, 0) + 1
    # Check schema
    for field in ["id", "category", "question", "answer", "source_entry", "intent"]:
        assert field in entry, f"Missing {field} in {entry['id']}"
    for lang in ["en", "fr", "ar", "tn"]:
        assert entry["question"][lang], f"Missing question.{lang} in {entry['id']}"
        assert entry["answer"][lang], f"Missing answer.{lang} in {entry['id']}"
print("Category distribution:", categories)
print("Intent distribution:", intents)

# Check source_entry against fiscal_data.json
fiscal_ids = set()
with open(r"C:\Users\oussa\Documents\WORK\AI\FiscAssistant\Backend\data\fiscal_data.json", "r", encoding="utf-8") as f:
    fiscal_data = json.load(f)
    for entry in fiscal_data:
        if "id" in entry:
            fiscal_ids.add(entry["id"])

for entry in dialogue_data:
    if entry["source_entry"]:
        assert entry["source_entry"] in fiscal_ids, f"Invalid source_entry {entry['source_entry']} in {entry['id']}"

print("Source entries validated")