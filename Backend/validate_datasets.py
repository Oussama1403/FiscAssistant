import json

def validate_source_entries(dialogue_path, fiscal_path):
    with open(fiscal_path, 'r', encoding='utf-8') as f:
        fiscal_data = json.load(f)
    with open(dialogue_path, 'r', encoding='utf-8') as f:
        dialogue_data = json.load(f)

    # Assume fiscal_data is a list of dicts with 'id' field
    fiscal_ids = set(entry['id'] for entry in fiscal_data)
    missing_references = set()

    # Assume dialogue_data is a list of dicts with 'source_entry' field
    for entry in dialogue_data:
        source_id = entry.get('source_entry')
        if source_id not in fiscal_ids:
            missing_references.add(source_id)

    if missing_references:
        print("Missing references in dialogue_dataset.json:")
        for ref in missing_references:
            print(ref)
    else:
        print("All source_entry references are valid.")

def validate_unreferenced_fiscal_entries(dialogue_path, fiscal_path, temp_path=None):
    with open(fiscal_path, 'r', encoding='utf-8') as f:
        fiscal_data = json.load(f)
    with open(dialogue_path, 'r', encoding='utf-8') as f:
        dialogue_data = json.load(f)

    fiscal_ids = set(entry['id'] for entry in fiscal_data)
    referenced_ids = set(entry.get('source_entry') for entry in dialogue_data)

    unreferenced_ids = fiscal_ids - referenced_ids

    unreferenced_entries = [entry for entry in fiscal_data if entry['id'] in unreferenced_ids]

    if unreferenced_entries:
        print("Entries in fiscal_data.json not referenced by any dialogue_dataset.json entry:")
        for entry in unreferenced_entries:
            print(entry["id"])
        if temp_path:
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(unreferenced_entries, f, ensure_ascii=False, indent=2)
            print(f"Unreferenced entries written to {temp_path}")
    else:
        print("All fiscal_data entries are referenced in dialogue_dataset.json.")

if __name__ == "__main__":
    # Update the paths as needed
    dialogue_path = "data/dialogue_dataset.json"
    fiscal_path = "data/fiscal_data.json"
    temp_path = "temp.json"
    validate_source_entries(dialogue_path, fiscal_path)
    validate_unreferenced_fiscal_entries(dialogue_path, fiscal_path, temp_path)