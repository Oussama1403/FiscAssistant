import json

INPUT_FILE = "C:\\Users\\oussa\\Documents\\WORK\\AI\\FiscAssistant\\Backend\\data\\dialogue_data.json"
OUTPUT_FILE = "C:\\Users\\oussa\\Documents\\WORK\\AI\\FiscAssistant\\Backend\\data\\dialogue_data_fixed.json"

def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    for idx, entry in enumerate(data, 1):
        entry["id"] = f"dialogue_{idx:03d}"

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Fixed IDs and wrote {len(data)} records to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()