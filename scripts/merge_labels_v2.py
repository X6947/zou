import json
import os

MERGE_MAP = {
    "ADE": "Other",
    "Amount": "Other",
    "Anatomy": "Other",
    "Class": "Disease",
    "Disease": "Disease",
    "Drug": "Drug",
    "Duration": "Other",
    "Frequency": "Other",
    "Level": "Other",
    "Method": "Treatment",
    "Operation": "Treatment",
    "Pathogenesis": "Other",
    "Reason": "Other",
    "Symptom": "Disease",
    "Test": "Test",
    "Test_Value": "Test",
    "Test_items": "Test",
    "Treatment": "Treatment"
}

def merge_tags(tags):
    new_tags = []
    for tag in tags:
        if tag == "O":
            new_tags.append("O")
        else:
            prefix, etype = tag.split("-", 1)
            new_type = MERGE_MAP.get(etype, etype)
            new_tags.append(f"{prefix}-{new_type}")
    return new_tags

SRC_DIR = "ner_data"
DST_DIR = "ner_data_v2"
os.makedirs(DST_DIR, exist_ok=True)

for split in ["train", "dev", "test"]:
    with open(os.path.join(SRC_DIR, f"{split}.json"), "r", encoding="utf-8") as f:
        data = json.load(f)
    new_data = [{"text": item["text"], "tags": merge_tags(item["tags"])} for item in data]
    with open(os.path.join(DST_DIR, f"{split}.json"), "w", encoding="utf-8") as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)
    print(f"{split}: {len(new_data)} sentences converted")

all_tags = set()
for split in ["train", "dev", "test"]:
    with open(os.path.join(DST_DIR, f"{split}.json"), "r", encoding="utf-8") as f:
        for item in json.load(f):
            all_tags.update(item["tags"])
label_list = ["O"] + sorted([t for t in all_tags if t != "O"])
with open(os.path.join(DST_DIR, "label_list.json"), "w", encoding="utf-8") as f:
    json.dump(label_list, f, ensure_ascii=False, indent=2)
print("新标签列表:", label_list)
print(f"新标签数量: {len(label_list)}")