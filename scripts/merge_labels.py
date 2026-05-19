import json
import os

# 定义合并映射：旧类别 -> 新类别
MERGE_MAP = {
    "ADE": "ADE",
    "Amount": "Measurement",
    "Anatomy": "Anatomy",
    "Class": "Disease",
    "Disease": "Disease",
    "Drug": "Drug",
    "Duration": "Measurement",
    "Frequency": "Measurement",
    "Level": "Measurement",
    "Method": "Treatment",
    "Operation": "Treatment",
    "Pathogenesis": "Anatomy",
    "Reason": "Reason",
    "Symptom": "Disease",
    "Test": "Test",
    "Test_Value": "Test",
    "Test_items": "Test",
    "Treatment": "Treatment"
}

def merge_tags(tags):
    """将旧的BIO标签列表转换为新的BIO标签列表"""
    new_tags = []
    for tag in tags:
        if tag == "O":
            new_tags.append("O")
        else:
            prefix, etype = tag.split("-", 1)
            if etype in MERGE_MAP:
                new_type = MERGE_MAP[etype]
                new_tags.append(f"{prefix}-{new_type}")
            else:
                # 未知类型保留原样（理论上不会发生）
                new_tags.append(tag)
    return new_tags

# 处理三个数据集
SRC_DIR = "ner_data"
DST_DIR = "ner_data_merged"
os.makedirs(DST_DIR, exist_ok=True)

for split in ["train", "dev", "test"]:
    with open(os.path.join(SRC_DIR, f"{split}.json"), "r", encoding="utf-8") as f:
        data = json.load(f)
    new_data = []
    for item in data:
        new_tags = merge_tags(item["tags"])
        new_data.append({
            "text": item["text"],
            "tags": new_tags
        })
    with open(os.path.join(DST_DIR, f"{split}.json"), "w", encoding="utf-8") as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)
    print(f"{split}: {len(new_data)} sentences converted.")

# 生成新的标签列表
all_tags = set()
for split in ["train", "dev", "test"]:
    with open(os.path.join(DST_DIR, f"{split}.json"), "r", encoding="utf-8") as f:
        for item in json.load(f):
            all_tags.update(item["tags"])

label_list = sorted([t for t in all_tags if t != "O"])
label_list = ["O"] + label_list
with open(os.path.join(DST_DIR, "label_list.json"), "w", encoding="utf-8") as f:
    json.dump(label_list, f, ensure_ascii=False, indent=2)
print(f"新标签列表: {label_list}")
print(f"新标签数量: {len(label_list)}")