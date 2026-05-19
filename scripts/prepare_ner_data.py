import json
import os
import random

# ========== 配置 ==========
DATA_DIR = r"D:\data\diakg"          # 41个原始JSON所在目录
OUTPUT_DIR = "ner_data"
SEED = 42
TRAIN_RATIO = 0.70
DEV_RATIO = 0.15

# 18类实体完整列表（来自 DiaKG 官方）
ENTITY_TYPES = [
    "Disease", "Class", "Drug", "Symptom", "Test_items", "Test_Value",
    "Treatment", "Pathogenesis", "Anatomy", "Method", "Frequency",
    "Duration", "Amount", "Operation", "ADE", "Reason", "Test",
    "Diet"
]

random.seed(SEED)

# 读取所有 JSON 文件名
all_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".json")]
random.shuffle(all_files)
n = len(all_files)
n_train = int(n * TRAIN_RATIO)
n_dev = int(n * DEV_RATIO)

train_files = all_files[:n_train]
dev_files = all_files[n_train:n_train + n_dev]
test_files = all_files[n_train + n_dev:]

print(f"训练集文件数: {len(train_files)}, 验证集: {len(dev_files)}, 测试集: {len(test_files)}")

def load_doc(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return [data]

def extract_sentence_samples(doc):
    """从一篇文档中提取所有句子的BIO标注样本"""
    samples = []
    for para in doc["paragraphs"]:
        for sent in para["sentences"]:
            text = sent["sentence"]
            # 初始化全O标签
            tags = ["O"] * len(text)
            # 按实体长度降序排序，避免嵌套时短实体覆盖长实体
            entities = sorted(sent["entities"], key=lambda e: e["end_idx"] - e["start_idx"], reverse=True)
            for ent in entities:
                start = ent["start_idx"]
                end = ent["end_idx"]
                etype = ent["entity_type"]
                # 检查位置是否已被更长的实体占用
                if all(tags[i] == "O" for i in range(start, end)):
                    tags[start] = f"B-{etype}"
                    for i in range(start + 1, end):
                        tags[i] = f"I-{etype}"
            samples.append({
                "text": text,
                "tags": tags
            })
    return samples

# 批量处理并保存
os.makedirs(OUTPUT_DIR, exist_ok=True)
for split_name, file_list in [("train", train_files), ("dev", dev_files), ("test", test_files)]:
    all_samples = []
    for fname in file_list:
        path = os.path.join(DATA_DIR, fname)
        docs = load_doc(path)
        for doc in docs:
            all_samples.extend(extract_sentence_samples(doc))
    out_path = os.path.join(OUTPUT_DIR, f"{split_name}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_samples, f, ensure_ascii=False, indent=2)
    print(f"{split_name}: {len(all_samples)} sentences → {out_path}")

# 统计所有标签
all_tags = set()
for split in ["train", "dev", "test"]:
    with open(os.path.join(OUTPUT_DIR, f"{split}.json"), "r", encoding="utf-8") as f:
        data = json.load(f)
    for item in data:
        all_tags.update(item["tags"])
label_list = sorted([t for t in all_tags if t != "O"])
label_list = ["O"] + label_list
with open(os.path.join(OUTPUT_DIR, "label_list.json"), "w", encoding="utf-8") as f:
    json.dump(label_list, f, ensure_ascii=False, indent=2)
print(f"标签数量: {len(label_list)}，已保存至 label_list.json")