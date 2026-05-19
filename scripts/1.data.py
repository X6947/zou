import json
import os
import random

# 配置
DATA_DIR = r"D:\data\diakg"        # 你的41个JSON所在目录
OUTPUT_DIR = "casrel_data"         # 输出目录
TRAIN_RATIO = 0.70
DEV_RATIO = 0.15
TEST_RATIO = 0.15
SEED = 42

random.seed(SEED)

# 1. 读取所有JSON文件
all_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".json")]
print(f"共找到 {len(all_files)} 个JSON文件")

# 2. 随机打乱并划分
random.shuffle(all_files)
n = len(all_files)
n_train = int(n * TRAIN_RATIO)
n_dev = int(n * DEV_RATIO)

train_files = all_files[:n_train]
dev_files = all_files[n_train:n_train + n_dev]
test_files = all_files[n_train + n_dev:]

print(f"训练集: {len(train_files)} 篇, 验证集: {len(dev_files)} 篇, 测试集: {len(test_files)} 篇")

# 3. 转换函数（与之前一样）
def load_and_convert(file_path):
    """加载单个DiaKG JSON，返回CasRel格式的样本列表"""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # 确保数据是文档格式（可能有顶层是列表或单个对象）
    if isinstance(data, list):
        docs = data
    else:
        docs = [data]

    samples = []
    for doc in docs:
        # 构建全局 entity_id -> entity_name 映射
        id2entity = {}
        for para in doc["paragraphs"]:
            for sent in para["sentences"]:
                for ent in sent["entities"]:
                    id2entity[ent["entity_id"]] = ent["entity"]

        # 遍历句子，提取三元组
        for para in doc["paragraphs"]:
            for sent in para["sentences"]:
                spo_list = []
                for rel in sent["relations"]:
                    head_id = rel["head_entity_id"]
                    tail_id = rel["tail_entity_id"]
                    if head_id in id2entity and tail_id in id2entity:
                        h = id2entity[head_id]
                        t = id2entity[tail_id]
                        r = rel["relation_type"]
                        spo_list.append((h, r, t))
                if spo_list:
                    samples.append({
                        "text": sent["sentence"],
                        "spo_list": spo_list
                    })
    return samples

# 4. 批量处理并保存
os.makedirs(OUTPUT_DIR, exist_ok=True)

for split_name, file_list in [("train", train_files), ("dev", dev_files), ("test", test_files)]:
    all_samples = []
    for fname in file_list:
        path = os.path.join(DATA_DIR, fname)
        all_samples.extend(load_and_convert(path))
    out_path = os.path.join(OUTPUT_DIR, f"{split_name}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_samples, f, ensure_ascii=False, indent=2)
    print(f"✅ {split_name}: {len(all_samples)} 条句子 → {out_path}")

print("全部完成！")
# 收集所有关系类型
all_relations = set()
for split_name in ["train", "dev", "test"]:
    with open(os.path.join(OUTPUT_DIR, f"{split_name}.json"), "r", encoding="utf-8") as f:
        data = json.load(f)
    for item in data:
        for spo in item["spo_list"]:
            all_relations.add(spo[1])  # spo是(h, r, t)
print("关系类型:", sorted(all_relations))
with open("rel.json", "w", encoding="utf-8") as f:
    json.dump(sorted(list(all_relations)), f, ensure_ascii=False, indent=2)
print("rel.json 已保存")
