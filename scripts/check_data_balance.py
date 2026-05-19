import json
from collections import Counter

def check_split(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    counter = Counter()
    for item in data:
        for tag in item["tags"]:
            if tag != "O":
                counter[tag.split("-", 1)[1]] += 1
    return counter

print("训练集:", check_split("ner_data_merged/train.json"))
print("验证集:", check_split("ner_data_merged/dev.json"))
print("测试集:", check_split("ner_data_merged/test.json"))