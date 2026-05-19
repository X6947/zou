import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

import json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertForTokenClassification
from torch.optim import AdamW
from seqeval.metrics import classification_report, f1_score
from tqdm import tqdm

# ========== 配置 ==========
MODEL_PATH = "bert-base-chinese"
DATA_DIR = "ner_data"
TRAIN_PATH = f"{DATA_DIR}/train.json"
DEV_PATH = f"{DATA_DIR}/dev.json"
LABEL_PATH = f"{DATA_DIR}/label_list.json"

with open(LABEL_PATH, "r", encoding="utf-8") as f:
    LABEL_LIST = json.load(f)
NUM_LABELS = len(LABEL_LIST)
label2id = {l: i for i, l in enumerate(LABEL_LIST)}
id2label = {i: l for l, i in label2id.items()}

MAX_LEN = 256
BATCH_SIZE = 8
EPOCHS = 15
LR = 3e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizerFast.from_pretrained(MODEL_PATH, local_files_only=True)

# ========== 数据集 ==========
class NERDataset(Dataset):
    def __init__(self, path):
        with open(path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    texts = [item["text"] for item in batch]
    char_tags_list = [item["tags"] for item in batch]

    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt",
        return_offsets_mapping=True,
        is_split_into_words=False
    )

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    batch_size, seq_len = input_ids.shape
    labels = torch.full((batch_size, seq_len), -100, dtype=torch.long)

    for i in range(batch_size):
        word_ids = encoded.word_ids(batch_index=i)
        for j, word_idx in enumerate(word_ids):
            if word_idx is None:
                continue
            char_tag = char_tags_list[i][word_idx]
            label_id = label2id.get(char_tag, 0)
            labels[i, j] = label_id
    return {
        "input_ids": input_ids.to(DEVICE),
        "attention_mask": attention_mask.to(DEVICE),
        "labels": labels.to(DEVICE)
    }

# ========== 计算类别权重（基于训练集） ==========
train_dataset = NERDataset(TRAIN_PATH)
all_labels = []
for item in train_dataset:
    all_labels.extend(item["tags"])
# 统计每个标签的出现次数
label_counts = {label: 0 for label in LABEL_LIST}
for tag in all_labels:
    if tag in label_counts:
        label_counts[tag] += 1

# 计算权重：总数 / (类别数 * 该类频次)，即标准的逆频率权重
total_count = sum(label_counts.values())
class_weights = []
for label in LABEL_LIST:
    count = label_counts[label]
    # 防止除零，最少赋 1
    weight = total_count / (NUM_LABELS * max(count, 1))
    class_weights.append(weight)

class_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
print("类别权重已计算，最小权重类：", LABEL_LIST[np.argmin(class_weights.cpu().numpy())],
      "权重 =", class_weights.min().item())
print("最大权重类：", LABEL_LIST[np.argmax(class_weights.cpu().numpy())],
      "权重 =", class_weights.max().item())

# ========== 模型加载 ==========
model = BertForTokenClassification.from_pretrained(
    MODEL_PATH,
    num_labels=NUM_LABELS,
    id2label=id2label,
    label2id=label2id,
    local_files_only=True
).to(DEVICE)

optimizer = AdamW(model.parameters(), lr=LR)

# 损失函数（带类别权重，忽略 -100）
loss_fct = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)

# ========== DataLoader ==========
dev_dataset = NERDataset(DEV_PATH)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# ========== 训练循环 ==========
best_f1 = 0.0
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        outputs = model(input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"])
        logits = outputs.logits
        # 手动计算加权损失
        loss = loss_fct(logits.view(-1, NUM_LABELS), batch["labels"].view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")

    # 验证
    model.eval()
    all_pred_tags = []
    all_true_tags = []
    with torch.no_grad():
        for batch in dev_loader:
            outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            labels = batch["labels"]
            for i in range(len(predictions)):
                mask = labels[i] != -100
                pred_ids = predictions[i][mask].cpu().numpy()
                true_ids = labels[i][mask].cpu().numpy()
                pred_tags = [id2label[p] for p in pred_ids]
                true_tags = [id2label[t] for t in true_ids]
                all_pred_tags.append(pred_tags)
                all_true_tags.append(true_tags)

    f1 = f1_score(all_true_tags, all_pred_tags)
    print(f"Validation F1: {f1:.4f}")

    if f1 > best_f1:
        best_f1 = f1
        torch.save(model.state_dict(), "best_ner_weighted.pt")
        print("Best model saved.")

    # 每 5 轮输出详细报告
    if (epoch + 1) % 5 == 0:
        print(classification_report(all_true_tags, all_pred_tags))

print(f"训练完成，最佳 F1: {best_f1:.4f}")