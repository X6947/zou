import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertForTokenClassification, AdamW
from seqeval.metrics import f1_score
from tqdm import tqdm

# 配置
MODEL_PATH = "bert-base-chinese"  # 如果缓存没了，改成 r"D:\models\bert-base-chinese"
DATA_DIR = "ner_data_v2"
TRAIN_PATH = os.path.join(DATA_DIR, "train.json")
DEV_PATH = os.path.join(DATA_DIR, "dev.json")
TEST_PATH = os.path.join(DATA_DIR, "test.json")
LABEL_PATH = os.path.join(DATA_DIR, "label_list.json")

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
    tags_list = [item["tags"] for item in batch]

    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt",
        return_offsets_mapping=True
    )

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    batch_size, seq_len = input_ids.shape
    labels = torch.full((batch_size, seq_len), -100, dtype=torch.long)
    for i in range(batch_size):
        word_ids = encoded.word_ids(batch_index=i)
        previous_word_idx = None
        for j, word_idx in enumerate(word_ids):
            if word_idx is None:
                continue
            if word_idx != previous_word_idx:
                labels[i, j] = label2id.get(tags_list[i][word_idx], 0)
            else:
                labels[i, j] = label2id.get(tags_list[i][word_idx], 0)
            previous_word_idx = word_idx

    return {
        "input_ids": input_ids.to(DEVICE),
        "attention_mask": attention_mask.to(DEVICE),
        "labels": labels.to(DEVICE)
    }

# 模型加载
model = BertForTokenClassification.from_pretrained(
    MODEL_PATH,
    num_labels=NUM_LABELS,
    id2label=id2label,
    label2id=label2id,
    local_files_only=True
).to(DEVICE)
optimizer = AdamW(model.parameters(), lr=LR)

# 数据加载器
train_dataset = NERDataset(TRAIN_PATH)
dev_dataset = NERDataset(DEV_PATH)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# 训练
best_f1 = 0.0
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        outputs = model(**batch)
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")

    # 验证
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in dev_loader:
            outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            labels = batch["labels"]
            for i in range(len(predictions)):
                mask = labels[i] != -100
                pred_tags = [id2label[p] for p in predictions[i][mask].cpu().numpy()]
                true_tags = [id2label[l.item()] for l in labels[i][mask]]
                all_preds.append(pred_tags)
                all_labels.append(true_tags)

    f1 = f1_score(all_labels, all_preds)
    print(f"Validation F1: {f1:.4f}")
    if f1 > best_f1:
        best_f1 = f1
        torch.save(model.state_dict(), "best_ner_model.pt")
        print("Best model saved.")

print(f"Training finished. Best F1: {best_f1:.4f}")