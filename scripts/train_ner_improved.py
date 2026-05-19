import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertForTokenClassification, AdamW
from seqeval.metrics import f1_score
from tqdm import tqdm

# ========== 配置 ==========
# 可替换为 "bert-base-chinese" 若无法下载 RoBERTa
MODEL_PATH = "bert-base-chinese"
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

MAX_LEN = 128          # RoBERTa 官方推荐 128，且大多数句子 < 128
BATCH_SIZE = 16
EPOCHS = 15
LR = 2e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== Focal Loss ==========
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean', ignore_index=-100):
        super().__init__()
        self.alpha = alpha          # 类别权重 [num_labels]
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        # inputs: [batch*seq, num_labels], targets: [batch*seq]
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha,
                                  reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)    # 预测正确概率
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            valid = (targets != self.ignore_index).float()
            return focal_loss.sum() / valid.sum()
        return focal_loss.sum()

# 计算类别权重（基于训练集）
train_dataset_raw = []
with open(TRAIN_PATH, "r", encoding="utf-8") as f:
    train_data = json.load(f)
# 统计标签出现次数
label_counts = torch.zeros(NUM_LABELS)
for item in train_data:
    for tag in item["tags"]:
        label_counts[label2id[tag]] += 1
# 逆频率，平滑
alpha = 1.0 / (label_counts + 1)
alpha = alpha / alpha.sum() * NUM_LABELS  # 归一化
alpha = alpha.to(DEVICE)

loss_fn = FocalLoss(alpha=alpha, gamma=2.0, ignore_index=-100)

# ========== FGM 对抗训练 ==========
class FGM:
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=0.5):
        for name, param in self.model.named_parameters():
            if param.requires_grad and 'embeddings' in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and 'embeddings' in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

fgm = None  # 初始化，等模型创建后绑定

# ========== 数据集 ==========
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
        for j, word_idx in enumerate(word_ids):
            if word_idx is None:
                continue
            labels[i, j] = label2id.get(tags_list[i][word_idx], 0)
    return {
        "input_ids": input_ids.to(DEVICE),
        "attention_mask": attention_mask.to(DEVICE),
        "labels": labels.to(DEVICE)
    }

# ========== 模型加载 ==========
model = BertForTokenClassification.from_pretrained(
    MODEL_PATH,
    num_labels=NUM_LABELS,
    id2label=id2label,
    label2id=label2id,
    local_files_only=True
).to(DEVICE)

optimizer = AdamW(model.parameters(), lr=LR)
fgm = FGM(model)

# ========== DataLoader ==========
train_dataset = NERDataset(TRAIN_PATH)
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
        loss = loss_fn(logits.view(-1, NUM_LABELS), batch["labels"].view(-1))

        optimizer.zero_grad()
        loss.backward()

        # FGM 对抗训练
        fgm.attack(epsilon=0.5)
        outputs_adv = model(input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"])
        loss_adv = loss_fn(outputs_adv.logits.view(-1, NUM_LABELS),
                           batch["labels"].view(-1))
        loss_adv.backward()
        fgm.restore()

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
        torch.save(model.state_dict(), "best_ner_improved.pt")
        print("Best model saved.")

print(f"Training finished. Best F1: {best_f1:.4f}")