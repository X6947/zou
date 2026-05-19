import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertModel, AdamW
from tqdm import tqdm
import os

# ========== 配置 ==========
BERT_PATH = "bert-base-chinese"          # 或改成你的本地路径
TRAIN_PATH = "casrel_data/train.json"
DEV_PATH = "casrel_data/dev.json"
TEST_PATH = "casrel_data/test.json"
REL_PATH = "rel.json"
MAX_LEN = 256
BATCH_SIZE = 8
EPOCHS = 20
LR = 1e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载关系列表
with open(REL_PATH, "r", encoding="utf-8") as f:
    REL_LIST = json.load(f)
NUM_RELS = len(REL_LIST)
rel2id = {r: i for i, r in enumerate(REL_LIST)}

tokenizer = BertTokenizerFast.from_pretrained(BERT_PATH)

# ========== 数据集 ==========
class DiaKGDataset(Dataset):
    def __init__(self, data_path):
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    texts = [item["text"] for item in batch]
    spo_lists = [item["spo_list"] for item in batch]

    # tokenize
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
    offset_mapping = encoded["offset_mapping"]

    # 生成标签矩阵
    batch_size, seq_len = input_ids.shape
    # 头实体标签: [batch, seq_len, 2] (start, end)
    head_labels = torch.zeros(batch_size, seq_len, 2, dtype=torch.float)
    # 尾实体标签: [batch, NUM_RELS, seq_len, 2]
    tail_labels = torch.zeros(batch_size, NUM_RELS, seq_len, 2, dtype=torch.float)

    for i, (text, spo_list) in enumerate(zip(texts, spo_lists)):
        # 对每个spo，找到对应token位置
        for h, r, t in spo_list:
            if r not in rel2id:
                continue
            rid = rel2id[r]
            # 头实体位置
            h_start = text.find(h)
            if h_start == -1:
                continue
            h_end = h_start + len(h) - 1
            # 尾实体位置
            t_start = text.find(t)
            if t_start == -1:
                continue
            t_end = t_start + len(t) - 1

            # 找到token级别的start/end索引
            token_start_h = token_end_h = None
            token_start_t = token_end_t = None
            for j, (start, end) in enumerate(offset_mapping[i]):
                if start == end:  # 特殊token，跳过
                    continue
                if start <= h_start < end and token_start_h is None:
                    token_start_h = j
                if start <= h_end < end:
                    token_end_h = j
                if start <= t_start < end and token_start_t is None:
                    token_start_t = j
                if start <= t_end < end:
                    token_end_t = j

            if token_start_h is not None and token_end_h is not None:
                head_labels[i, token_start_h, 0] = 1  # start
                head_labels[i, token_end_h, 1] = 1    # end

            if token_start_t is not None and token_end_t is not None:
                tail_labels[i, rid, token_start_t, 0] = 1
                tail_labels[i, rid, token_end_t, 1] = 1

    return {
        "input_ids": input_ids.to(DEVICE),
        "attention_mask": attention_mask.to(DEVICE),
        "head_labels": head_labels.to(DEVICE),
        "tail_labels": tail_labels.to(DEVICE),
        "texts": texts,
        "spo_lists": spo_lists,
        "offset_mapping": offset_mapping
    }

# ========== 模型 ==========
class CasRelModel(torch.nn.Module):
    def __init__(self, bert_path, num_rels):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        hidden_size = self.bert.config.hidden_size

        # 头实体识别器: 2分类(start/end)
        self.head_start = torch.nn.Linear(hidden_size, 1)
        self.head_end = torch.nn.Linear(hidden_size, 1)

        # 尾实体识别器：每种关系一个start/end二分类器
        self.tail_start = torch.nn.Linear(hidden_size, num_rels)
        self.tail_end = torch.nn.Linear(hidden_size, num_rels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        seq_output = outputs.last_hidden_state  # [batch, seq_len, hidden]

        # 头实体预测
        head_start_logits = self.head_start(seq_output).squeeze(-1)  # [batch, seq_len]
        head_end_logits = self.head_end(seq_output).squeeze(-1)

        # 尾实体预测
        tail_start_logits = self.tail_start(seq_output)  # [batch, seq_len, num_rels]
        tail_start_logits = tail_start_logits.permute(0, 2, 1)  # [batch, num_rels, seq_len]
        tail_end_logits = self.tail_end(seq_output).permute(0, 2, 1)

        return head_start_logits, head_end_logits, tail_start_logits, tail_end_logits

# ========== 损失函数 ==========
def loss_fn(head_logits, tail_logits, head_labels, tail_labels, attention_mask):
    # head loss
    head_start_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        head_logits[0], head_labels[:, :, 0], reduction="none"
    )
    head_end_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        head_logits[1], head_labels[:, :, 1], reduction="none"
    )
    head_loss = (head_start_loss + head_end_loss) * attention_mask
    head_loss = head_loss.sum() / attention_mask.sum()

    # tail loss
    tail_start_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        tail_logits[0], tail_labels[:, :, :, 0], reduction="none"
    )  # [batch, num_rels, seq_len]
    tail_end_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        tail_logits[1], tail_labels[:, :, :, 1], reduction="none"
    )
    # 掩码只在seq_len维度，需要广播到num_rels
    mask = attention_mask.unsqueeze(1)  # [batch, 1, seq_len]
    tail_start_loss = tail_start_loss * mask
    tail_end_loss = tail_end_loss * mask
    tail_loss = (tail_start_loss + tail_end_loss).sum() / mask.sum()

    return head_loss + tail_loss

# ========== 训练 ==========
def train():
    train_dataset = DiaKGDataset(TRAIN_PATH)
    dev_dataset = DiaKGDataset(DEV_PATH)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model = CasRelModel(BERT_PATH, NUM_RELS).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LR)

    best_f1 = 0.0
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            head_start_logits, head_end_logits, tail_start_logits, tail_end_logits = model(
                batch["input_ids"], batch["attention_mask"]
            )
            loss = loss_fn(
                (head_start_logits, head_end_logits),
                (tail_start_logits, tail_end_logits),
                batch["head_labels"],
                batch["tail_labels"],
                batch["attention_mask"]
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")

        # 验证（简化版，只计算损失，如需F1可后续添加）
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in dev_loader:
                head_start_logits, head_end_logits, tail_start_logits, tail_end_logits = model(
                    batch["input_ids"], batch["attention_mask"]
                )
                loss = loss_fn(
                    (head_start_logits, head_end_logits),
                    (tail_start_logits, tail_end_logits),
                    batch["head_labels"],
                    batch["tail_labels"],
                    batch["attention_mask"]
                )
                val_loss += loss.item()
        avg_val_loss = val_loss / len(dev_loader)
        print(f"Val Loss: {avg_val_loss:.4f}")

        # 保存最佳模型
        if avg_val_loss < best_f1 or best_f1 == 0.0:
            best_f1 = avg_val_loss
            torch.save(model.state_dict(), "best_model.pt")
            print("Best model saved.")

    print("Training finished.")

if __name__ == "__main__":
    train()