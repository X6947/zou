import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast
from torch.optim import AdamW
from tqdm import tqdm
import numpy as np

# ========== 配置 ==========
BERT_PATH = "bert-base-chinese"
TRAIN_PATH = "casrel_data/train.json"
DEV_PATH = "casrel_data/dev.json"
TEST_PATH = "casrel_data/test.json"
REL_PATH = "rel.json"
MAX_LEN = 256
BATCH_SIZE = 4
EPOCHS = 20
LR = 1e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载关系
with open(REL_PATH, "r", encoding="utf-8") as f:
    REL_LIST = json.load(f)
NUM_RELS = len(REL_LIST)
rel2id = {r: i for i, r in enumerate(REL_LIST)}

tokenizer = BertTokenizerFast.from_pretrained(BERT_PATH)

# ========== 你的 CasRel 模型（原样复制，只改预训练路径） ==========
import torch.nn as nn
from transformers import BertModel, BertConfig

class CasRel(nn.Module):
    def __init__(self, config):
        super(CasRel, self).__init__()
        self.config = config
        self.bert_dim = 768
        # 直接使用 BERT_PATH，不再依赖外部 macbert 路径
        self.bert_path = BERT_PATH
        self.bert_config = BertConfig.from_pretrained(self.bert_path)
        self.bert_encoder = BertModel.from_pretrained(self.bert_path, config=self.bert_config)
        self.sub_start_tagger = nn.Linear(self.bert_dim, 1)
        self.sub_end_tagger = nn.Linear(self.bert_dim, 1)
        self.obj_start_tagger = nn.Linear(self.bert_dim, NUM_RELS)
        self.obj_end_tagger = nn.Linear(self.bert_dim, NUM_RELS)

    def get_encoded_text(self, data):
        encoded_text = self.bert_encoder(data['token_ids'],
                                         attention_mask=data['mask'])[0]
        return encoded_text

    def get_sub(self, encoded_text):
        pred_sub_start = self.sub_start_tagger(encoded_text)
        pred_sub_start = torch.sigmoid(pred_sub_start)
        pred_sub_end = self.sub_end_tagger(encoded_text)
        pred_sub_end = torch.sigmoid(pred_sub_end)
        return pred_sub_start, pred_sub_end

    def get_obj(self, sub_start_mapping, sub_end_mapping, encoded_text):
        sub_start = torch.matmul(sub_start_mapping.float(), encoded_text)
        sub_end = torch.matmul(sub_end_mapping.float(), encoded_text)
        sub = (sub_start + sub_end) / 2
        encoded_text = encoded_text + sub
        pred_obj_start = self.obj_start_tagger(encoded_text)
        pred_obj_end = self.obj_end_tagger(encoded_text)
        pred_obj_start = torch.sigmoid(pred_obj_start)
        pred_obj_end = torch.sigmoid(pred_obj_end)
        return pred_obj_start, pred_obj_end

    def get_list(self, start, end, text, h_bar=0.5, t_bar=0.5):
        res = []
        start, end = start[:512], end[:512]
        start_idxs, end_idxs = [], []
        for idx in range(len(start)):
            if start[idx] > h_bar:
                start_idxs.append(idx)
            if end[idx] > t_bar:
                end_idxs.append(idx)
        for start_idx in start_idxs:
            for end_idx in end_idxs:
                if end_idx >= start_idx:
                    entry = {}
                    entry['text'] = text[start_idx:end_idx+1]
                    entry['start'] = start_idx
                    entry['end'] = end_idx
                    res.append(entry)
                    break
        return res

    def forward(self, data):
        encoded_text = self.get_encoded_text(data)
        pred_sub_start, pred_sub_end = self.get_sub(encoded_text)
        sub_start_mapping = data['sub_start'].unsqueeze(1)
        sub_end_mapping = data['sub_end'].unsqueeze(1)
        pred_obj_start, pred_obj_end = self.get_obj(sub_start_mapping,
                                                    sub_end_mapping, encoded_text)
        return pred_sub_start, pred_sub_end, pred_obj_start, pred_obj_end

    def test(self, data):
        encoded_text = self.get_encoded_text(data)
        pred_sub_start, pred_sub_end = self.get_sub(encoded_text)
        sub_list = self.get_list(pred_sub_start.squeeze(0).squeeze(-1),
                                 pred_sub_end.squeeze(0).squeeze(-1),
                                 data['text'])
        if sub_list:
            repeated_encoded_text = encoded_text.repeat(len(sub_list), 1, 1)
            sub_start_mapping = torch.zeros(len(sub_list), 1, encoded_text.shape[1]).to(DEVICE)
            sub_end_mapping = torch.zeros(len(sub_list), 1, encoded_text.shape[1]).to(DEVICE)
            for idx, sub in enumerate(sub_list):
                sub_start_mapping[idx][0][sub['start']] = 1
                sub_end_mapping[idx][0][sub['end']] = 1
            pred_obj_start, pred_obj_end = self.get_obj(sub_start_mapping,
                                                        sub_end_mapping,
                                                        repeated_encoded_text)
            return sub_list, pred_obj_start, pred_obj_end
        else:
            return None

# ========== 数据集与 collate_fn ==========
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

    batch_size, seq_len = input_ids.shape
    sub_start = torch.zeros(batch_size, seq_len, dtype=torch.float)
    sub_end = torch.zeros(batch_size, seq_len, dtype=torch.float)
    obj_start = torch.zeros(batch_size, NUM_RELS, seq_len, dtype=torch.float)
    obj_end = torch.zeros(batch_size, NUM_RELS, seq_len, dtype=torch.float)

    for i, (text, spo_list) in enumerate(zip(texts, spo_lists)):
        for h, r, t in spo_list:
            if r not in rel2id:
                continue
            rid = rel2id[r]
            # 找到头实体的字符偏移
            h_start_char = text.find(h)
            if h_start_char == -1:
                continue
            h_end_char = h_start_char + len(h) - 1
            t_start_char = text.find(t)
            if t_start_char == -1:
                continue
            t_end_char = t_start_char + len(t) - 1

            # 字符偏移 -> token 索引
            h_token_start = h_token_end = None
            t_token_start = t_token_end = None
            for j, (c_start, c_end) in enumerate(offset_mapping[i]):
                if c_start == c_end:  # 特殊 token
                    continue
                if c_start <= h_start_char < c_end and h_token_start is None:
                    h_token_start = j
                if c_start <= h_end_char < c_end:
                    h_token_end = j
                if c_start <= t_start_char < c_end and t_token_start is None:
                    t_token_start = j
                if c_start <= t_end_char < c_end:
                    t_token_end = j

            if h_token_start is not None and h_token_end is not None:
                sub_start[i, h_token_start] = 1.0
                sub_end[i, h_token_end] = 1.0
            if t_token_start is not None and t_token_end is not None:
                obj_start[i, rid, t_token_start] = 1.0
                obj_end[i, rid, t_token_end] = 1.0

    return {
        "token_ids": input_ids.to(DEVICE),
        "mask": attention_mask.to(DEVICE),
        "sub_start": sub_start.to(DEVICE),
        "sub_end": sub_end.to(DEVICE),
        "obj_start": obj_start.to(DEVICE),
        "obj_end": obj_end.to(DEVICE),
        "texts": texts,
        "spo_lists": spo_lists,
        "offset_mapping": offset_mapping
    }

# ========== 损失函数 ==========
def loss_fn(model, batch):
    # 前向传播
    pred_sub_start, pred_sub_end, pred_obj_start, pred_obj_end = model({
        "token_ids": batch["token_ids"],
        "mask": batch["mask"],
        "sub_start": batch["sub_start"],
        "sub_end": batch["sub_end"]
    })

    # 头实体损失
    sub_start_loss = nn.functional.binary_cross_entropy(
        pred_sub_start.squeeze(-1), batch["sub_start"], reduction="none")
    sub_end_loss = nn.functional.binary_cross_entropy(
        pred_sub_end.squeeze(-1), batch["sub_end"], reduction="none")
    head_loss = ((sub_start_loss + sub_end_loss) * batch["mask"]).sum() / batch["mask"].sum()

    # 尾实体损失 (pred_obj_start: [batch, seq_len, rels] -> [batch, rels, seq_len])
    obj_start_loss = nn.functional.binary_cross_entropy(
        pred_obj_start.permute(0, 2, 1), batch["obj_start"], reduction="none")
    obj_end_loss = nn.functional.binary_cross_entropy(
        pred_obj_end.permute(0, 2, 1), batch["obj_end"], reduction="none")
    mask = batch["mask"].unsqueeze(1)  # [batch, 1, seq_len]
    tail_loss = ((obj_start_loss + obj_end_loss) * mask).sum() / mask.sum()

    return head_loss + tail_loss

# ========== 训练循环 ==========
def train():
    train_dataset = DiaKGDataset(TRAIN_PATH)
    dev_dataset = DiaKGDataset(DEV_PATH)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model = CasRel({"relation_types": NUM_RELS}).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LR)

    best_val_loss = float("inf")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            loss = loss_fn(model, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")

        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in dev_loader:
                loss = loss_fn(model, batch)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(dev_loader)
        print(f"Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_model_standard.pt")
            print("Best model saved.")

    print("Training finished.")

if __name__ == "__main__":
    train()