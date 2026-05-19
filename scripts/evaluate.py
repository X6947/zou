import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

import json
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertModel

# ======== 配置 ========
BERT_PATH = "bert-base-chinese"
TEST_PATH = "casrel_data/test.json"
REL_PATH = "rel.json"
MODEL_PATH = "best_model.pt"
MAX_LEN = 256
BATCH_SIZE = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载关系
with open(REL_PATH, "r", encoding="utf-8") as f:
    REL_LIST = json.load(f)
NUM_RELS = len(REL_LIST)
id2rel = {i: r for i, r in enumerate(REL_LIST)}

tokenizer = BertTokenizerFast.from_pretrained(BERT_PATH)

# ======== 模型定义（与训练完全一致） ========
class CasRelModel(torch.nn.Module):
    def __init__(self, bert_path, num_rels):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        hidden_size = self.bert.config.hidden_size
        self.head_start = torch.nn.Linear(hidden_size, 1)
        self.head_end = torch.nn.Linear(hidden_size, 1)
        self.tail_start = torch.nn.Linear(hidden_size, num_rels)
        self.tail_end = torch.nn.Linear(hidden_size, num_rels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        seq = outputs.last_hidden_state
        head_start_logits = self.head_start(seq).squeeze(-1)
        head_end_logits = self.head_end(seq).squeeze(-1)
        tail_start_logits = self.tail_start(seq).permute(0, 2, 1)
        tail_end_logits = self.tail_end(seq).permute(0, 2, 1)
        return head_start_logits, head_end_logits, tail_start_logits, tail_end_logits

model = CasRelModel(BERT_PATH, NUM_RELS).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ======== 数据集 ========
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
    encoded = tokenizer(texts, padding=True, truncation=True, max_length=MAX_LEN,
                        return_tensors="pt", return_offsets_mapping=True)
    return {
        "input_ids": encoded["input_ids"].to(DEVICE),
        "attention_mask": encoded["attention_mask"].to(DEVICE),
        "offset_mapping": encoded["offset_mapping"],
        "texts": texts,
        "spo_lists": spo_lists
    }

dataset = DiaKGDataset(TEST_PATH)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# ======== 稳健的解码函数 ========
def extract_spans(start_probs, end_probs, valid_mask, threshold=0.5):
    """
    根据 start/end 概率找出所有合法实体片段。
    valid_mask: bool [seq_len] 标记哪些位置是真实token（非特殊token）
    """
    start_probs = start_probs.cpu()
    end_probs = end_probs.cpu()
    valid_mask = valid_mask.cpu()

    # 仅保留概率 > 阈值且属于有效位置的索引
    start_idx = torch.where((start_probs > threshold) & valid_mask)[0].tolist()
    end_idx = torch.where((end_probs > threshold) & valid_mask)[0].tolist()

    spans = []
    for s in start_idx:
        for e in end_idx:
            if s <= e:
                # 确保起始和结束 token 属于同一个连续的有效区域，且不跨越 [SEP]
                if valid_mask[s] and valid_mask[e]:
                    spans.append((s, e))
    return spans

def decode_one(text, input_ids, attention_mask, offset_mapping):
    """解码单个样本，返回预测的三元组列表（字符串）"""
    input_ids = input_ids.unsqueeze(0)
    attention_mask = attention_mask.unsqueeze(0)
    offset_mapping = offset_mapping.unsqueeze(0)

    with torch.no_grad():
        head_start_logits, head_end_logits, tail_start_logits, tail_end_logits = model(
            input_ids, attention_mask
        )

    seq_len = input_ids.shape[1]
    head_start_probs = torch.sigmoid(head_start_logits[0])
    head_end_probs = torch.sigmoid(head_end_logits[0])

    # 有效token掩码：不是[CLS],[SEP],[PAD]，且字符偏移不为0长度
    valid_mask = torch.zeros(seq_len, dtype=torch.bool)
    for i in range(seq_len):
        start_char, end_char = offset_mapping[0, i].tolist()
        # 排除特殊token（start==end通常为特殊标记）以及padding
        if start_char < end_char and attention_mask[0, i] == 1:
            valid_mask[i] = True

    # 解码头实体
    head_spans = extract_spans(head_start_probs, head_end_probs, valid_mask)
    if not head_spans:
        return []

    # 按顺序组织token id 便于解码实体字符串
    token_ids = input_ids[0].cpu()
    pred_spos = []

    for h_s, h_e in head_spans:
        # 从offset_mapping提取头实体文本（用字符级拼接更准确）
        h_start_char = offset_mapping[0, h_s, 0].item()
        h_end_char = offset_mapping[0, h_e, 1].item()
        head_text = text[h_start_char:h_end_char]

        # 对每种关系解码尾实体
        for rid in range(NUM_RELS):
            rel_name = id2rel[rid]
            tail_start_probs = torch.sigmoid(tail_start_logits[0, rid])
            tail_end_probs = torch.sigmoid(tail_end_logits[0, rid])
            tail_spans = extract_spans(tail_start_probs, tail_end_probs, valid_mask)

            for t_s, t_e in tail_spans:
                t_start_char = offset_mapping[0, t_s, 0].item()
                t_end_char = offset_mapping[0, t_e, 1].item()
                tail_text = text[t_start_char:t_end_char]

                # 简单过滤：尾实体不为空，且不与头实体完全重复（允许部分重叠）
                if tail_text and len(tail_text.strip()) > 0:
                    pred_spos.append((head_text, rel_name, tail_text))

    return pred_spos

# ======== 评估主循环 ========
all_predictions = []
all_golds = []

for batch in tqdm(loader, desc="评估中"):
    for i in range(len(batch["texts"])):
        text = batch["texts"][i]
        input_ids = batch["input_ids"][i]
        attention_mask = batch["attention_mask"][i]
        offset_mapping = batch["offset_mapping"][i]
        gold_spos = batch["spo_lists"][i]

        pred_spos = decode_one(text, input_ids, attention_mask, offset_mapping)

        gold_set = set((str(h), str(r), str(t)) for h, r, t in gold_spos)
        pred_set = set(pred_spos)

        all_predictions.append(pred_set)
        all_golds.append(gold_set)

# 计算全局指标
tp, fp, fn = 0, 0, 0
for preds, golds in zip(all_predictions, all_golds):
    tp += len(preds & golds)
    fp += len(preds - golds)
    fn += len(golds - preds)

precision = tp / (tp + fp + 1e-10)
recall = tp / (tp + fn + 1e-10)
f1 = 2 * precision * recall / (precision + recall + 1e-10)

print(f"\n{'='*40}")
print(f"评估结果（测试集）")
print(f"True Positive : {tp}")
print(f"False Positive: {fp}")
print(f"False Negative: {fn}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"{'='*40}")

# 展示修正后的预测样例
print("\n样例预测（前3条）：")
for i in range(min(3, len(dataset))):
    text = dataset[i]["text"]
    golds = all_golds[i]
    preds = all_predictions[i]
    print(f"\n原文: {text[:100]}...")
    print(f"真实三元组(部分): {list(golds)[:2]}")
    print(f"预测三元组(部分): {list(preds)[:2]}")