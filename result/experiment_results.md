 实验结果汇总

## 一、NER 实验

### 1.1 18类原始标注（标签数 37）
- 训练脚本: `train_ner.py`
- 模型: `bert-base-chinese`
- 参数: MAX_LEN=256, BATCH_SIZE=8, EPOCHS=10, LR=3e-5
- 数据: `ner_data`（原始18类，2343条训练句）

**训练日志节选**:
Epoch 1 Loss: 1.0680 Val F1: 0.2538
Epoch 2 Loss: 0.7421 Val F1: 0.2832
Epoch 3 Loss: 0.6037 Val F1: 0.3101
Epoch 4 Loss: 0.5172 Val F1: 0.3561
Epoch 5 Loss: 0.4442 Val F1: 0.3634
Epoch 6 Loss: 0.4000 Val F1: 0.3575
Epoch 7 Loss: 0.3612 Val F1: 0.3727
Epoch 8 Loss: 0.3434 Val F1: 0.3727
Epoch 9 Loss: 0.3112 Val F1: 0.3741
Epoch 10 Loss: 0.2872 Val F1: 0.3622
Training finished. Best F1: 0.3741

text

**分类报告（Epoch 9）**:
precision recall f1-score support
ADE 0.48 0.23 0.31 132
Amount 0.00 0.00 0.00 21
Anatomy 0.00 0.00 0.00 42
Class 0.20 0.07 0.11 14
Disease 0.47 0.35 0.40 868
Drug 0.52 0.40 0.45 637
Duration 0.00 0.00 0.00 18
Frequency 0.20 0.18 0.19 11
Level 0.42 0.40 0.41 35
Method 0.74 0.41 0.53 63
Operation 0.25 0.12 0.16 26
Pathogenesis 0.42 0.32 0.37 34
Reason 0.06 0.05 0.05 21
Symptom 0.14 0.18 0.16 22
Test 0.11 0.03 0.05 60
Test_Value 0.20 0.15 0.17 232
Test_items 0.38 0.32 0.35 405
Treatment 0.23 0.24 0.23 76
micro avg 0.42 0.31 0.35 2717

text

**问题**: 类别极度不平衡，低频类无法学习。

---

### 1.2 语义合并为8类（标签数 17）
- 合并映射: ADE, Anatomy, Disease, Drug, Measurement (含Amount/Duration/Frequency/Level), Reason, Test (含Test_items/Test_Value/Test), Treatment (含Method/Operation)
- 训练脚本: `train_ner.py` (DATA_DIR=ner_data_merged)
- 参数: 同上

**结果**:
Best F1: 0.3784

text
与18类相比无明显提升，验证集中Reason、Measurement等类样本仍过少。

---

### 1.3 语义合并为5类（标签数 11）
- 合并映射: Disease (含Symptom/Class), Drug, Test (含Test_items/Test_Value/Test), Treatment (含Method/Operation), Other (ADE/Anatomy/Measurement/Reason/Pathogenesis等)
- 训练脚本: `train_ner.py` (DATA_DIR=ner_data_v2)
- 参数: MAX_LEN=256, BATCH_SIZE=8, EPOCHS=10, LR=3e-5

**训练日志**:
Epoch 1 Loss: 0.8459 Val F1: 0.2603
Epoch 2 Loss: 0.6047 Val F1: 0.3336
Epoch 3 Loss: 0.5029 Val F1: 0.3640
Epoch 4 Loss: 0.4331 Val F1: 0.3678
Epoch 5 Loss: 0.3738 Val F1: 0.3578
Epoch 6 Loss: 0.3526 Val F1: 0.3722
Epoch 7 Loss: 0.3057 Val F1: 0.3739
Epoch 8 Loss: 0.2928 Val F1: 0.3733
Epoch 9 Loss: 0.2751 Val F1: 0.3679
Epoch 10 Loss: 0.2512 Val F1: 0.3762
Best F1: 0.3762

text

**分析**: 合并后并未带来预期的提升，F1依然停留在0.37附近。说明 `bert-base-chinese` 在 DiaKG 上的基础能力有限，可能需要更强的预训练模型或更复杂的网络结构。

---

### 1.4 类别加权尝试（8类）
- 脚本: `train_ner_weighted.py`
- 方法: 逆频率权重（最大值被限制在 ~119）
- 结果: F1 跌至 0.0466，模型崩溃。

---

## 二、CasRel 联合抽取实验

### 2.1 训练收敛情况
- 脚本: `casrel_train.py`
- 模型: `bert-base-chinese`
- 数据: `casrel_data` (训练集1036句，验证集263句，16种关系)

**训练 Loss 下降曲线**:
Epoch 1 Loss: 5.9279 Val Loss: 2.7957
Epoch 2 Loss: 2.4536 Val Loss: 1.9728
Epoch 3 Loss: 1.7717 Val Loss: 1.5242
Epoch 4 Loss: 1.3899 Val Loss: 1.2598
Epoch 5 Loss: 1.1501 Val Loss: 1.0943
Epoch 6 Loss: 0.9992 Val Loss: 0.9572
Epoch 7 Loss: 0.8591 Val Loss: 0.8154
Epoch 8 Loss: 0.7439 Val Loss: 0.6995
Epoch 9 Loss: 0.6421 Val Loss: 0.6197
Epoch 10 Loss: 0.5608 Val Loss: 0.5417
Epoch 11 Loss: 0.4968 Val Loss: 0.5037
Epoch 12 Loss: 0.4472 Val Loss: 0.4602
Epoch 13 Loss: 0.4043 Val Loss: 0.4416
Epoch 14 Loss: 0.3696 Val Loss: 0.4139
Epoch 15 Loss: 0.3385 Val Loss: 0.3937
Epoch 16 Loss: 0.3102 Val Loss: 0.3782
Epoch 17 Loss: 0.2852 Val Loss: 0.3681
Epoch 18 Loss: 0.2637 Val Loss: 0.3683
Epoch 19 Loss: 0.2440 Val Loss: 0.3469
Epoch 20 Loss: 0.2272 Val Loss: 0.3425

text
模型收敛健康，验证损失持续下降，无过拟合。

### 2.2 解码评估
- 脚本: `evaluate.py`
- 测试集: 262句

**结果**:
Precision: 0.0185
Recall: 0.1755
F1 Score: 0.0334

text
预测样例中出现大量超长实体或 UNK 标签。

**修复解码后**:
- 增加有效token掩码、长度限制（≤30字）、最近优先匹配
- 实体不再出现整句覆盖，但整体 F1 仍极低，说明模型在未注入头实体信息的条件下难以准确预测尾实体。

---

## 三、总结

- **NER**: `bert-base-chinese` 在 DiaKG 上无论多少类合并，F1 难以突破 0.38，官方 0.833 的结果依赖 RoBERTa-large 和 MRC 框架，普通序列标注方法存在明显天花板。
- **CasRel**: 训练损失正常收敛，但解码三元组效果极差，需改进模型结构（如注入头实体条件信息）或采用更强大的预训练模型。

下一步计划：更换预训练模型为 `hfl/chinese-roberta-wwm-ext` 或直接复现官方 NER 代码，并优化 CasRel 的条件编码器。