# 基于DiaKG数据集的糖尿病医学知识图谱构建（中期报告）

## 📌 项目简介
本项目以高质量标注数据集 **DiaKG** 为核心，研究从中文糖尿病临床文本中自动抽取医学实体与关系，并构建可查询、可扩展的知识图谱。核心任务包括：
- **命名实体识别（NER）**：识别 18 类糖尿病相关实体，并探索类别不平衡的解决方案。
- **关系三元组联合抽取**：采用 CasRel 框架，端到端提取“头实体-关系-尾实体”。
- **知识图谱存储与查询**：将抽取的结构化知识导入 Neo4j 图数据库，实现可视化与语义查询。

本仓库为课题 **中期成果**，包含数据预处理、NER 对比实验、CasRel 模型训练与评估、错误分析等工作。

---

## 📖 研究背景与动机
- 全球糖尿病患者超 5 亿，我国患病人数超 1.4 亿，临床知识散落在指南、电子病历等非结构化文本中。
- DiaKG 作为中文糖尿病领域首个细粒度标注数据集（18类实体、15+种关系），为结构化知识抽取提供了理想基础。
- 传统流水线方法存在错误传播问题，且现有研究多聚焦单一任务，缺乏从抽取到图谱构建的系统化验证。
- 本研究旨在填补这一空白，通过对比不同预训练模型与解码策略，建立稳健基线，并形成完整的知识图谱构建流水线。

---

## 🧩 已完成工作（中期）

### 1. 数据准备与预处理
- 整理 DiaKG 官方数据集（41篇指南/共识），严格按文档级随机划分（70/15/15）。
- 编写转换脚本，生成两种任务所需的数据格式：
  - **CasRel 格式**（`{text, spo_list}`）：含 16 种关系类型。
  - **NER BIO 格式**（字符级标签）：原始 18 类实体，37 个 BIO 标签。
- 生成关系列表 `rel.json` 与实体标签列表 `label_list.json`。

### 2. NER 实验与分析（研究问题 Q1）
- 使用 `BertForTokenClassification` + `bert-base-chinese` 构建基线。
- **初步结果**：18 类原始标签下验证集 F1 仅 0.374，发现核心障碍是**实体类别极度不平衡**（如 Anatomy、Duration 等占比 < 2%）。
- **类别加权实验**：尝试逆频率权重，因权重值过大导致模型崩溃（F1 降至 0.05）。
- **语义合并探索**：
  - 首次合并为 8 类 → F1 提升微弱（~0.378）。
  - 二次合并为 5 类（Disease, Drug, Test, Treatment, Other）→ F1 仍停滞在 0.376，表明基础模型能力存在天花板，需更换更强预训练模型。
- 诊断工具脚本已编写，可随时检查数据分布与标签对齐。

### 3. CasRel 联合抽取模型（研究问题 Q2）
- 实现标准 CasRel 框架：头实体识别器 + 关系特定的尾实体标注器。
- 解决中文子词对齐问题（使用 `BertTokenizerFast` + `offset_mapping`）。
- 训练过程 Loss 从 5.93 稳定收敛至 0.22，模型已充分学习。
- 修复解码逻辑（有效 token 掩码、实体长度限制、最近优先策略），消除“整句实体”误判。
- 当前解码 F1 仍偏低，已定位原因（标签稀疏 + 头实体信息未充分利用），后续将优化模型结构。

### 4. 思路调研
- 研究官方 diaKG-code 仓库的 MRC 框架 + RoBERTa-large 配置。
- 分析其 18 类结果：总体 F1 0.833，但 Reason 等低频类 F1 仅 0.316，印证长尾挑战。
- 尝试复现 MRC 数据格式，为后续对比实验提供参考。

### 5. 工具链与实验管理
- 所有脚本支持离线加载模型（`local_files_only=True`），适应国内网络环境。
- 数据划分固定随机种子（42），保证可复现。
- 结果日志已整理至 `results/experiment_results.md`。

---

## ⚠️ 主要挑战与应对

| 挑战 | 表现 | 解决方案 |
|------|------|----------|
| 实体类别极度不平衡 | 低频类 Recall=0 | 语义合并为 5 大类 |
| 简单加权导致模型崩溃 | F1 不升反降 | 改用 Focal Loss + 温和权重（进行中） |
| CasRel 解码产生超长实体 | 尾实体覆盖整句 | 长度限制 + 最近优先匹配 |
| 中文子词对齐错误 | 出现 [UNK] 标签 | 使用 `word_ids` + `offset_mapping` 精确对齐 |
| 网络限制，模型下载超时 | 训练无法启动 | 强制离线加载 + 手动缓存模型文件 |

---

## 📊 中期实验结果（详见 `results/experiment_results.md`）

### NER 任务（bert-base-chinese，验证集）
| 实体粒度 | 标签数 | 最佳 F1 | 备注 |
|----------|--------|---------|------|
| 18 类（原始） | 37 | 0.3741 | 低频类无法学习 |
| 8 类（首次合并） | 17 | 0.3784 | 改善甚微 |
| 5 类（二次合并） | 11 | 0.3762 | 基础模型天花板，需换模型 |

### CasRel 联合抽取
- 训练 Loss 曲线健康（5.93→0.22），模型收敛。
- 解码 F1 目前较低，待模型结构优化后重新评估。

---

## 📁 仓库文件结构
├── README.md # 本文件
├── requirements.txt # Python 依赖
├── .gitignore # 忽略大文件与缓存
├── data/ # 数据样例（不含原始 DiaKG）
├── scripts/ # 所有核心代码
│ ├── 1.data.py
│ ├── casrel.py
│ ├── check_data_balance.py
│ ├── prepare_ner_data.py
│ ├── evaluate.py
│ ├── merge_labels.py
│ ├── merge_labels_v2.py
│ ├── train_ner.py
│ ├── train_ner_weighted.py
│ ├── train_ner_improved.py
│ ├── train_ner_weighted.py
│ └── train_standard_casrel.py
└── results/ # 实验结果汇总
└── experiment_results.md
📅 下一步计划
更换预训练模型为 hfl/chinese-roberta-wwm-ext 或更大规模模型，重新评估 NER 性能。

对比不同解码策略（CRF、Focal Loss），完成 Q1 消融实验。

优化 CasRel 模型（注入头实体条件信息），提升三元组抽取 F1。

将高质量三元组导入 Neo4j，设计“药物-疾病-并发症”等典型查询，验证知识图谱有效性。

撰写完整论文，输出实验表格、图谱可视化与案例分析。

📚 参考文献
[1] Chang D, et al. DiaKG: an Annotated Diabetes Dataset for Medical Knowledge Graph Construction. CCKS 2021.
[2] Wei Z, et al. A Novel Cascade Binary Tagging Framework for Relational Triple Extraction. ACL 2020.
[3] Devlin J, et al. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. NAACL 2019.
[4] Lee J, et al. BioBERT: a pre-trained biomedical language representation model for biomedical text mining. Bioinformatics, 2020.
[5] Xu Y, et al. Enhancing Diabetes Management With CRIBC: A Novel NER Model... Engineering Reports, 2025.
[6] Ji X, et al. CPMFA: A Character Pair-Based Method for Chinese Nested NER. ADMA 2023.