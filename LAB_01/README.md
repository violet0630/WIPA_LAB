# Web 信息处理与应用 · 实验一：信息检索

基于清华大学 THUIR 与小红书开源的 **Qilin-IR-50K** 教学子集（50,000 篇中文笔记及其搜索、行为数据），实现从分词规范化、倒排索引构建与压缩，到布尔 / TF-IDF / 预训练语义检索的完整信息检索流水线。本仓库用于复现实验，只包含代码与复现说明；实验输出与报告在本地生成，不入库。

## LAB_01目录结构

```
LAB_01/
├── data/                  Qilin-IR-50K 数据集（需解压缩data.zip获得）
│   └── optional/          选做任务（个性化推荐）的行为数据，本次未使用
├── src/                   全部实验代码（任务↔代码↔输出对照见 src/README.md）
├── output/                实验输出：1 分词 / 2 倒排索引 / 3 位置与压缩 / 4 检索实验（本地生成，不入库）
├── requirements.txt       Python 依赖
├── 实验1.pdf              实验要求原文
├── data.zip               数据集压缩包
└── README.md              本文件
```

## 环境配置

- Python ≥ 3.10，CPU 即可复现全部结果（语义编码阶段较慢，GPU 可加速）
- 数据集从 [Qilin-IR-50K](https://huggingface.co/datasets/THUIR/Qilin) 获取，将 7 个 JSONL 文件放入 `data/`（`documents.jsonl`、`queries.jsonl`、`search_logs.jsonl`、`user_history.jsonl`、`users.jsonl`、`optional/` 两个文件）；`data/stopwords.txt` 已随仓库提供

```bash
cd LAB_01
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt   # Windows；Linux/macOS 用 .venv/bin/pip
```

## 复现命令

所有脚本均在 **`LAB_01/` 目录**执行（即 `src/` 的上一级），按下列顺序运行即为完整实验流程。规模约定：`--size 100 / 1000` 为正确性自测（可用 `--limit 100` 等参数先行验证），`--size all`（50,000 篇）为核心实验数据；`--variant keep / drop` 表示保留 / 去除停用词。

```bash
# 数据体检（可选）：校验 7 个 JSONL 的行数与字段
python src/00_check_data.py

# 任务 1：分词与规范化（FMM + HMM，keep/drop 两套）→ output/1/
python src/01_segment.py --limit 0

# 任务 2：文档级倒排索引 + 单层跳表 → output/2/
python src/02_build_index.py --size all
python src/02_build_index.py --size all --variant drop

# 任务 3：位置索引 + 四组索引压缩（Raw/Front/VByte/Combined）与解压验证 → output/3/
python src/03_build_positions.py --size all
python src/03_compress_index.py --size all
python src/03_verify_compression.py --size all --full --mode both

# 任务 4B：TF-IDF 矩阵与批量 keep/drop 检索对照（12 条查询）→ output/4/
python src/04B_build_tfidf.py --size all --variant keep
python src/04B_build_tfidf.py --size all --variant drop
python src/04B_search_tfidf_implement.py --size all --top 10

# 任务 4C：BGE 语义向量（三种输入方式）与检索对照 → output/4/semantic/
#   首次运行需联网下载模型权重；全量逐词编码耗时较长
python src/04C_build_semantic.py --size all --mode raw
python src/04C_build_semantic.py --size all --mode seg_keep
python src/04C_build_semantic.py --size all --mode seg_drop
python src/04C_compare_semantic.py --size all --top 10
# 人工填写 output/4/semantic/relevance_labels_all_*.csv 的 0/1 标注后：
python src/04C_evaluate_semantic.py --size all

# 任务 4A：布尔 / 跳表 / 短语基准（9 条布尔式 × 4 编码 × 2 顺序，5 短语含 AND 对照）
python src/04A_boolean_retrieval.py --size all --bench
python src/04A_skip_storage.py --size all      # 跳表步长的指针存储开销
```

各脚本的参数、输入输出文件与任务要求的详细对照见 [src/README.md](src/README.md)。

## 说明

- `data/`（数据集与 data.zip）、`output/`（全部实验输出）、`report/`（实验报告）与 `report_data/`（报告数据附件）均按仓库根目录 `.gitignore` 的 `LAB_*` 规则排除，仅存本地；仓库中的代码 + `data/stopwords.txt` + 自行下载的数据集即可完整复现上述全部结果。
- 任务 5（基于用户画像的个性化推荐）为选做，本项目未实施。
