# Web 信息处理与应用 · 实验一：信息检索

基于清华大学 THUIR 与小红书开源的 **Qilin-IR-50K** 教学子集（50,000 篇中文笔记及其搜索、行为数据），实现从分词规范化、倒排索引构建与压缩，到布尔 / TF-IDF / 预训练语义检索的完整信息检索流水线，并完成多角度量化评估。

## 目录结构

```
实验/
├── data/                  Qilin-IR-50K 数据集（7 个 JSONL 文件 + 停用词表）
│   └── optional/          选做任务（个性化推荐）的行为数据，本次未使用
├── src/                   全部实验代码（任务↔代码↔输出对照见 src/README.md）
├── output/                各阶段输出：1 分词 / 2 倒排索引 / 3 位置与压缩 / 4 检索实验
│   └── 4/semantic/        语义向量、检索对照与人工相关性标注
├── report/                实验报告（web_report01.tex → web_report01.pdf）
├── requirements.txt       Python 依赖
└── .gitignore
```

## 环境配置

- Python ≥ 3.13（项目自带 `.venv`，Windows 11 / CPU 推理即可复现全部结果）
- TeX 发行版（MiKTeX / TeX Live，含 `ctex`、`pgfplots`），用于编译报告

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt   # Windows；Linux/macOS 用 .venv/bin/pip
```

## 快速开始

所有脚本均在**项目根目录**执行。规模约定：`--size 100 / 1000` 为正确性自测，`--size all`（50,000 篇）为核心实验数据；`--variant keep / drop` 表示保留 / 去除停用词。完整执行顺序与各脚本参数见 [src/README.md](src/README.md)。

```bash
# 任务 1：分词与规范化（FMM + HMM，keep/drop 两套）
python src/01_segment.py --limit 0

# 任务 2：文档级倒排索引 + 单层跳表
python src/02_build_index.py --size all
python src/02_build_index.py --size all --variant drop

# 任务 3：位置索引 + 四组索引压缩（Raw/Front/VByte/Combined）与解压验证
python src/03_build_positions.py --size all
python src/03_compress_index.py --size all
python src/03_verify_compression.py --size all --full --mode both

# 任务 4B：TF-IDF 矩阵与批量 keep/drop 检索对照（12 条查询）
python src/04B_build_tfidf.py --size all --variant keep
python src/04B_build_tfidf.py --size all --variant drop
python src/04B_search_tfidf_implement.py --size all --top 10

# 任务 4C：BGE 语义向量（三种输入方式）与检索对照、P@10 评价
python src/04C_build_semantic.py --size all --mode raw
python src/04C_build_semantic.py --size all --mode seg_keep
python src/04C_build_semantic.py --size all --mode seg_drop
python src/04C_compare_semantic.py --size all --top 10
# 人工填写 output/4/semantic/relevance_labels_all_*.csv 后：
python src/04C_evaluate_semantic.py --size all

# 任务 4A：布尔 / 跳表 / 短语基准（9 条布尔式 × 4 编码 × 2 顺序，5 短语含 AND 对照）
python src/04A_boolean_retrieval.py --size all --bench
python src/04A_skip_storage.py --size all      # 跳表步长的指针存储开销
```

任务与代码、输出的详细对照表见 [src/README.md](src/README.md)。

## 实验报告

```bash
cd report
xelatex -interaction=nonstopmode web_report01.tex   # 运行两遍以生成目录/引用
```

编译产物为 `report/web_report01.pdf`；报告中的每个定量结论均标注了对应的输出文件，可直接回溯核验。

## 主要结果摘要（50,000 篇全量）

| 维度 | 结果 |
|---|---|
| 分词 | FMM 比 HMM 快 2.56×，词表小 3.37×（115,215 vs 388,632） |
| 索引压缩 | Front Coding + Gap/VByte 组合节省 **68.35%**（79.66 → 25.21 MB） |
| 短语检索 | 位置索引相对普通 AND 过滤 74–98% 命中文档 |
| 跳表 | √df 步长以 2.8% 指针存储换极端失配查询 89% 比较次数下降 |
| 检索质量 | 12 条查询、333 条人工标注：Raw 语义编码 P@10 **0.533** / MRR@10 **1.000**，优于词项聚合（0.300 / 0.561）与 TF-IDF（0.275 / 0.750） |

## 说明

- 原始数据（`data/`）与大体积中间产物（分词结果、索引、向量等）不含体积意义上的完整入库，仅统计与汇总 JSON 随仓库跟踪；全部输出均可由上述脚本从原始数据复现。
- 任务 5（基于用户画像的个性化推荐）为选做，本项目未实施。
- 数据集许可证与引用要求见 [Qilin 数据集主页](https://huggingface.co/datasets/THUIR/Qilin)。
