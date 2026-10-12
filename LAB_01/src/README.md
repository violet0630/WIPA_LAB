# 实验一代码与数据说明

数据：Qilin-IR-50K（`data/` 下 7 个 JSONL 文件）。所有脚本均在 **`LAB_01/` 目录**（`src/` 的上一级）执行，
形如 `python src/<脚本>.py`；脚本内部以自身位置解析路径，移动 `LAB_01` 整体目录不影响运行。

规模约定：`--size 100 / 1000` 为小规模**正确性自测**（两套分词、索引、压缩、检索
结果与全量流程完全一致，仅文档数不同）；`--size all`（50,000 篇）为**核心实验数据**，
报告中的定量结论全部来自 `all` 规模。`--variant keep/drop` 表示保留/去除停用词。

## 任务 ↔ 代码 ↔ 输出对照表

| 任务 | 脚本 | 输入 | 输出（均相对 `output/`） | 已运行规模 |
|---|---|---|---|---|
| 数据体检 | `00_check_data.py` | `data/*.jsonl` | `data_check.json` | 全量 |
| **任务1** 分词与规范化 | `01_segment.py` | `data/documents.jsonl`、`data/stopwords.txt` | `1/tokens_{fmm,hmm}_{size}.jsonl`、`1/segmentation_stats_{size}.json`、`1/segmentation_samples_{size}.json` | 100 / 1000 / all |
| 任务1 结果查看 | `01_compare_samples.py` | `1/segmentation_samples_100.json` | 控制台 | — |
| **任务2** 倒排表+跳表 | `02_build_index.py` | `1/tokens_{m}_{size}.jsonl` | `2/inverted_{m}_{v}_{size}.jsonl`、`2/docids_{m}_{size}.jsonl`、`2/index_stats_{m}_{v}_{size}.json` | fmm: keep 100 / keep all / drop all |
| 任务2 结构抽查 | `02_probe_index.py` | 任务2输出 | 控制台（含 AND/OR 演示） | — |
| **任务3-1** 位置索引 | `03_build_positions.py` | `1/tokens_*.jsonl`、任务2输出 | `3/positional_{m}_keep_{size}.jsonl`、`3/positional_stats_*.json` | 100 / all |
| 任务3-1 一致性抽查 | `03_check_positions.py` | 任务2+3A输出 | 控制台 | — |
| **任务3-2** 编解码库 | `stage3b_codec.py` | —（被下列脚本复用） | — | — |
| **任务3-2** 索引压缩 | `03_compress_index.py` | `3/positional_*.jsonl` | `3/compressed_{m}_keep_{size}/{raw,front,vbyte,both}/`、`3/compression_summary_*.json` | 100 / all |
| 任务3-2 解压验证 | `03_verify_compression.py` | 任务3A+3B输出 | `3/compression_verified_*_{full,sample}.json` | 100 sample / all sample+full(both) |
| **任务4A** 布尔/跳表/短语基准 | `04A_boolean_retrieval.py --bench` | `3/compressed_*/`、`2/docids_*.jsonl` | `4/boolean_benchmark_{m}_{size}.json`（9 布尔式 × 4 编码 × 2 顺序 × 2 跳表；3 查询 × 6 步长；5 短语含 AND 对照） | all |
| **任务4A-4** 跳表存储开销 | `04A_skip_storage.py` | `2/inverted_*.jsonl`、`3/compressed_*/raw/posts.bin` | `4/skip_storage_{m}_{size}.json` | all |
| **任务4B** TF-IDF 矩阵 | `04B_build_tfidf.py` | `1/tokens_fmm_{size}.jsonl` | `4/tfidf_fmm_{v}_{size}.npz` + `_vocab/_idf/_docids/_stats` | keep/drop × 100 / 1000 / all |
| 任务4B 单查询 | `04B_search_tfidf.py` | 上行输出 | 控制台 + `4/tfidf_result_{v}_{size}.json` | — |
| **任务4B** 批量 keep/drop 对照 | `04B_search_tfidf_implement.py` | `4/tfidf_fmm_*.npz` 等 | `4/tfidf_comparison/{slug}_{v}_{size}.json`、`tfidf_keep_drop_comparison_{size}.json` | all（12 查询） |
| **任务4C** 语义向量构建 | `04C_build_semantic.py`（raw / seg_keep / seg_drop 三种 mode） | `data/documents.jsonl`、`1/tokens_*.jsonl` | `4/semantic/embeddings_{mode}_{size}_{key}.npy`、`term_vectors_keep_*.npy`、`documents_{size}.jsonl`、`stats_*.json` | 100 / 1000 / all |
| **任务4C** 检索对照 | `04C_compare_semantic.py` | 上一行输出、`4/tfidf_comparison/` | `4/semantic/comparison_{size}_{key}.json`、`relevance_labels_{size}_{key}.csv` | all（12 查询） |
| **任务4C** 指标计算 | `04C_evaluate_semantic.py` | 上行输出（需人工标注 CSV） | `4/semantic/precision_{size}_{key}.json` | all |

任务5（个性化推荐）为选做，未实施。

## 复现全量实验的执行顺序

```bash
python src/01_segment.py --limit 0                    # 任务1（fmm+hmm 全量分词）
python src/02_build_index.py --size all               # 任务2（keep；drop 可选）
python src/02_build_index.py --size all --variant drop
python src/03_build_positions.py --size all           # 任务3-1
python src/03_compress_index.py --size all            # 任务3-2（四组编码）
python src/03_verify_compression.py --size all --full --mode both
python src/04B_build_tfidf.py --size all --variant keep   # 任务4B
python src/04B_build_tfidf.py --size all --variant drop
python src/04C_build_semantic.py --size all --mode raw     # 任务4C（三种 mode）
python src/04C_build_semantic.py --size all --mode seg_keep
python src/04C_build_semantic.py --size all --mode seg_drop
python src/04B_search_tfidf_implement.py --size all --top 10   # 任务4B 批量对照
python src/04C_compare_semantic.py --size all --top 10         # 任务4C 对照（后人工标注 CSV）
python src/04C_evaluate_semantic.py --size all                 # P@10
python src/04A_boolean_retrieval.py --size all --bench         # 任务4A 基准
python src/04A_skip_storage.py --size all                      # 任务4A(4) 存储开销
```

## 一致性保证

- 每个阶段的输出文件内含自身统计（`*_stats.json` / `manifest.json`），报告数字均可回溯。
- `02_probe_index` / `03_check_positions` / `03_verify_compression` 完成 df、文档编号、
  位置、压缩解码的逐项一致性校验；`04A --bench` 内置跨编码/顺序/步长的严格结果一致性检查。
