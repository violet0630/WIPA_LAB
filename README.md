# Web 信息处理与应用 · 课程实验

中国科学技术大学《Web 信息处理与应用》课程实验仓库。每个实验独立存放于 `LAB_xx/` 目录，自带代码、依赖清单与复现说明，互不依赖。

## 项目结构

```
├── LAB_01/                实验一：信息检索（Qilin-IR-50K 中文检索流水线）
│   ├── data/              数据集（以压缩包形式入库）
│   ├── src/               实验代码（任务↔代码↔输出对照见 LAB_01/src/README.md）
│   ├── output/            实验输出（本地生成，不入库）
│   ├── report/            实验报告 tex/pdf（本地保留，不入库）
│   ├── report_data/       报告引用的数据文件副本（本地保留，不入库）
│   ├── requirements.txt   本实验 Python 依赖
│   ├── 实验1.pdf          实验要求原文
│   └── README.md          实验介绍与复现命令
├── LAB_02/                （后续实验按此约定新增）
└── README.md              本文件
```

## 实验列表

| 目录 | 实验 | 主题 | 状态 |
|---|---|---|---|
| [LAB_01](LAB_01/) | 实验一 | 信息检索：分词、倒排索引、索引压缩、布尔 / TF-IDF / 语义检索 | 已完成 |

## 通用说明

- **环境**：每个实验目录自带 `requirements.txt`，可共用仓库根目录的一个虚拟环境，也可按实验分别创建：
  ```bash
  python -m venv .venv                       # 根目录共享环境；或 cd LAB_xx 后在各实验内创建
  .venv/Scripts/pip install -r LAB_01/requirements.txt   # Windows；Linux/macOS 用 .venv/bin/pip
  ```
- **入库范围**：仓库只跟踪代码、复现说明与复现必需的小文件（如停用词表）；各实验的数据集、`output/` 输出与 `report/` 报告均由 `.gitignore` 排除、仅存本地，克隆后按各实验 README 的命令即可完整复现。
- **新增实验**：新建 `LAB_xx/` 目录，沿用 `data / src / output / report / requirements.txt / README.md` 的结构约定即可被 `.gitignore` 的 `LAB_*` 规则统一覆盖。
