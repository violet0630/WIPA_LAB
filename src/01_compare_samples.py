"""阶段1：人工对比查看 FMM 与 HMM 的分词样例（读取阶段1输出）。

使用方法（在项目根目录执行）：
  python src/01_compare_samples.py

输入：output/1/segmentation_samples_100.json（由 01_segment.py --limit 100 生成）
输出：控制台打印前 5 篇文档两种分词方法的原始/规范化/去停用词结果
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "output" / "1" / "segmentation_samples_100.json"

samples = json.loads(path.read_text(encoding="utf-8"))

for i, sample in enumerate(samples[:5], 1):
    print(f"\n========== 文档 {i} ==========")
    print("标题：", sample["title"])

    for method in ["fmm", "hmm"]:
        result = sample[method]
        print(f"\n{method.upper()}：")
        print("原始分词：", " / ".join(result["title_raw"]))
        print("规范化：", " / ".join(result["title_keep"]))
        print("去停用词：", " / ".join(result["title_drop"]))
