"""实验一 / 阶段2：验证倒排表、跳表指针，演示 AND / OR 双指针归并（正确性抽查）。

使用方法（在项目根目录执行）：
  python src/02_probe_index.py --size all
  python src/02_probe_index.py --size all --terms 登机牌 习惯

输入：output/2/inverted_{method}_{variant}_{size}.jsonl、output/2/docids_{method}_{size}.jsonl
输出：控制台输出全量词项的结构校验结果与指定词项的 AND/OR 归并演示
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output"/"2"



def intersect(a: list[int], b: list[int]) -> list[int]:
    """有序倒排表交集（AND），复杂度 O(len(a)+len(b))。"""
    result = []
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            result.append(a[i])
            i += 1
            j += 1
        elif a[i] < b[j]:
            i += 1
        else:
            j += 1
    return result


def union(a: list[int], b: list[int]) -> list[int]:
    """有序倒排表并集（OR），复杂度 O(len(a)+len(b))。"""
    result = []
    i = j = 0
    while i < len(a) or j < len(b):
        if j >= len(b) or (i < len(a) and a[i] < b[j]):
            x = a[i]
            i += 1
        elif i >= len(a) or b[j] < a[i]:
            x = b[j]
            j += 1
        else:
            x = a[i]
            i += 1
            j += 1
        result.append(x)
    return result


def validate(row: dict) -> None:
    postings = row["postings"]
    if row["df"] != len(postings):
        raise AssertionError(f"{row['term']}：df 与 postings 长度不一致")
    if any(x >= y for x, y in zip(postings, postings[1:])):
        raise AssertionError(f"{row['term']}：倒排表未严格递增或存在重复")
    for start, end in row["skip_edges"]:
        if not (0 <= start < end < len(postings)):
            raise AssertionError(f"{row['term']}：跳表指针越界")
        if end - start != row["skip_step"]:
            raise AssertionError(f"{row['term']}：跳表步长不一致")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["fmm", "hmm"], default="fmm")
    parser.add_argument("--size", choices=["100", "1000", "all"], default="all")
    parser.add_argument("--variant", choices=["keep", "drop"], default="keep")
    parser.add_argument("--terms", nargs=2, default=["登机牌", "习惯"], metavar=("TERM1", "TERM2"))
    args = parser.parse_args()
    index_path = OUTPUT / f"inverted_{args.method}_{args.variant}_{args.size}.jsonl"
    map_path = OUTPUT / f"docids_{args.method}_{args.size}.jsonl"
    if not index_path.is_file() or not map_path.is_file():
        raise FileNotFoundError("未找到阶段2索引，请先运行 02_build_index.py")

    targets = set(args.terms)
    records = {}
    total_terms_checked = 0
    with index_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            validate(row)
            total_terms_checked += 1
            if row["term"] in targets:
                records[row["term"]] = row

    numbers_to_ids = {}
    with map_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            rec = json.loads(line)
            numbers_to_ids[rec["doc_num"]] = rec["doc_id"]

    print(f"已验证 {total_terms_checked:,} 个词项的倒排表与跳表指针")
    postings_for_queries = []
    for term in args.terms:
        row = records.get(term)
        if row is None:
            print(f"词项 {term!r}：未出现在此索引，df=0")
            postings_for_queries.append([])
            continue
        postings_for_queries.append(row["postings"])
        print(f"词项 {term!r}：df={row['df']}, step={row['skip_step']}, "
              f"跳表边数={len(row['skip_edges'])}")
        print("前10个内部文档编号：", row["postings"][:10])
        print("前3条跳表边：", row["skip_edges"][:3])

    a, b = postings_for_queries
    for operator, result in [("AND", intersect(a, b)), ("OR", union(a, b))]:
        print(f"\n{args.terms[0]} {operator} {args.terms[1]}：命中 {len(result)} 篇")
        print("前10个原始文档ID：", [numbers_to_ids[num] for num in result[:10]])
    print("\n说明：本程序只验证跳表结构，AND/OR 暂未利用跳跃加速；后续任务4进行加速对比。")


if __name__ == "__main__":
    main()
