"""实验一 / 阶段2（任务2）：构建倒排索引和单层跳表指针。

使用方法（在实验项目根目录执行）：
  python src/02_build_index.py --size 100            # 100 篇快速自测
  python src/02_build_index.py --size all            # 50,000 篇全量（核心数据）
  python src/02_build_index.py --size all --variant drop

输入：output/1/tokens_{method}_{size}.jsonl（阶段1分词结果）
输出：output/2/inverted_{method}_{variant}_{size}.jsonl   倒排表（含跳表边）
      output/2/docids_{method}_{size}.jsonl              内部编号↔原始 doc_id 映射
      output/2/index_stats_{method}_{variant}_{size}.json 索引统计

说明：阶段2采用布尔索引，仅存词项所在文档编号（不存出现次数和位置）；
      阶段3要用 title_keep/content_keep 重新添加词项位置。
      跳表默认步长为每条倒排表的 floor(sqrt(df))，可用 --skip-step 固定。
"""
from __future__ import annotations

import argparse
import json
import math
import time
from array import array
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output"/"2"
INPUT =ROOT/"output"/"1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建实验一阶段2的倒排表与单层跳表")
    parser.add_argument("--method", choices=["fmm", "hmm"], default="fmm")
    parser.add_argument("--size", choices=["100", "1000", "all"], default="all")
    parser.add_argument("--variant", choices=["keep", "drop"], default="keep")
    parser.add_argument("--skip-step", type=int, default=0,
                        help="跳表固定步长；0 表示按每条倒排表的 sqrt(df) 自动选择")
    args = parser.parse_args()
    if args.skip_step < 0 or args.skip_step == 1:
        parser.error("--skip-step 必须为 0（自动），或不小于 2")
    return args


def make_skip_edges(posting_count: int, fixed_step: int) -> tuple[int, list[list[int]]]:
    """返回跳表边 [起点下标, 目标下标]，两个下标均指向 postings 列表。"""
    step = fixed_step if fixed_step else max(2, math.isqrt(posting_count))
    if posting_count < 4:
        return step, []
    # 最后一个不完整间隔不加跳跃边；比较简单，避免越界。
    links = [[i, i + step] for i in range(0, posting_count - step, step)]
    return step, links


def main() -> None:
    args = parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    stem = f"{args.method}_{args.size}"
    name = f"{args.method}_{args.variant}_{args.size}"
    source = INPUT /f"tokens_{stem}.jsonl"
    index_path = OUTPUT / f"inverted_{name}.jsonl"
    docmap_path = OUTPUT / f"docids_{stem}.jsonl"
    stats_path = OUTPUT / f"index_stats_{name}.json"
    if not source.is_file():
        raise FileNotFoundError(f"找不到分词结果：{source}\n请先生成 tokens_{stem}.jsonl")

    started = time.perf_counter()
    # array('I') 以紧凑整数数组存储文档编号，减少 5 万篇数据的内存压力。
    inverted: dict[str, array] = {}
    count_docs = 0
    input_tokens = 0
    seen_document_ids: set[str] = set()
    temporary_docmap = docmap_path.with_suffix(docmap_path.suffix + ".tmp")
    print(f"读取分词数据：{source}", flush=True)
    try:
        with source.open("r", encoding="utf-8-sig") as inp, temporary_docmap.open("w", encoding="utf-8") as mapping:
            for lineno, line in enumerate(inp, start=1):
                if not line.strip():
                    continue
                doc = json.loads(line)
                if not isinstance(doc, dict):
                    raise ValueError(f"第 {lineno} 行不是 JSON 对象")
                doc_id = doc["doc_id"]
                # 不修改 ID；对不同 JSON 类型的 ID 进行严格区分。
                unique_id = json.dumps(doc_id, ensure_ascii=False, sort_keys=True)
                if unique_id in seen_document_ids:
                    raise ValueError(f"重复的文档 ID：{doc_id!r}；源文件第 {lineno} 行")
                seen_document_ids.add(unique_id)

                title_key, body_key = f"title_{args.variant}", f"content_{args.variant}"
                title_tokens, body_tokens = doc[title_key], doc[body_key]
                if not isinstance(title_tokens, list) or not isinstance(body_tokens, list):
                    raise TypeError(f"第 {lineno} 行的 {title_key}/{body_key} 应为数组")
                input_tokens += len(title_tokens) + len(body_tokens)
                # 每篇文档只记录词项是否出现：一个词即使出现 20 次也只追加一次 doc_num。
                terms = set(title_tokens)
                terms.update(body_tokens)
                for term in terms:
                    if not isinstance(term, str) or not term:
                        raise ValueError(f"第 {lineno} 行出现无效词项：{term!r}")
                    posting = inverted.get(term)
                    if posting is None:
                        posting = array("I")
                        inverted[term] = posting
                    posting.append(count_docs)

                mapping.write(json.dumps({"doc_num": count_docs, "doc_id": doc_id}, ensure_ascii=False) + "\n")
                count_docs += 1
                if count_docs % 5000 == 0:
                    print(f"已处理 {count_docs:,} 篇；词汇表 {len(inverted):,} 词", flush=True)
        temporary_docmap.replace(docmap_path)
    except Exception:
        temporary_docmap.unlink(missing_ok=True)
        raise

    print("开始输出有序倒排表与跳表指针……", flush=True)
    total_postings = 0
    total_skip_edges = 0
    most_frequent_term = None
    max_df = 0
    temporary_index = index_path.with_suffix(index_path.suffix + ".tmp")
    try:
        with temporary_index.open("w", encoding="utf-8") as out:
            for term in sorted(inverted):
                posting = inverted[term]
                df = len(posting)
                # 文档按输入先后编号；每个词在每篇文档中只追加一次，保证严格升序。
                step, skip_edges = make_skip_edges(df, args.skip_step)
                row = {
                    "term": term,
                    "df": df,
                    "postings": posting.tolist(),
                    "skip_step": step,
                    "skip_edges": skip_edges,
                }
                out.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                total_postings += df
                total_skip_edges += len(skip_edges)
                if df > max_df:
                    max_df, most_frequent_term = df, term
        temporary_index.replace(index_path)
    except Exception:
        temporary_index.unlink(missing_ok=True)
        raise

    stats = {
        "method": args.method,
        "variant": args.variant,
        "size": args.size,
        "doc_count": count_docs,
        "input_token_occurrences": input_tokens,
        "vocabulary_size": len(inverted),
        "posting_pairs_unique_term_document": total_postings,
        "skip_edges_total": total_skip_edges,
        "skip_step_rule": "floor(sqrt(df)), min 2" if args.skip_step == 0 else f"fixed {args.skip_step}",
        "max_df_term": most_frequent_term,
        "max_df": max_df,
        "index_jsonl_bytes": index_path.stat().st_size,
        "document_map_bytes": docmap_path.stat().st_size,
        "elapsed_seconds_including_io": round(time.perf_counter() - started, 4),
    }
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== 阶段2索引统计 ===")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print("\n索引：", index_path)
    print("文档编号映射：", docmap_path)
    print("实验统计：", stats_path)


if __name__ == "__main__":
    main()
