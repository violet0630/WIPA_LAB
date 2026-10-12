"""实验 1 / 阶段 3A（任务3-1）：分块外部排序构建位置倒排索引（节省内存）。

使用方法（在项目根目录执行）：
  python src/03_build_positions.py --size 100    # 100 篇快速自测
  python src/03_build_positions.py --size all    # 50,000 篇全量（核心数据）

输入：output/1/tokens_{method}_{size}.jsonl（阶段1分词结果）
      output/2/inverted_{method}_keep_{size}.jsonl、output/2/docids_{method}_{size}.jsonl（阶段2）
输出：output/3/positional_{method}_keep_{size}.jsonl        位置倒排索引
      output/3/positional_stats_{method}_keep_{size}.json   统计

每条 posting = [doc_num, title_positions, content_positions]。
位置是“规范化 + 保留停用词”后的字段内词项下标，从 0 开始；不是原文字节或字符位置。
"""
from __future__ import annotations

import argparse
import heapq
import itertools
import json
import math
import struct
import tempfile
import time
from contextlib import ExitStack
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" /"3"
INPUT_01 = ROOT / "output" /"1"
INPUT_02 = ROOT / "output" /"2"
RECORD = struct.Struct("<IIBI")  # term_id, doc_num, field(0=title/1=content), position


def get_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", choices=("fmm", "hmm"), default="fmm")
    p.add_argument("--size", choices=("100", "1000", "all"), default="100")
    p.add_argument("--shards", type=int, default=64, help="临时分桶数，默认 64")
    args = p.parse_args()
    if not 1 <= args.shards <= 128:
        p.error("--shards 必须在 1 至 128 之间")
    return args


def read_dictionary(stage2_path):
    """stage2 index sorted by term; only retain term/df, not all postings."""
    terms, expected_dfs = [], []
    with stage2_path.open("r", encoding="utf-8") as handle:
        for row_num, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            if terms and row["term"] <= terms[-1]:
                raise ValueError("阶段2倒排表没有按词项严格排序")
            terms.append(row["term"])
            expected_dfs.append(row["df"])
    return terms, expected_dfs


def iter_bucket_records(path):
    raw = path.read_bytes()
    if len(raw) % RECORD.size:
        raise ValueError(f"临时桶文件损坏：{path}")
    return sorted(RECORD.iter_unpack(raw))


def records_to_rows(records, terms, expected_dfs):
    """records 已按 (term_id,doc_num,field,pos) 升序；按词分组并组装 posting。"""
    for tid, grouped in itertools.groupby(records, key=lambda rec: rec[0]):
        postings = []
        for _, doc_num, field, pos in grouped:
            if field not in (0, 1):
                raise ValueError("字段编号无效")
            if not postings or postings[-1][0] != doc_num:
                postings.append([doc_num, [], []])
            current_field = postings[-1][field + 1]
            if current_field and pos <= current_field[-1]:
                raise ValueError("位置未严格升序或发生重复")
            current_field.append(pos)
        df = len(postings)
        if df != expected_dfs[tid]:
            raise AssertionError(f"词项 {terms[tid]!r} df 不一致: 阶段2={expected_dfs[tid]}, 位置={df}")
        yield {
            "term_id": tid,
            "term": terms[tid],
            "df": df,
            "postings": postings,
            # 单层跳表与阶段2采用同一规则；实际边可从 df/step 重新生成
            "skip_step": max(2, math.isqrt(df)),
        }


def read_rows(path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def build(args):
    started = time.perf_counter()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    source = INPUT_01 / f"tokens_{args.method}_{args.size}.jsonl"
    stage2 = INPUT_02 / f"inverted_{args.method}_keep_{args.size}.jsonl"
    docmap = INPUT_02 / f"docids_{args.method}_{args.size}.jsonl"
    target = OUTPUT / f"positional_{args.method}_keep_{args.size}.jsonl"
    stats_path = OUTPUT / f"positional_stats_{args.method}_keep_{args.size}.json"
    for path in [source, stage2, docmap]:
        if not path.is_file():
            raise FileNotFoundError(f"缺少输入文件：{path}")

    terms, expected_dfs = read_dictionary(stage2)
    term_to_id = {term: tid for tid, term in enumerate(terms)}
    print(f"词典加载成功：{len(terms):,} 词，开始生成位置记录", flush=True)
    total_tokens = 0
    doc_count = 0
    with tempfile.TemporaryDirectory(dir=OUTPUT, prefix="_stage3A_tmp_") as temp_dir:
        temp = Path(temp_dir)
        with ExitStack() as stack:
            buckets = [stack.enter_context((temp / f"shard_{i:03d}.bin").open("wb")) for i in range(args.shards)]
            buffers = [bytearray() for _ in buckets]
            with source.open("r", encoding="utf-8-sig") as src, docmap.open("r", encoding="utf-8") as mapping:
                mapping_iter = (json.loads(row) for row in mapping if row.strip())
                for lineno, line in enumerate(src, start=1):
                    if not line.strip():
                        continue
                    doc = json.loads(line)
                    actual_mapping = next(mapping_iter, None)
                    if actual_mapping is None or actual_mapping["doc_num"] != doc_count or actual_mapping["doc_id"] != doc["doc_id"]:
                        raise ValueError(f"第 {lineno} 行文档编号或 ID 与阶段2映射不一致")
                    for field, key in enumerate(("title_keep", "content_keep")):
                        for pos, term in enumerate(doc[key]):
                            try:
                                tid = term_to_id[term]
                            except KeyError as err:
                                raise KeyError(f"阶段2词典缺少词项 {term!r}，第 {lineno} 行") from err
                            shard = tid % args.shards
                            buf = buffers[shard]
                            buf.extend(RECORD.pack(tid, doc_count, field, pos))
                            if len(buf) >= 256 * 1024:
                                buckets[shard].write(buf)
                                buf.clear()
                            total_tokens += 1
                    doc_count += 1
                    if doc_count % 5000 == 0:
                        print(f"已扫描 {doc_count:,} 篇文档", flush=True)
                if next(mapping_iter, None) is not None:
                    raise ValueError("阶段2文档映射中的文档比当前位置索引多")
            for bucket, buf in zip(buckets, buffers):
                if buf:
                    bucket.write(buf)
                    buf.clear()
        print("开始分桶排序并汇总词项位置……", flush=True)
        intermediate_files = []
        for shard in range(args.shards):
            binary_path = temp / f"shard_{shard:03d}.bin"
            text_path = temp / f"sorted_{shard:03d}.jsonl"
            with text_path.open("w", encoding="utf-8") as out:
                for row in records_to_rows(iter_bucket_records(binary_path), terms, expected_dfs):
                    out.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            intermediate_files.append(text_path)
            binary_path.unlink()  # 尽早释放临时磁盘空间
        temp_target = target.with_suffix(target.suffix + ".tmp")
        term_count = posting_pairs = positions_count = 0
        try:
            with temp_target.open("w", encoding="utf-8") as out:
                merged = heapq.merge(*(read_rows(path) for path in intermediate_files), key=lambda r: r["term_id"])
                for row in merged:
                    if row["term_id"] != term_count:
                        raise AssertionError("词项 ID 丢失、重复或排序错误")
                    out.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                    term_count += 1
                    posting_pairs += row["df"]
                    positions_count += sum(len(p[1]) + len(p[2]) for p in row["postings"])
            if term_count != len(terms) or total_tokens != positions_count or posting_pairs != sum(expected_dfs):
                raise AssertionError("位置索引与阶段2索引或输入词项数量不一致")
            temp_target.replace(target)
        except Exception:
            temp_target.unlink(missing_ok=True)
            raise

    stats = {
        "method": args.method,
        "variant": "keep",
        "size": args.size,
        "doc_count": doc_count,
        "vocabulary_size": term_count,
        "posting_pairs_unique_term_document": posting_pairs,
        "input_token_occurrences": total_tokens,
        "indexed_position_occurrences": positions_count,
        "shards": args.shards,
        "position_definition": "0-based token indices in normalized title_keep/content_keep separately",
        "positional_jsonl_bytes": target.stat().st_size,
        "elapsed_seconds_including_io": round(time.perf_counter() - started, 4),
    }
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== 阶段3A位置索引统计 ===")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print("输出位置索引：", target)


if __name__ == "__main__":
    build(get_args())
