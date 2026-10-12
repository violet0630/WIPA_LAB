"""阶段4B（任务4B）：批量运行 keep/drop 两个 TF-IDF 模型的相同查询，并比较 Top-K 排名。

使用方法（在项目根目录执行，需先运行 04B_build_tfidf.py 生成 keep/drop 两套矩阵）：
  python src/04B_search_tfidf_implement.py --size 100 --top 10    # 100 篇快速自测
  python src/04B_search_tfidf_implement.py --size all --top 10    # 50,000 篇全量（核心数据）

输入：output/4/tfidf_fmm_{keep|drop}_{size}.npz 及同前缀 vocab/idf/docids 文件
输出：output/4/tfidf_comparison/{slug}_{variant}_{size}.json          逐查询逐模型结果
      output/4/tfidf_comparison/tfidf_keep_drop_comparison_{size}.json 汇总对照
12 条查询覆盖不同主题、长度与停用词比例，与 04C_compare_semantic.py 的
查询集保持一致，便于报告在同一查询集合上对比 TF-IDF 与语义检索。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "4"

# 与 04C_compare_semantic.py 共用的查询集（slug 需一致）。
QUERIES = [
    ("boarding", "收藏 登机牌"),
    ("boarding_stopwords", "有 收藏 登机牌 的 习惯 吗"),
    ("shanxi", "山西 古建 壁画"),
    ("skincare", "平价 护肤品 学生党"),
    ("travel_guide", "旅行 攻略 民宿"),
    ("kaoyan", "考研 英语 复习 经验"),
    ("coffee", "咖啡 做法 教程"),
    ("palace", "北京 故宫 门票"),
    ("fitness", "减肥 运动 食谱"),
    ("wardrobe", "衣柜 颜色 搭配"),
    ("jiuniang", "酒酿好处"),
    ("skincare_long", "适合学生党的平价护肤品有哪些推荐"),
]


def load_stage4b_module():
    """复用 04B_search_tfidf.py 的查询分词与停用词逻辑，保证口径一致。"""
    path = ROOT / "src" / "04B_search_tfidf.py"
    spec = importlib.util.spec_from_file_location("stage4b_search", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["stage4b_search"] = module
    spec.loader.exec_module(module)
    return module


def search(module, matrix, lookup, idf, stopwords, query: str, variant: str, top: int,
           titles: dict[str, str], docmap: list) -> dict:
    """与 04B_search_tfidf.py 完全一致的打分流程，但复用已加载的矩阵。"""
    tokens_before_stop = module.query_fmm(query)
    tokens = ([x for x in tokens_before_stop if x not in stopwords]
              if variant == "drop" else tokens_before_stop)
    counts = Counter(tokens)
    kept = [(lookup[t], float(freq) * float(idf[lookup[t]]))
            for t, freq in counts.items() if t in lookup]
    oov = [t for t in counts if t not in lookup]
    if not kept:
        return {"query": query, "query_tokens": tokens, "oov": oov, "variant": variant,
                "positive_score_documents": 0, "top_k": []}
    cols = np.asarray([col for col, _ in kept], dtype=np.int32)
    weights = np.asarray([val for _, val in kept], dtype=np.float64)
    weights /= np.linalg.norm(weights)
    scores = np.asarray(matrix[:, cols] @ weights).ravel()
    matched = np.flatnonzero(scores > 0)
    if not len(matched):
        return {"query": query, "query_tokens": tokens, "oov": oov, "variant": variant,
                "positive_score_documents": 0, "top_k": []}
    order = np.lexsort((matched, -scores[matched]))
    ranked = matched[order][:top]
    rows = []
    for rank, docno in enumerate(ranked, 1):
        docno = int(docno)
        docid = docmap[docno]
        rows.append({"rank": rank, "doc_no": docno, "doc_id": docid,
                     "cosine_similarity": round(float(scores[docno]), 6),
                     "title": titles.get(str(docid), "[未读取到原始标题]")})
    return {"query": query, "query_tokens": tokens, "oov": oov,
            "variant": variant, "positive_score_documents": int(len(matched)), "top_k": rows}


def compare(keep: dict, drop: dict, k: int) -> dict:
    ka = keep.get("top_k", [])[:k]
    kb = drop.get("top_k", [])[:k]
    # 比较内部编号；同一文档库的 keep/drop 映射保持一致。
    rank_a = {int(v["doc_no"]): i + 1 for i, v in enumerate(ka)}
    rank_b = {int(v["doc_no"]): i + 1 for i, v in enumerate(kb)}
    sa, sb = set(rank_a), set(rank_b)
    common = sorted(sa & sb, key=lambda doc: rank_a[doc])
    union = sa | sb
    return {
        "query": keep["query"],
        "k": k,
        "keep_positive_documents": keep.get("positive_score_documents", 0),
        "drop_positive_documents": drop.get("positive_score_documents", 0),
        "keep_result_count": len(ka),
        "drop_result_count": len(kb),
        "overlap_count": len(common),
        "jaccard_at_k": round(len(common) / len(union), 6) if union else 1.0,
        "shared_docs_and_ranks": [
            {"doc_no": doc, "keep_rank": rank_a[doc], "drop_rank": rank_b[doc]}
            for doc in common
        ],
        "keep_top3": ka[:3],
        "drop_top3": kb[:3],
    }


def load_model(variant: str, size: str, module):
    prefix = f"tfidf_fmm_{variant}_{size}"
    paths = {s: OUT / (prefix + s) for s in (".npz", "_vocab.json", "_idf.npy", "_docids.jsonl")}
    for p in paths.values():
        if not p.exists():
            raise FileNotFoundError(f"缺少模型文件 {p}，请先运行 04B_build_tfidf.py --variant {variant}")
    matrix = sparse.load_npz(paths[".npz"]).tocsr()
    vocabulary = json.loads(paths["_vocab.json"].read_text(encoding="utf-8"))
    idf = np.load(paths["_idf.npy"])
    docmap = [json.loads(line)["doc_id"] for line in paths["_docids.jsonl"]
              .read_text(encoding="utf-8").splitlines() if line.strip()]
    return matrix, {t: i for i, t in enumerate(vocabulary)}, idf, docmap


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", choices=["100", "1000", "all"], default="all")
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()
    if args.top <= 0:
        parser.error("--top 必须大于 0")

    module = load_stage4b_module()
    stopwords = module.stopword_set()
    dest = OUT / "tfidf_comparison"
    dest.mkdir(parents=True, exist_ok=True)

    models = {}
    for variant in ("keep", "drop"):
        started = time.perf_counter()
        models[variant] = load_model(variant, args.size, module)
        print(f"已加载 {variant} 模型（{time.perf_counter()-started:.1f}s）")

    # 先跑一遍收集全部查询的 Top-K doc_id 并集，再统一读取一次原始标题。
    wanted_titles: set[str] = set()
    cached_results = {}
    for variant, (matrix, lookup, idf, docmap) in models.items():
        for slug, query in QUERIES:
            res = search(module, matrix, lookup, idf, stopwords, query, variant,
                         args.top, {}, docmap)
            cached_results[(variant, slug)] = res
            for row in res.get("top_k", []):
                wanted_titles.add(str(row["doc_id"]))
    titles = module.titles_by_id(wanted_titles)

    comparisons = []
    for slug, query in QUERIES:
        print(f"\n=== 查询：{query} ===", flush=True)
        results = {}
        for variant in ("keep", "drop"):
            res = cached_results[(variant, slug)]
            for row in res.get("top_k", []):
                row["title"] = titles.get(str(row["doc_id"]), "[未读取到原始标题]")
            results[variant] = res
            file = dest / f"{slug}_{variant}_{args.size}.json"
            file.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"  {variant}: 正相似度文档 {res.get('positive_score_documents', 0):,}，"
                  f"Top-{len(res.get('top_k', []))} 已保存")
        row = compare(results["keep"], results["drop"], args.top)
        comparisons.append(row)
        print(f"  Top-{args.top} 交集数：{row['overlap_count']}，"
              f"Jaccard：{row['jaccard_at_k']:.4f}")

    summary = {"size": args.size, "top_k": args.top, "method": "FMM TF-IDF",
               "query_count": len(QUERIES), "comparisons": comparisons}
    summary_path = dest / f"tfidf_keep_drop_comparison_{args.size}.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n全部对照完成：{summary_path}")


if __name__ == "__main__":
    main()
