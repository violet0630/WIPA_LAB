"""阶段4C（任务4C）：相同查询比较原文、先分词(保留/去除停用词)、以及阶段4B TF-IDF。

使用方法（在项目根目录执行；必须先运行 04C_build_semantic.py 的三种 mode，并使用同一个 --model）：
  python src/04C_compare_semantic.py --size 100 --top 10    # 100 篇快速自测
  python src/04C_compare_semantic.py --size all --top 10    # 50,000 篇全量（核心数据）

输入：output/4/semantic/ 下三种 mode 的向量与统计、output/4/tfidf_comparison/{slug}_keep_{size}.json
输出：output/4/semantic/comparison_{size}_{modelkey}.json          检索结果与 Jaccard 对照
      output/4/semantic/relevance_labels_{size}_{modelkey}.csv     人工相关性标注模板
12 条查询与 04B_search_tfidf_implement.py 的查询集一致（slug 相同）；
标注完成后运行 04C_evaluate_semantic.py 计算 P@10。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

# 复用本阶段脚本对 model_key、向量聚合与模型加载的实现。
ROOT = Path(__file__).resolve().parents[1]
SEM = ROOT / 'output' /"4"/ 'semantic'
_stage_spec = importlib.util.spec_from_file_location('stage4c_builder', ROOT / 'src' / '04C_build_semantic.py')
STAGE = importlib.util.module_from_spec(_stage_spec)
_stage_spec.loader.exec_module(STAGE)

QUERIES = [
    ('boarding', '收藏 登机牌'),
    ('boarding_stopwords', '有 收藏 登机牌 的 习惯 吗'),
    ('shanxi', '山西 古建 壁画'),
    ('skincare', '平价 护肤品 学生党'),
    ('travel_guide', '旅行 攻略 民宿'),
    ('kaoyan', '考研 英语 复习 经验'),
    ('coffee', '咖啡 做法 教程'),
    ('palace', '北京 故宫 门票'),
    ('fitness', '减肥 运动 食谱'),
    ('wardrobe', '衣柜 颜色 搭配'),
    ('jiuniang', '酒酿好处'),
    ('skincare_long', '适合学生党的平价护肤品有哪些推荐'),
]
MODES = ('raw', 'seg_keep', 'seg_drop')


def load_fmm_helpers():
    path = ROOT / 'src' / '04B_search_tfidf.py'
    if not path.exists():
        raise FileNotFoundError('缺少阶段4B代码 src/04B_search_tfidf.py')
    spec = importlib.util.spec_from_file_location('stage4b_search', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_doc_metadata(size: str):
    path = SEM / f'documents_{size}.jsonl'
    docs = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                if rec['doc_no'] != len(docs):
                    raise AssertionError('文档编号映射不连续')
                docs.append(rec)
    return docs


def embed_query(model, raw: str, mode: str, fmm, lookup, vectors, dim: int, batch_size: int):
    if mode == 'raw':
        return STAGE.unit(STAGE.encode(model, [raw], batch_size)[0]), []
    terms = fmm.query_fmm(raw)
    if mode == 'seg_drop':
        stop = fmm.stopword_set()
        terms = [x for x in terms if x not in stop]
    counts = Counter(terms)
    if not counts:
        return np.zeros(dim, dtype=np.float32), terms
    known = {x: c for x, c in counts.items() if x in lookup}
    unknown = [x for x in counts if x not in lookup]
    collected = np.zeros(dim, dtype=np.float32)
    if known:
        keys = list(known)
        indices = [lookup[x] for x in keys]
        weights = np.asarray([known[x] for x in keys], dtype=np.float32)
        collected += (np.asarray(vectors[indices], dtype=np.float32) * weights[:, None]).sum(axis=0)
    if unknown:
        encoded = STAGE.encode(model, unknown, batch_size)
        weights = np.asarray([counts[x] for x in unknown], dtype=np.float32)
        collected += (encoded * weights[:, None]).sum(axis=0)
    return STAGE.unit(collected / len(terms)), terms


def jaccard(left, right, k):
    a = {x['doc_no'] for x in left[:k]}
    b = {x['doc_no'] for x in right[:k]}
    return round(len(a & b) / len(a | b), 6) if a | b else 1.0


def rank_docs(doc_vectors: np.ndarray, valid_docs: np.ndarray, query: np.ndarray, docs: list[dict], k: int):
    if not np.any(query):
        return []
    scores = np.asarray(doc_vectors @ query, dtype=np.float32)
    # 排除没有词项导致的零向量文档
    candidates = valid_docs
    if len(candidates) == 0:
        return []
    # 先分区，再对Top候选按分数降序、doc_no升序稳定排序
    vals = scores[candidates]
    if len(candidates) > k:
        threshold = float(np.partition(vals, -k)[-k])
        # 包括恰好处于边界分数的同分文档，保证稳定排名
        candidate_docs = candidates[vals >= threshold]
    else:
        candidate_docs = candidates
    indices = np.lexsort((candidate_docs, -scores[candidate_docs]))
    ranked = candidate_docs[indices][:k]
    return [{
        'rank': i+1, 'doc_no': int(n), 'doc_id': docs[int(n)]['doc_id'],
        'cosine_similarity': round(float(scores[int(n)]), 6),
        'title': docs[int(n)]['title'],
        'content_preview': docs[int(n)]['content_preview'],
    } for i,n in enumerate(ranked)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--size', choices=['100','1000','all'], default='100')
    ap.add_argument('--model', default='BAAI/bge-small-zh-v1.5')
    ap.add_argument('--device', default='auto')
    ap.add_argument('--top', type=int, default=10)
    ap.add_argument('--batch-size', type=int, default=32)
    args = ap.parse_args()
    if args.top < 1:
        ap.error('--top 必须 >0')
    key = STAGE.model_key(args.model)
    stats = {}
    for mode in MODES:
        f = SEM / f'stats_{mode}_{args.size}_{key}.json'
        if not f.exists():
            raise FileNotFoundError(f'缺少 {f}。请先运行该模式的 04C_build_semantic.py。')
        stats[mode] = json.loads(f.read_text(encoding='utf-8'))
    max_lens = {v['max_seq_length'] for v in stats.values()}
    if len(max_lens) != 1:
        raise AssertionError('不同模式的 max_seq_length 不相同，必须统一设置')
    if len({v['embedding_dim'] for v in stats.values()}) != 1:
        raise AssertionError('向量维度不一致')
    doc_vectors = {mode: np.load(SEM / f'embeddings_{mode}_{args.size}_{key}.npy', mmap_mode='r') for mode in MODES}
    docs = load_doc_metadata(args.size)
    valid_docs = {mode: np.flatnonzero(np.any(doc_vectors[mode] != 0, axis=1))
                  for mode in MODES}
    if any(v.shape[0] != len(docs) for v in doc_vectors.values()):
        raise AssertionError('文档数不一致')

    model = STAGE.get_model(args.model, args.device, next(iter(max_lens)))
    fmm = load_fmm_helpers()
    vocfile = SEM / f'vocab_keep_{args.size}_{key}.json'
    cachefile = SEM / f'term_vectors_keep_{args.size}_{key}.npy'
    vocab = json.loads(vocfile.read_text(encoding='utf-8'))
    lookup = {word: i for i, word in enumerate(vocab)}
    term_vectors = np.load(cachefile, mmap_mode='r')
    dim = stats['raw']['embedding_dim']
    summary = {'model': args.model, 'size': args.size, 'top_k': args.top,
               'max_seq_length': next(iter(max_lens)), 'queries': []}
    labels = {}  # 生成去重后的人工相关性标注模板，不猜测答案。
    for slug, text in QUERIES:
        group = {'query_id': slug, 'query': text, 'methods': {}, 'jaccard_at_k': {}}
        for mode in MODES:
            q0 = time.perf_counter()
            vector, words = embed_query(model, text, mode, fmm, lookup, term_vectors, dim, args.batch_size)
            rows = rank_docs(doc_vectors[mode], valid_docs[mode], vector, docs, args.top)
            elapsed = (time.perf_counter()-q0)*1000
            group['methods'][mode] = {'tokens': words, 'top_k': rows,
                                      'elapsed_ms_including_query_embedding': round(elapsed, 4)}
            for row in rows:
                labels[(slug, row['doc_no'])] = row
        for a,b in [('raw','seg_keep'), ('seg_keep','seg_drop')]:
            group['jaccard_at_k'][f'{a}_vs_{b}'] = jaccard(group['methods'][a]['top_k'], group['methods'][b]['top_k'], args.top)
        reference = ROOT / 'output' /'4'/ 'tfidf_comparison' / f'{slug}_keep_{args.size}.json'
        if reference.exists():
            tfidf = json.loads(reference.read_text(encoding='utf-8'))
            group['tfidf_top_k'] = tfidf.get('top_k', [])[:args.top]
            group['jaccard_at_k']['raw_vs_tfidf'] = jaccard(group['methods']['raw']['top_k'], group['tfidf_top_k'], args.top)
            group['jaccard_at_k']['seg_keep_vs_tfidf'] = jaccard(group['methods']['seg_keep']['top_k'], group['tfidf_top_k'], args.top)
            for row in group['tfidf_top_k']:
                labels.setdefault((slug, row['doc_no']), {'doc_no': row['doc_no'], 'doc_id': row['doc_id'], 'title': row['title']})
        summary['queries'].append(group)
        print(f'查询: {text}; J(raw,seg_keep)={group["jaccard_at_k"]["raw_vs_seg_keep"]}; '
              f'J(seg_keep,seg_drop)={group["jaccard_at_k"]["seg_keep_vs_seg_drop"]}', flush=True)

    path = SEM / f'comparison_{args.size}_{key}.json'
    STAGE.save_json(path, summary)
    # 人工判断每篇文档是否与该查询真正相关，0=无关，1=相关。
    import csv
    csvpath = SEM / f'relevance_labels_{args.size}_{key}.csv'
    with csvpath.open('w', encoding='utf-8-sig', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['query_id','query','doc_no','doc_id','title','relevant_0_or_1'])
        query_for = dict(QUERIES)
        for (slug, number), row in sorted(labels.items()):
            writer.writerow([slug, query_for[slug], number, row.get('doc_id'), row.get('title',''), ''])
    print('结果:',path)
    print('人工相关性标注模板:',csvpath)

if __name__ == '__main__':
    main()
