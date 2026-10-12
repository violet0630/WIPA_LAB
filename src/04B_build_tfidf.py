"""阶段4B（任务4B）：利用阶段1 FMM 词项构建文档 TF-IDF 稀疏矩阵（两遍流式处理）。

使用方法（在项目根目录执行）：
  python src/04B_build_tfidf.py --size 100 --variant keep    # 100 篇快速自测
  python src/04B_build_tfidf.py --size all --variant keep    # 50,000 篇全量（核心数据）
  python src/04B_build_tfidf.py --size all --variant drop

输入：output/1/tokens_fmm_{size}.jsonl（阶段1分词结果）
输出：output/4/tfidf_fmm_{variant}_{size}.npz            文档 TF-IDF CSR 矩阵（L2 归一化）
      output/4/tfidf_fmm_{variant}_{size}_vocab.json     词汇表
      output/4/tfidf_fmm_{variant}_{size}_idf.npy        IDF 向量
      output/4/tfidf_fmm_{variant}_{size}_docids.jsonl   行号↔doc_id 映射
      output/4/tfidf_fmm_{variant}_{size}_stats.json     构建统计

依赖：numpy scipy；不用 scikit-learn，以便完整展示 TF-IDF 公式。
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'output'
FIELDS = ('title', 'content')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', choices=['100', '1000', 'all'], default='100')
    parser.add_argument('--variant', choices=['keep', 'drop'], default='keep')
    return parser.parse_args()


def iterate_tokens(path: Path, variant: str):
    with path.open('r', encoding='utf-8-sig') as fp:
        for line_num, line in enumerate(fp, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f'{path.name} 第{line_num}行不是对象')
            for field in FIELDS:
                name = f'{field}_{variant}'
                if name not in record or not isinstance(record[name], list):
                    raise ValueError(f'{path.name} 第{line_num}行缺少数组字段 {name}')
            tokens = record[f'title_{variant}'] + record[f'content_{variant}']
            yield record.get('doc_id'), tokens


def main() -> None:
    args = parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    source = OUTPUT /"1"/ f'tokens_fmm_{args.size}.jsonl'
    if not source.exists():
        raise FileNotFoundError(f'找不到 {source}。先确认阶段1的输出。')
    prefix = f'tfidf_fmm_{args.variant}_{args.size}'
    started = time.perf_counter()

    # 第一遍：统计 N、df、非零项数量；避免将几百万个词项同时放入内存。
    df = Counter()
    n_docs = 0
    nnz = 0
    n_tokens = 0
    for doc_id, words in iterate_tokens(source, args.variant):
        if doc_id is None:
            raise ValueError(f'第{n_docs+1}个文档缺少doc_id')
        if any(not isinstance(w, str) for w in words):
            raise ValueError('词项必须是字符串')
        unique = set(words)
        df.update(unique)
        n_docs += 1
        nnz += len(unique)
        n_tokens += len(words)
    if n_docs == 0:
        raise ValueError('输入文档为空')
    vocabulary = sorted(df)
    term_to_idx = {word: i for i, word in enumerate(vocabulary)}
    idf = np.asarray(
        [math.log((1.0 + n_docs) / (1.0 + df[word])) + 1.0 for word in vocabulary],
        dtype=np.float32,
    )
    print(f'第1遍完成：文档 {n_docs:,}，词汇 {len(vocabulary):,}，非零项 {nnz:,}。')

    # 第二遍：分配精确大小的CSR数组，tf采用原始词频，doc向量进行L2归一化。
    indices = np.empty(nnz, dtype=np.int32)
    data = np.empty(nnz, dtype=np.float32)
    indptr = np.empty(n_docs + 1, dtype=np.int64)
    indptr[0] = 0
    written = 0
    id_path = OUTPUT / "4"/f'{prefix}_docids.jsonl'
    with id_path.open('w', encoding='utf-8') as id_file:
        for doc_no, (doc_id, tokens) in enumerate(iterate_tokens(source, args.variant)):
            counts = Counter(tokens)
            # 列号排序保证CSR结构规则；排名无关列顺序。
            terms = sorted(((term_to_idx[word], freq) for word, freq in counts.items()))
            norm2 = 0.0
            for col, tf in terms:
                weight = float(tf) * float(idf[col])
                norm2 += weight * weight
            norm = math.sqrt(norm2) if norm2 else 1.0
            for col, tf in terms:
                indices[written] = col
                data[written] = float(tf) * float(idf[col]) / norm
                written += 1
            indptr[doc_no + 1] = written
            id_file.write(json.dumps({'doc_no': doc_no, 'doc_id': doc_id}, ensure_ascii=False) + '\n')
            if (doc_no + 1) % 10000 == 0:
                print(f'第2遍：已构建 {doc_no+1:,} 篇')
    if written != nnz:
        raise AssertionError(f'CSR非零项个数不一致: {written} != {nnz}')
    matrix = sparse.csr_matrix((data, indices, indptr), shape=(n_docs, len(vocabulary)), dtype=np.float32)
    matrix.sort_indices()
    matrix_path = OUTPUT /"4"/ f'{prefix}.npz'
    sparse.save_npz(matrix_path, matrix, compressed=True)
    vocab_path = OUTPUT /"4"/  f'{prefix}_vocab.json'
    vocab_path.write_text(json.dumps(vocabulary, ensure_ascii=False), encoding='utf-8')
    idf_path = OUTPUT /"4"/  f'{prefix}_idf.npy'
    np.save(idf_path, idf)
    result = {
        'size': args.size, 'variant': args.variant, 'method': 'FMM',
        'documents': n_docs, 'vocabulary_size': len(vocabulary),
        'input_token_occurrences': n_tokens, 'unique_term_document_pairs': nnz,
        'tf_definition': 'raw term frequency (count)',
        'idf_definition': 'log((1+N)/(1+df)) + 1',
        'normalization': 'L2 per document (cosine)',
        'fields': ['title', 'content'], 'field_weights': [1.0, 1.0],
        'dtype': 'float32',
        'elapsed_seconds_including_io': round(time.perf_counter() - started, 4),
        'matrix_file_bytes': matrix_path.stat().st_size,
        'vocab_file_bytes': vocab_path.stat().st_size,
        'idf_file_bytes': idf_path.stat().st_size,
        'docids_file_bytes': id_path.stat().st_size,
    }
    stats_path = OUTPUT /"4"/ f'{prefix}_stats.json'
    stats_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print('\n=== TF-IDF 构建统计 ===')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print('\n输出文件均在：', OUTPUT/"4")


if __name__ == '__main__':
    main()
