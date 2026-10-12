"""阶段4C（任务4C）：预训练语义模型，两条输入路线（原文 / FMM分词后逐词编码并聚合）。

使用方法（在项目根目录执行；三种 mode 必须都运行，共用同一 --model）：
  python src/04C_build_semantic.py --size 100 --mode raw        # 100 篇快速自测
  python src/04C_build_semantic.py --size all --mode raw        # 50,000 篇全量（核心数据）
  python src/04C_build_semantic.py --size all --mode seg_keep
  python src/04C_build_semantic.py --size all --mode seg_drop

输入：data/documents.jsonl、output/1/tokens_fmm_{size}.jsonl（seg 模式）
输出：output/4/semantic/embeddings_{mode}_{size}_{modelkey}.npy   文档向量
      output/4/semantic/term_vectors_keep_{size}_{modelkey}.npy  词项向量缓存（seg 模式）
      output/4/semantic/documents_{size}.jsonl                   文档元数据（标题+预览）
      output/4/semantic/stats_{mode}_{size}_{modelkey}.json      构建统计

依赖：pip install numpy jieba sentence-transformers
注：全部文档的逐词编码首次运行可能耗时很久；模型权重首次下载需要联网。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / 'data' / 'documents.jsonl'
OUTPUT = ROOT / 'output'
SEM = OUTPUT /"4"/ 'semantic'
IDS = ('doc_id', 'note_id', 'document_id', 'id')


def model_key(name: str) -> str:
    slug = re.sub(r'[^A-Za-z0-9_-]+', '_', name).strip('_')[-48:]
    return slug + '_' + hashlib.sha256(name.encode('utf-8')).hexdigest()[:8]


def get_id(rec: dict):
    for key in IDS:
        if key in rec and rec[key] is not None:
            return rec[key]
    raise ValueError('原始文档缺少 doc_id/note_id/document_id/id')


def iter_jsonl(path: Path):
    with path.open('r', encoding='utf-8-sig') as fh:
        for lineno, line in enumerate(fh, 1):
            if line.strip():
                rec = json.loads(line)
                if not isinstance(rec, dict):
                    raise ValueError(f'{path.name} 行 {lineno} 不是 JSON 对象')
                yield rec


def iter_aligned(size: str, seg: bool):
    limit = None if size == 'all' else int(size)
    tok_iter = iter_jsonl(OUTPUT /"1"/ f'tokens_fmm_{size}.jsonl') if seg else None
    count = 0
    for doc in iter_jsonl(DOCS):
        if limit is not None and count >= limit:
            break
        tokens = next(tok_iter) if tok_iter is not None else None
        did = get_id(doc)
        if tokens is not None and str(tokens.get('doc_id')) != str(did):
            raise ValueError(f'文档顺序或ID不一致，第 {count} 篇，documents={did} tokens={tokens.get("doc_id")}')
        yield doc, tokens
        count += 1


def save_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def get_model(name: str, device: str, max_seq_length: int):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError('先安装依赖：python -m pip install sentence-transformers') from exc
    model = SentenceTransformer(name, **({} if device == 'auto' else {'device': device}))
    native_length = model.max_seq_length
    if native_length is not None:
        model.max_seq_length = min(int(native_length), max_seq_length)
    else:
        model.max_seq_length = max_seq_length
    return model


def unit(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 0 else vec


def encode(model, texts: list[str], batch: int) -> np.ndarray:
    if not texts:
        return np.empty((0, model.get_sentence_embedding_dimension()), dtype=np.float32)
    vectors = model.encode(texts, batch_size=batch, show_progress_bar=False,
                           convert_to_numpy=True, normalize_embeddings=True)
    return np.asarray(vectors, dtype=np.float32).reshape(len(texts), -1)


def vocab_cache(size: str, model_name: str, model, batch: int, word_chunk: int):
    key = model_key(model_name)
    vocfile = SEM / f'vocab_keep_{size}_{key}.json'
    vecfile = SEM / f'term_vectors_keep_{size}_{key}.npy'
    metapath = SEM / f'term_vectors_keep_{size}_{key}_meta.json'
    if vocfile.exists() and vecfile.exists() and metapath.exists():
        meta = json.loads(metapath.read_text(encoding='utf-8'))
        if meta.get('max_seq_length') == model.max_seq_length:
            vocab = json.loads(vocfile.read_text(encoding='utf-8'))
            vectors = np.load(vecfile, mmap_mode='r')
            if len(vocab) == len(vectors):
                print('使用已有词项向量缓存:', vecfile, flush=True)
                return vocab, vectors

    tokens_path = OUTPUT /"1"/ f'tokens_fmm_{size}.jsonl'
    vocab_set = set()
    for rec in iter_jsonl(tokens_path):
        vocab_set.update(rec['title_keep'])
        vocab_set.update(rec['content_keep'])
    vocab = sorted(vocab_set)
    print(f'需要独立编码 {len(vocab):,} 个词项；一个词项只编码一次。', flush=True)
    dim = model.get_sentence_embedding_dimension()
    temp = vecfile.with_suffix('.building.npy')
    vectors = np.lib.format.open_memmap(temp, mode='w+', dtype=np.float32, shape=(len(vocab), dim))
    started = time.perf_counter()
    for start in range(0, len(vocab), word_chunk):
        end = min(len(vocab), start + word_chunk)
        vectors[start:end] = encode(model, vocab[start:end], batch)
        print(f'词项编码 {end:,}/{len(vocab):,}，用时 {time.perf_counter()-started:.1f}s', flush=True)
    vectors.flush()
    del vectors
    temp.replace(vecfile)
    save_json(vocfile, vocab)
    save_json(metapath, {'model': model_name, 'max_seq_length': model.max_seq_length,
                         'vocabulary_size': len(vocab), 'embedding_dim': dim,
                         'term_vectors_unit_normalized': True})
    return vocab, np.load(vecfile, mmap_mode='r')


def aggregate(tokens: list[str], lookup: dict[str, int], vectors: np.ndarray, dim: int) -> np.ndarray:
    """每个词项独立编码，使用词频加权平均，然后L2归一化。"""
    counts = Counter(tokens)
    if not counts:
        return np.zeros(dim, dtype=np.float32)
    ids = np.fromiter((lookup[word] for word in counts), dtype=np.int32, count=len(counts))
    weights = np.fromiter(counts.values(), dtype=np.float32, count=len(counts))
    mixed = np.sum(np.asarray(vectors[ids], dtype=np.float32) * weights[:, None], axis=0)
    return unit(mixed / weights.sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', choices=['100', '1000', 'all'], default='100')
    parser.add_argument('--mode', choices=['raw', 'seg_keep', 'seg_drop'], required=True)
    parser.add_argument('--model', default='BAAI/bge-small-zh-v1.5')
    parser.add_argument('--device', default='auto', help='auto/cpu/cuda（取决于PyTorch支持）')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--word-chunk', type=int, default=2048)
    parser.add_argument('--max-seq-length', type=int, default=512)
    parser.add_argument('--doc-chunk', type=int, default=256)
    args = parser.parse_args()
    if min(args.batch_size, args.word_chunk, args.max_seq_length, args.doc_chunk) <= 0:
        parser.error('批次和长度参数均必须为正整数')
    if not DOCS.exists():
        raise FileNotFoundError(f'缺少原始文档: {DOCS}')
    if args.mode != 'raw' and not (OUTPUT /"1"/ f'tokens_fmm_{args.size}.jsonl').exists():
        raise FileNotFoundError(f'缺少分词文件: tokens_fmm_{args.size}.jsonl')
    SEM.mkdir(parents=True, exist_ok=True)
    model = get_model(args.model, args.device, args.max_seq_length)
    key = model_key(args.model)
    dim = model.get_sentence_embedding_dimension()
    print(f'模型：{args.model}；维度：{dim}；模型最大长度：{model.max_seq_length}', flush=True)
    vocab = vectors = lookup = None
    if args.mode != 'raw':
        vocab, vectors = vocab_cache(args.size, args.model, model, args.batch_size, args.word_chunk)
        lookup = {word: i for i, word in enumerate(vocab)}

    n = 50000 if args.size == 'all' else int(args.size)
    filename = f'embeddings_{args.mode}_{args.size}_{key}.npy'
    dest = SEM / filename
    temp = dest.with_suffix('.building.npy')
    result = np.lib.format.open_memmap(temp, mode='w+', dtype=np.float32, shape=(n, dim))
    meta_path = SEM / f'documents_{args.size}.jsonl'
    started = time.perf_counter()
    raw_texts, raw_positions = [], []
    count = 0
    with meta_path.open('w', encoding='utf-8') as meta:
        for doc, tok in iter_aligned(args.size, args.mode != 'raw'):
            did = get_id(doc)
            title = str(doc.get('title') or '')
            content = str(doc.get('content') or '')
            meta.write(json.dumps({'doc_no': count, 'doc_id': did, 'title': title[:180],
                                   'content_preview': content[:220]}, ensure_ascii=False) + '\n')
            if args.mode == 'raw':
                raw_texts.append((title + '\n' + content).strip() or ' ')
                raw_positions.append(count)
                if len(raw_texts) >= args.doc_chunk:
                    result[raw_positions] = encode(model, raw_texts, args.batch_size)
                    raw_texts.clear()
                    raw_positions.clear()
            else:
                variant = 'keep' if args.mode == 'seg_keep' else 'drop'
                doc_tokens = tok[f'title_{variant}'] + tok[f'content_{variant}']
                result[count] = aggregate(doc_tokens, lookup, vectors, dim)
            count += 1
            if count % 1000 == 0:
                print(f'文档编码 {count:,}/{n:,}；耗时 {time.perf_counter()-started:.1f}s', flush=True)
        if raw_texts:
            result[raw_positions] = encode(model, raw_texts, args.batch_size)
    if count != n:
        temp.unlink(missing_ok=True)
        raise ValueError(f'文档数不匹配：预期 {n}，实际 {count}；请检查文件是否完整')
    result.flush()
    del result
    temp.replace(dest)
    stats = {
        'model': args.model, 'model_key': key, 'mode': args.mode, 'size': args.size,
        'documents': count, 'embedding_dim': dim, 'max_seq_length': model.max_seq_length,
        'encoding': 'raw original title+content' if args.mode == 'raw' else
                    'independent term embeddings, frequency weighted average, L2 normalize',
        'tokenizer': 'model native tokenizer for all encodings',
        'term_vocab_size': None if vocab is None else len(vocab),
        'elapsed_seconds_including_io': round(time.perf_counter()-started, 3),
        'matrix_file_bytes': dest.stat().st_size,
        'vectors_file': str(dest),
        'raw_text_truncation_note': '原文超过模型 max_seq_length 时由模型 tokenizer 截断'
    }
    stats_path = SEM / f'stats_{args.mode}_{args.size}_{key}.json'
    save_json(stats_path, stats)
    print('=== 语义向量构建统计 ===')
    print(json.dumps(stats, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
