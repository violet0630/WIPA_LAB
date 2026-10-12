"""阶段4B（任务4B）：单条查询的 TF-IDF 检索——同规则 FMM 分词 + 余弦相似度 Top-K。

使用方法（在项目根目录执行，需先运行 04B_build_tfidf.py）：
  python src/04B_search_tfidf.py --size all --variant keep --query "平价 护肤品 学生党" --top 10

输入：output/4/tfidf_fmm_{variant}_{size}.npz 及同前缀 vocab/idf/docids 文件、data/documents.jsonl
输出：控制台打印 Top-K；并将结果写入 output/4/tfidf_result_{variant}_{size}.json（每次覆盖）
批量查询与 keep/drop 对照请用 04B_search_tfidf_implement.py。
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import jieba
import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'output'/"4"
STOPWORDS = ROOT / 'data' / 'stopwords.txt'
DOCS = ROOT / 'data' / 'documents.jsonl'
ID_FIELDS = ('doc_id', 'note_id', 'document_id', 'id')
FRAGMENT_RE = re.compile(r'[\u4e00-\u9fff]+|[A-Za-z]+|\d+(?:\.\d+)?|[^\s]', re.UNICODE)
HAN_RE = re.compile(r'[\u4e00-\u9fff]+\Z')
VALID_RE = re.compile(r'[\u4e00-\u9fffA-Za-z0-9_]+\Z')
NUMBER_RE = re.compile(r'\d+(?:\.\d+)?\Z')


def normalize_token(token: str) -> str | None:
    token = unicodedata.normalize('NFKC', token).strip().lower()
    if not token:
        return None
    if NUMBER_RE.fullmatch(token):
        return '<NUM>'
    return token if VALID_RE.fullmatch(token) else None


def query_fmm(text: str, max_word_len: int = 6) -> list[str]:
    """与阶段1一致：中文片段FMM，非中文片段单独规范化。"""
    jieba.initialize()
    dictionary = jieba.dt.FREQ
    text = unicodedata.normalize('NFKC', text)
    result = []
    for fragment in FRAGMENT_RE.findall(text):
        if HAN_RE.fullmatch(fragment):
            start = 0
            while start < len(fragment):
                best = fragment[start:start+1]
                for length in range(min(max_word_len, len(fragment)-start), 1, -1):
                    word = fragment[start:start+length]
                    if dictionary.get(word, 0) > 0:
                        best = word
                        break
                result.append(best)
                start += len(best)
        else:
            result.append(fragment)
    return [v for raw in result if (v := normalize_token(raw)) is not None]


def stopword_set():
    if not STOPWORDS.exists():
        raise FileNotFoundError(f'缺少 {STOPWORDS}')
    out = set()
    for word in STOPWORDS.read_text(encoding='utf-8-sig').splitlines():
        word = word.strip()
        if not word or word.startswith('#'):
            continue
        v = normalize_token(word)
        if v:
            out.add(v)
    return out


def get_doc_id(record: dict):
    return next((record[k] for k in ID_FIELDS if k in record and record[k] is not None), None)


def titles_by_id(wanted_ids: set[str]) -> dict[str, str]:
    """按实际doc_id检索原文标题，不依赖位置索引和原始行号一致。"""
    results = {}
    if not DOCS.exists() or not wanted_ids:
        return results
    with DOCS.open('r', encoding='utf-8-sig') as file:
        for line in file:
            if not line.strip():
                continue
            document = json.loads(line)
            did = get_doc_id(document)
            key = str(did)
            if key in wanted_ids:
                results[key] = str(document.get('title') or '')[:120]
                if len(results) == len(wanted_ids):
                    break
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', choices=['100', '1000', 'all'], default='100')
    parser.add_argument('--variant', choices=['keep', 'drop'], default='keep')
    parser.add_argument('--query', required=True, help='自然语言查询，例如：平价护肤品学生党')
    parser.add_argument('--top', type=int, default=10)
    args = parser.parse_args()
    if args.top < 1:
        parser.error('--top 必须为正整数')
    prefix = f'tfidf_fmm_{args.variant}_{args.size}'
    matrix_path = OUTPUT / f'{prefix}.npz'
    if not matrix_path.is_file():
        raise FileNotFoundError(f'缺少 {matrix_path}，请先运行04B_build_tfidf.py')
    matrix = sparse.load_npz(matrix_path).tocsr()
    vocabulary = json.loads((OUTPUT / f'{prefix}_vocab.json').read_text(encoding='utf-8'))
    idf = np.load(OUTPUT / f'{prefix}_idf.npy')
    lookup = {t:i for i,t in enumerate(vocabulary)}
    tokens_before_stop = query_fmm(args.query)
    tokens = [x for x in tokens_before_stop if x not in stopword_set()] if args.variant == 'drop' else tokens_before_stop
    counts = Counter(tokens)
    kept = [(lookup[t], float(freq)*float(idf[lookup[t]])) for t,freq in counts.items() if t in lookup]
    oov = [t for t in counts if t not in lookup]
    print('原始查询：', args.query)
    print('FMM规范化查询：', tokens_before_stop)
    print('实际使用词项：', tokens)
    print('未登录词项：', oov)
    if not kept:
        print('查询词项均未出现在此索引的词汇表中，无法产生非零余弦相似度。')
        return
    cols = np.asarray([col for col,_ in kept], dtype=np.int32)
    weights = np.asarray([val for _,val in kept], dtype=np.float64)
    weights /= np.linalg.norm(weights)
    scores = np.asarray(matrix[:, cols] @ weights).ravel()
    matched = np.flatnonzero(scores > 0)
    if not len(matched):
        print('没有任何具有正余弦相似度的文档。')
        return
    # 用内部编号升序打破相同分数，确保重复运行结果稳定。
    order = np.lexsort((matched, -scores[matched]))
    ranked = matched[order][:args.top]
    mapping = {}
    ranked_docnos = {int(d) for d in ranked}
    with (OUTPUT / f'{prefix}_docids.jsonl').open('r', encoding='utf-8') as inp:
        for line in inp:
            r = json.loads(line)
            if r['doc_no'] in ranked_docnos:
                mapping[r['doc_no']] = r['doc_id']
    titles = titles_by_id({str(mapping[int(d)]) for d in ranked})
    rows = []
    for rank, docno in enumerate(ranked, 1):
        docno = int(docno)
        docid = mapping[docno]
        rows.append({'rank':rank, 'doc_no':docno, 'doc_id':docid,
                     'cosine_similarity': round(float(scores[docno]), 6),
                     'title': titles.get(str(docid), '[未读取到原始标题]')})
    print(f'非零匹配文档数：{len(matched)}；Top-{len(rows)}：')
    for row in rows:
        print(f"{row['rank']:2d}. doc_id={row['doc_id']} score={row['cosine_similarity']:.6f}  {row['title']}")
    result_file = OUTPUT / f'tfidf_result_{args.variant}_{args.size}.json'
    result_file.write_text(json.dumps({
        'query': args.query, 'query_tokens': tokens, 'oov':oov,
        'size': args.size,'variant':args.variant,
        'positive_score_documents':len(matched), 'top_k': rows,
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    print('结果保存：', result_file)

if __name__ == '__main__':
    main()
