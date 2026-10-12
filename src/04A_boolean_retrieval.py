"""阶段4A（任务4A）：布尔检索、查询顺序、单层跳表、位置短语与压缩查询对照。

使用方法（在项目根目录执行，依赖 src/stage3b_codec.py 与阶段3B生成的二进制索引）：
  python src/04A_boolean_retrieval.py --size 100 --phrase "收藏 登机牌"   # 单次短语检索
  python src/04A_boolean_retrieval.py --size all --expr "(山西 AND 壁画) OR (古建 AND 艺术)"
  python src/04A_boolean_retrieval.py --size all --bench                  # 全量基准测试（核心数据）

输入：output/3/compressed_{method}_keep_{size}/{raw|front|vbyte|both}/、output/2/docids_{method}_{size}.jsonl
输出（--bench 时）：output/4/boolean_benchmark_{method}_{size}.json
  含三组实验：(1) 8 条复杂布尔表达式 × 4 种索引编码 × 2 种处理顺序 × 跳表开关；
             (2) 3 条选择性不同的 AND 查询 × 6 种跳表步长；
             (3) 5 个短语的位置检索及其与普通 AND 的命中数对照。
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage3b_codec import DictionaryDecoder, ENTRY, decode_postings

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'output'
MODES = ('raw', 'front', 'vbyte', 'both')
# 8 条复杂布尔表达式：覆盖 NOT、嵌套括号、OR 合取等结构，词项均为 FMM 词表内词。
EXAMPLES = [
    '((登机牌 AND 习惯) OR 收藏) AND NOT 话题',
    '(平价 OR 护肤品) AND (学生 AND 党)',
    '(山西 AND 壁画) OR (古建 AND 艺术)',
    '(旅行 AND 攻略) AND NOT 广告',
    '(考研 AND (英语 OR 政治)) AND 经验',
    '(北京 AND 故宫) OR (西安 AND 兵马俑)',
    '(咖啡 AND (做法 OR 教程)) AND NOT 广告',
    'NOT (话题 OR 分享)',
    # 故意按“最坏顺序”书写：左结合先归并最大词项 分享(df=8826)×攻略(df=3259)，
    # 用于对照 df_optimized 重排(旅行×攻略×分享)带来的比较次数差异。
    '(分享 AND 攻略) AND 旅行',
]
# 跳表步长实验：3 条词项选择性（df 量级）不同的 AND 查询。
SKIP_QUERIES = [
    '(话题 AND 登机牌) AND 习惯',
    '(攻略 AND 旅行) AND 民宿',
    '(考研 AND 英语) AND 复习',
]
# 短语检索实验：5 个短语，均已在位置索引中确认存在相邻出现。
PHRASES = [
    '收藏 登机牌',
    '平价 护肤品',
    '北京 故宫',
    '减肥 运动',
    '旅行 攻略',
]


def norm(value: str) -> str:
    return unicodedata.normalize('NFKC', value).strip().lower()


@dataclass(frozen=True)
class Entry:
    offset: int
    length: int
    df: int
    skip_step: int


class BinaryIndex:
    """读入词典和固定大小元数据；按需 seek/read 单个词项的 postings。"""
    def __init__(self, base: Path, expect_docs: int):
        self.base = base
        self.manifest = json.loads((base / 'manifest.json').read_text(encoding='utf-8'))
        self.mode = self.manifest['mode']
        self.num_docs = expect_docs
        if self.manifest['variant'] != 'keep':
            raise ValueError('本阶段位置短语比较需要保留停用词的索引')
        start = time.perf_counter()
        decoder = DictionaryDecoder(self.manifest['dictionary_codec'], self.manifest['block_size'])
        self.lookup: dict[str, Entry] = {}
        with (base / 'dict.bin').open('rb') as dictionary, (base / 'index.bin').open('rb') as meta:
            for _ in range(self.manifest['term_count']):
                term = decoder.decode(dictionary)
                raw = meta.read(ENTRY.size)
                if len(raw) != ENTRY.size:
                    raise EOFError('索引元数据截断')
                off, size, df, step = ENTRY.unpack(raw)
                self.lookup[term] = Entry(off, size, df, step)
            if dictionary.read(1) or meta.read(1):
                raise ValueError('词典/元数据存在多余字节')
        self.init_seconds = time.perf_counter() - start
        self.file = (base / 'posts.bin').open('rb')
        self.variable_byte = self.manifest['postings_codec'] == 'gap_vbyte'

    def close(self):
        self.file.close()

    def get(self, term: str) -> list:
        entry = self.lookup.get(norm(term))
        if entry is None:
            return []
        self.file.seek(entry.offset)
        payload = self.file.read(entry.length)
        if len(payload) != entry.length:
            raise EOFError('倒排记录截断')
        return decode_postings(payload, entry.df, self.variable_byte)

    def df(self, term: str) -> int:
        entry = self.lookup.get(norm(term))
        return entry.df if entry is not None else 0


# AST：('TERM', word)、('NOT', node)、('AND', left, right)、('OR', left, right)
def parse(expr: str) -> tuple:
    tokens = re.findall(r'\(|\)|[^\s()]+', expr)
    if not tokens:
        raise ValueError('空查询')
    pos = 0

    def atom():
        nonlocal pos
        if pos == len(tokens):
            raise ValueError('查询意外结束')
        token = tokens[pos]
        pos += 1
        if token == '(':
            node = disjunction()
            if pos >= len(tokens) or tokens[pos] != ')':
                raise ValueError('缺少右括号')
            pos += 1
            return node
        if token == ')' or token.upper() in ('AND', 'OR', 'NOT'):
            raise ValueError(f'意外的操作符：{token!r}')
        return ('TERM', norm(token))

    def negation():
        nonlocal pos
        if pos < len(tokens) and tokens[pos].upper() == 'NOT':
            pos += 1
            return ('NOT', negation())
        return atom()

    def conjunction():
        nonlocal pos
        node = negation()
        while pos < len(tokens) and tokens[pos].upper() == 'AND':
            pos += 1
            node = ('AND', node, negation())
        return node

    def disjunction():
        nonlocal pos
        node = conjunction()
        while pos < len(tokens) and tokens[pos].upper() == 'OR':
            pos += 1
            node = ('OR', node, conjunction())
        return node

    result = disjunction()
    if pos != len(tokens):
        raise ValueError(f'查询未完全解析，余下：{tokens[pos:]}')
    return result


def estimate(node: tuple, idx: BinaryIndex) -> int:
    kind = node[0]
    if kind == 'TERM':
        return idx.df(node[1])
    if kind == 'NOT':
        return max(0, idx.num_docs - estimate(node[1], idx))
    a, b = estimate(node[1], idx), estimate(node[2], idx)
    return min(a, b) if kind == 'AND' else min(idx.num_docs, a + b)


def optimized(node: tuple, idx: BinaryIndex) -> tuple:
    if node[0] in ('TERM',):
        return node
    if node[0] == 'NOT':
        return ('NOT', optimized(node[1], idx))
    kind = node[0]
    # 仅对AND链重排；先执行估计结果最小的子表达式。
    if kind == 'AND':
        parts = []
        def flatten(x):
            if x[0] == 'AND':
                flatten(x[1]); flatten(x[2])
            else:
                parts.append(optimized(x, idx))
        flatten(node)
        parts.sort(key=lambda x: estimate(x, idx))
        result = parts[0]
        for x in parts[1:]:
            result = ('AND', result, x)
        return result
    return ('OR', optimized(node[1], idx), optimized(node[2], idx))


def unique_union(a: list[int], b: list[int], counts: dict) -> list[int]:
    out = []
    i = j = 0
    while i < len(a) and j < len(b):
        counts['comparisons'] += 1
        if a[i] < b[j]: out.append(a[i]); i += 1
        elif a[i] > b[j]: out.append(b[j]); j += 1
        else: out.append(a[i]); i += 1; j += 1
    out.extend(a[i:]); out.extend(b[j:])
    return out


def intersection(a: list[int], b: list[int], counts: dict, step_mode: str) -> list[int]:
    out = []
    i = j = 0
    if step_mode == 'none':
        sa = sb = 0
    elif step_mode == 'sqrt':
        sa = max(2, math.isqrt(len(a)))
        sb = max(2, math.isqrt(len(b)))
    else:
        sa = sb = int(step_mode)
    while i < len(a) and j < len(b):
        counts['comparisons'] += 1
        if a[i] == b[j]:
            out.append(a[i]); i += 1; j += 1
        elif a[i] < b[j]:
            # 跳表边只从位置0, step, 2*step...连接到下一个节点。
            while sa and i % sa == 0 and i + sa < len(a):
                counts['comparisons'] += 1
                if a[i + sa] <= b[j]:
                    i += sa
                    counts['jumps'] += 1
                else:
                    break
            if i < len(a) and a[i] < b[j]:
                i += 1
        else:
            while sb and j % sb == 0 and j + sb < len(b):
                counts['comparisons'] += 1
                if b[j + sb] <= a[i]:
                    j += sb
                    counts['jumps'] += 1
                else:
                    break
            if j < len(b) and b[j] < a[i]:
                j += 1
    return out


def complement(a: list[int], n: int) -> list[int]:
    # a严格升序。universe是内部文档编号[0..n-1]。
    output = []
    cursor = 0
    for x in a:
        output.extend(range(cursor, x))
        cursor = x + 1
    output.extend(range(cursor, n))
    return output


def run_boolean(expr: str, idx: BinaryIndex, reorder: bool, step_mode: str) -> tuple[list[int], dict]:
    node = parse(expr)
    if reorder:
        node = optimized(node, idx)
    counts = {'comparisons': 0, 'jumps': 0, 'decoded_terms': 0}
    cache: dict[str, list[int]] = {}

    def evaluate(x):
        kind = x[0]
        if kind == 'TERM':
            term = x[1]
            if term not in cache:
                cache[term] = [p[0] for p in idx.get(term)]
                counts['decoded_terms'] += 1
            return cache[term]
        if kind == 'NOT':
            return complement(evaluate(x[1]), idx.num_docs)
        left = evaluate(x[1])
        # AND提前终止，避免解码无关右侧分支。
        if kind == 'AND' and not left:
            return []
        right = evaluate(x[2])
        return intersection(left, right, counts, step_mode) if kind == 'AND' else unique_union(left, right, counts)

    return evaluate(node), counts


def run_phrase(words: list[str], idx: BinaryIndex) -> list[int]:
    """每个词项的相邻位置严格+1；标题和正文独立匹配。"""
    if not words:
        raise ValueError('短语不能为空')
    postings = [idx.get(term) for term in words]
    if any(not p for p in postings):
        return []
    # 先按doc求交，随后校验相同字段中的连续位置；不跨越标题/正文边界。
    common = set(record[0] for record in postings[0])
    for listing in postings[1:]:
        common.intersection_update(rec[0] for rec in listing)
    lookup = [{rec[0]: (rec[1], rec[2]) for rec in posting} for posting in postings]
    answer = []
    for doc in sorted(common):
        for field in (0, 1):
            starts = set(lookup[0][doc][field])
            for offset, d in enumerate(lookup[1:], 1):
                target = set(d[doc][field])
                starts = {p for p in starts if (p + offset) in target}
                if not starts:
                    break
            if starts:
                answer.append(doc)
                break
    return answer


def median_test(fn, repeat: int):
    timings = []
    output = None
    counters = None
    for _ in range(repeat):
        started = time.perf_counter_ns()
        value = fn()
        elapsed = (time.perf_counter_ns() - started) / 1e6
        if isinstance(value, tuple):
            result, stats = value
        else:
            result, stats = value, None
        if output is not None and output != result:
            raise AssertionError('重复实验结果不一致')
        output, counters = result, stats
        timings.append(elapsed)
    return round(statistics.median(timings), 4), output, counters


def load_doc_count(method: str, size: str) -> int:
    path = OUTPUT /"2/" f'docids_{method}_{size}.jsonl'
    if not path.is_file():
        raise FileNotFoundError(f'未找到阶段2的文档编号映射：{path}')
    with path.open('r', encoding='utf-8') as f:
        return sum(bool(line.strip()) for line in f)


def run_benchmark(indexes: dict, queries: list[str], skip_queries: list[str],
                  phrases: list[str], repeat: int, size: str, method: str):
    rows = []
    for expr in queries:
        for mode, idx in indexes.items():
            for order in ('original', 'df_optimized'):
                for step in ('none', 'sqrt'):
                    fn = lambda: run_boolean(expr, idx, reorder=order == 'df_optimized', step_mode=step)
                    elapsed, docs, counts = median_test(fn, repeat)
                    rows.append({'experiment': 'boolean', 'query': expr, 'mode': mode,
                                 'order': order, 'skip_step': step, 'median_ms': elapsed,
                                 'hits': len(docs), 'sample_docnums': docs[:10], **counts})
    # 跳表步长实验：对多条选择性不同的 AND 查询逐一比较 6 种步长。
    for skip_query in skip_queries:
        for step in ('none', '2', '4', '8', '16', 'sqrt'):
            idx = indexes['raw'] if 'raw' in indexes else next(iter(indexes.values()))
            elapsed, docs, counts = median_test(
                lambda: run_boolean(skip_query, idx, reorder=False, step_mode=step), repeat)
            rows.append({'experiment': 'skip_step', 'query': skip_query, 'mode': idx.mode,
                         'order': 'original', 'skip_step': step, 'median_ms': elapsed,
                         'hits': len(docs), 'sample_docnums': docs[:10], **counts})
    # 短语检索：记录 4 种索引编码下的耗时，并同时统计同词项普通 AND 的命中数作对照，
    # 以量化位置信息带来的过滤效果。
    for phrase in phrases:
        and_expr = ' AND '.join(phrase.split())
        for mode, idx in indexes.items():
            elapsed, docs, _ = median_test(lambda: run_phrase(phrase.split(), idx), repeat)
            _, and_docs, _ = median_test(
                lambda: run_boolean(and_expr, idx, reorder=False, step_mode='none'), 1)
            rows.append({'experiment': 'phrase', 'query': phrase, 'mode': mode,
                         'median_ms': elapsed, 'hits': len(docs), 'sample_docnums': docs[:10],
                         'and_hits': len(and_docs)})

    # 同一查询（不同模式/顺序/步长）应有相同结果集，不能只有命中数相同。
    for experiment in ('boolean', 'skip_step', 'phrase'):
        groups = {}
        for row in rows:
            if row['experiment'] != experiment:
                continue
            groups.setdefault(row['query'], []).append(row)
        for q, chunk in groups.items():
            # 严格的完整结果检查在独立的 correctness 子测试中完成（见下）。
            if len(set(row['hits'] for row in chunk)) != 1:
                raise AssertionError(f'同一查询命中数不一致：{q}')

    def strict_consistency():
        for q in queries:
            expected, _ = run_boolean(q, indexes['raw'], False, 'none')
            for idx in indexes.values():
                for order in (False, True):
                    for step in ('none', 'sqrt'):
                        result, _ = run_boolean(q, idx, order, step)
                        if result != expected:
                            raise AssertionError(f'布尔结果不一致：{q} / {idx.mode}/{order}/{step}')
        for sq in skip_queries:
            expected, _ = run_boolean(sq, indexes['raw'], False, 'none')
            for step in ('2', '4', '8', '16', 'sqrt'):
                result, _ = run_boolean(sq, indexes['raw'], False, step)
                if result != expected:
                    raise AssertionError(f'跳表结果不一致: {sq} / {step}')
        for phrase in phrases:
            expected = run_phrase(phrase.split(), indexes['raw'])
            for idx in indexes.values():
                if run_phrase(phrase.split(), idx) != expected:
                    raise AssertionError(f'短语结果不一致: {phrase} / {idx.mode}')
    if 'raw' not in indexes:
        raise ValueError('性能对照需raw模式作为基准')
    strict_consistency()
    summary = {'size': size, 'method': method, 'repeat': repeat,
               'time_unit': 'milliseconds, median of repeated warm-cache executions',
               'timing_scope': 'postings byte reading, decoding, query operations (dictionary lookup already initialized)',
               'dictionary_init_seconds': {mode: round(idx.init_seconds, 5) for mode, idx in indexes.items()},
               'full_result_consistency': True,
               'examples': queries, 'skip_queries': skip_queries, 'phrases': phrases, 'rows': rows}
    out = OUTPUT /"4"/ f'boolean_benchmark_{method}_{size}.json'
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\n完成：{len(rows)} 条性能记录；严格结果一致性验证通过')
    print('统计文件:', out)
    for row in rows:
        if row['experiment'] == 'boolean' and row['order'] == 'df_optimized' and row['skip_step'] == 'sqrt':
            print(f"[{row['mode']:5}] {row['query']} -> {row['hits']} 篇, 中位 {row['median_ms']} ms")
    print('请查看JSON中所有原始/优化、跳表、四种压缩模式的记录，正式报告中按表作图。')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', choices=('100','1000','all'), default='100')
    parser.add_argument('--method', choices=('fmm','hmm'), default='fmm')
    parser.add_argument('--mode', choices=MODES, default='raw')
    parser.add_argument('--expr', type=str, help='单个布尔表达式，用空格连接AND/OR/NOT')
    parser.add_argument('--phrase', type=str, help='短语检索，以空格分隔已规范化的词项')
    parser.add_argument('--bench', action='store_true', help='4组压缩方案、3组布尔表达式与跳表实验')
    parser.add_argument('--repeat', type=int, default=5, help='每个条件重复测量次数')
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error('--repeat必须为正整数')
    if not (args.expr or args.phrase or args.bench):
        parser.error('请指定--expr、--phrase或--bench')
    size = args.size
    num_docs = load_doc_count(args.method, size)
    base = OUTPUT /"3"/ f'compressed_{args.method}_keep_{size}'
    modes = MODES if args.bench else (args.mode,)
    indexes = {}
    try:
        for mode in modes:
            path = base / mode
            if not path.is_dir():
                raise FileNotFoundError(f'未找到{mode}索引：{path}，请先执行阶段3B压缩')
            indexes[mode] = BinaryIndex(path, num_docs)
            print(f'{mode}: 已载入词典 {len(indexes[mode].lookup):,} 词，文档 {num_docs:,} 篇')
        if args.bench:
            run_benchmark(indexes, EXAMPLES, SKIP_QUERIES, PHRASES, args.repeat, size, args.method)
        else:
            idx = indexes[args.mode]
            if args.expr:
                elapsed, docs, counts = median_test(
                    lambda: run_boolean(args.expr, idx, True, 'sqrt'), args.repeat)
                print('布尔查询:', args.expr, '\n命中:', len(docs),
                      '\n内部文档编号前20个:', docs[:20],
                      '\n耗时中位数(ms):', elapsed, '\n计数:', counts)
            if args.phrase:
                elapsed, docs, _ = median_test(lambda: run_phrase(args.phrase.split(), idx), args.repeat)
                print('短语查询:', args.phrase, '\n命中:', len(docs),
                      '\n内部文档编号前20个:', docs[:20], '\n耗时中位数(ms):', elapsed)
    finally:
        for idx in indexes.values():
            idx.close()


if __name__ == '__main__':
    main()
