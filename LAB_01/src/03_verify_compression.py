"""阶段3B：解压四组索引并对照阶段3A原始位置索引逐词验证（压缩正确性检查）。

使用方法（在项目根目录执行）：
  python src/03_verify_compression.py --size 100 --full
  python src/03_verify_compression.py --size all --sample-terms 100
  python src/03_verify_compression.py --size all --mode both --full

输入：output/3/positional_{method}_keep_{size}.jsonl、output/3/compressed_{method}_keep_{size}/
输出：output/3/compression_verified_{method}_keep_{size}_{full|sample}.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import ExitStack
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage3b_codec import DictionaryDecoder, ENTRY, decode_postings

MODES = ('raw', 'front', 'vbyte', 'both')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--size', choices=('100', '1000', 'all'), default='100')
    p.add_argument('--method', choices=('fmm', 'hmm'), default='fmm')
    p.add_argument('--mode', choices=MODES + ('all',), default='all')
    p.add_argument('--sample-terms', type=int, default=100, help='默认抽样约100个词项；--full可全量验证')
    p.add_argument('--full', action='store_true', help='对所有词项与所有出现位置做严格逐项验证')
    args = p.parse_args()
    if args.sample_terms < 1:
        p.error('--sample-terms必须为正整数')
    return args


def main():
    args = parse_args()
    started = time.perf_counter()
    output = Path(__file__).resolve().parents[1] / 'output'/"3"
    source = output / f'positional_{args.method}_keep_{args.size}.jsonl'
    root = output / f'compressed_{args.method}_keep_{args.size}'
    if not source.is_file():
        raise FileNotFoundError(f'找不到原始索引：{source}')
    modes = MODES if args.mode == 'all' else (args.mode,)
    with ExitStack() as stack:
        input_file = stack.enter_context(source.open('r', encoding='utf-8'))
        readers = {}
        for mode in modes:
            base = root / mode
            manifest = json.loads((base / 'manifest.json').read_text(encoding='utf-8'))
            if manifest['mode'] != mode:
                raise AssertionError(f'{mode}清单与路径不匹配')
            readers[mode] = {
                'manifest': manifest,
                'dictionary': stack.enter_context((base / 'dict.bin').open('rb')),
                'decoder': DictionaryDecoder(manifest['dictionary_codec'], manifest['block_size']),
                'postings': stack.enter_context((base / 'posts.bin').open('rb')),
                'index': stack.enter_context((base / 'index.bin').open('rb')),
                'next_offset': 0,
                'checked_occurrences': 0,
            }
        term_count = next(iter(readers.values()))['manifest']['term_count']
        stride = max(1, term_count // args.sample_terms)
        visited = verified = checked_positions = checked_pairs = 0
        for line in input_file:
            if not line.strip():
                continue
            row = json.loads(line)
            ordinal = row['term_id']
            if ordinal != visited:
                raise AssertionError('原始索引term_id不连续')
            selected = args.full or ordinal % stride == 0 or ordinal == term_count - 1
            for mode, r in readers.items():
                term = r['decoder'].decode(r['dictionary'])
                meta = r['index'].read(ENTRY.size)
                if len(meta) != ENTRY.size:
                    raise EOFError(f'{mode}: 索引元数据不足')
                offset, length, df, skip_step = ENTRY.unpack(meta)
                if term != row['term']:
                    raise AssertionError(f'{mode}: 第{ordinal}词典不一致：{term!r} != {row["term"]!r}')
                if df != row['df'] or skip_step != row['skip_step']:
                    raise AssertionError(f'{mode}: 第{ordinal}项df/skip_step不一致')
                if offset != r['next_offset']:
                    raise AssertionError(f'{mode}: 第{ordinal}项offset不连续')
                if not selected:
                    r['next_offset'] += length
                    continue
                r['postings'].seek(offset)
                payload = r['postings'].read(length)
                if len(payload) != length:
                    raise EOFError(f'{mode}: 第{ordinal}项postings截断')
                decoded = decode_postings(payload, df, r['manifest']['postings_codec'] == 'gap_vbyte')
                if decoded != row['postings']:
                    raise AssertionError(f'{mode}: 第{ordinal}项解压后位置不一致')
                r['next_offset'] += length
            if selected:
                verified += 1
                checked_pairs += row['df']
                checked_positions += sum(len(p[1]) + len(p[2]) for p in row['postings'])
            visited += 1
            if visited % 20000 == 0:
                print(f'已扫描词项 {visited:,}, 验证 {verified:,}', flush=True)
        if visited != term_count:
            raise AssertionError(f'源词项数与元数据不同：{visited} / {term_count}')
        for mode, r in readers.items():
            if r['dictionary'].read(1):
                raise AssertionError(f'{mode}: 词典结尾存在额外字节')
            if r['index'].read(1):
                raise AssertionError(f'{mode}: 索引元数据结尾存在额外字节')
            if r['next_offset'] != (root / mode / 'posts.bin').stat().st_size:
                raise AssertionError(f'{mode}: postings文件长度不匹配')
        result = {
            'method': args.method,
            'size': args.size,
            'modes': list(modes),
            'verification': 'full' if args.full else 'sample',
            'source_terms': visited,
            'decoded_and_compared_terms': verified,
            'decoded_and_compared_posting_pairs': checked_pairs,
            'decoded_and_compared_occurrences': checked_positions,
            'passed': True,
            'elapsed_seconds_including_io': round(time.perf_counter() - started, 4),
        }
        stats_file = output / f'compression_verified_{args.method}_keep_{args.size}_{"full" if args.full else "sample"}.json'
        stats_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        print('\n=== 压缩解码验证通过 ===')
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print('输出文件:', stats_file)


if __name__ == '__main__':
    main()
