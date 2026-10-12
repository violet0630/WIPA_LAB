"""阶段3B（任务3-2）：同一种位置索引的 Baseline、Front Coding、VByte 和组合压缩对照。

使用方法（在项目根目录执行）：
  python src/03_compress_index.py --size 100    # 100 篇快速自测
  python src/03_compress_index.py --size all    # 50,000 篇全量（核心数据）

输入：output/3/positional_{method}_keep_{size}.jsonl（阶段3A位置索引）
输出：output/3/compressed_{method}_keep_{size}/{raw|front|vbyte|both}/
        dict.bin（词典）、posts.bin（倒排记录）、index.bin（元数据）、manifest.json
      output/3/compression_summary_{method}_keep_{size}.json  四种编码空间对照
"""
from __future__ import annotations

import argparse
import json
import math
import time
from contextlib import ExitStack
from pathlib import Path

# 以文件路径加载辅助模块，可让项目src保持简单独立。
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage3b_codec import DictionaryEncoder, ENTRY, encode_postings

MODES = ('raw', 'front', 'vbyte', 'both')


def flags(mode: str) -> tuple[str, bool]:
    return ('front' if mode in ('front', 'both') else 'plain', mode in ('vbyte', 'both'))


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', choices=('100', '1000', 'all'), default='100')
    parser.add_argument('--method', choices=('fmm', 'hmm'), default='fmm')
    parser.add_argument('--mode', choices=MODES + ('all',), default='all')
    parser.add_argument('--block-size', type=int, default=8, help='Front Coding块中词项数，默认8')
    args = parser.parse_args()
    if args.block_size < 2:
        parser.error('--block-size必须>=2')
    return args


def encode_variant(source: Path, directory: Path, mode: str, size: str,
                   method: str, block_size: int, source_baseline_bytes: int) -> dict:
    started = time.perf_counter()
    directory.mkdir(parents=True, exist_ok=True)
    lexicon_mode, use_vbyte = flags(mode)
    codec = DictionaryEncoder(lexicon_mode, block_size)
    tmpfiles = [directory / f'{name}.tmp' for name in ('dict.bin', 'posts.bin', 'index.bin')]
    final_files = [directory / name for name in ('dict.bin', 'posts.bin', 'index.bin')]
    manifest_path = directory / 'manifest.json'
    term_count = token_count = pair_count = 0
    try:
        with ExitStack() as stack:
            inp = stack.enter_context(source.open('r', encoding='utf-8'))
            dictionary = stack.enter_context(tmpfiles[0].open('wb'))
            postings_file = stack.enter_context(tmpfiles[1].open('wb'))
            index_file = stack.enter_context(tmpfiles[2].open('wb'))
            previous_term = None
            for line_number, line in enumerate(inp, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                term, postings = row['term'], row['postings']
                df = int(row['df'])
                if row['term_id'] != term_count:
                    raise ValueError(f'第{line_number}行term_id不连续')
                if previous_term is not None and term <= previous_term:
                    raise ValueError(f'第{line_number}行词典未严格升序')
                if df != len(postings):
                    raise ValueError(f'第{line_number}行df与postings数量不符')
                expected_skip = max(2, math.isqrt(df))
                if row['skip_step'] != expected_skip:
                    raise ValueError(f'第{line_number}行skip_step与阶段3A默认规则不符')
                dictionary.write(codec.encode(term))
                binary = encode_postings(postings, variable_byte=use_vbyte)
                offset = postings_file.tell()
                postings_file.write(binary)
                index_file.write(ENTRY.pack(offset, len(binary), df, row['skip_step']))
                pair_count += df
                token_count += sum(len(p[1]) + len(p[2]) for p in postings)
                previous_term = term
                term_count += 1
                if term_count % 20000 == 0:
                    print(f'  {mode}: 已编码 {term_count:,} 个词项', flush=True)
        for temp, final in zip(tmpfiles, final_files):
            temp.replace(final)
    except Exception:
        for temp in tmpfiles:
            temp.unlink(missing_ok=True)
        raise
    byte_sizes = {path.name: path.stat().st_size for path in final_files}
    binary_size = sum(byte_sizes.values())
    manifest = {
        'method': method, 'variant': 'keep', 'size': size, 'mode': mode,
        'dictionary_codec': lexicon_mode,
        'postings_codec': 'gap_vbyte' if use_vbyte else 'gap_uint32',
        'block_size': block_size,
        'doc_gap_definition': 'current_doc - previous_doc, initial previous_doc=-1',
        'position_gap_definition': 'current_pos - previous_pos, separately for title/content, initial previous_pos=-1',
        'metadata_entry': '<QQII: posts_byte_offset, posts_byte_length, df, skip_step',
        'term_count': term_count, 'posting_pairs': pair_count, 'position_occurrences': token_count,
        'binary_file_bytes': byte_sizes, 'binary_total_bytes': binary_size,
        'original_positional_jsonl_bytes': source_baseline_bytes,
        'elapsed_seconds_including_io': round(time.perf_counter() - started, 4),
        'external_docid_mapping': f'docids_{method}_{size}.jsonl (shared, excluded equally)',
        'skip_edge_storage': 'derive from df and skip_step; no extra skip edges are stored',
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    # Include manifest explicitly rather than implicitly hiding its overhead.
    manifest['manifest_bytes'] = manifest_path.stat().st_size
    manifest['total_on_disk_bytes'] = binary_size + manifest['manifest_bytes']
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    # Adding manifest_bytes / total_on_disk_bytes changed manifest size; update to convergence.
    for _ in range(3):
        manifest['manifest_bytes'] = manifest_path.stat().st_size
        manifest['total_on_disk_bytes'] = binary_size + manifest['manifest_bytes']
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'完成 {mode}: 词典 {byte_sizes["dict.bin"]:,} B; '
          f'倒排记录 {byte_sizes["posts.bin"]:,} B; '
          f'元数据 {byte_sizes["index.bin"]:,} B; '
          f'合计 {binary_size:,} B', flush=True)
    return manifest


def main():
    args = get_args()
    output = Path(__file__).resolve().parents[1] / 'output'/"3"
    source = output / f'positional_{args.method}_keep_{args.size}.jsonl'
    if not source.is_file():
        raise FileNotFoundError(f'找不到位置索引：{source}')
    root = output / f'compressed_{args.method}_keep_{args.size}'
    original_bytes = source.stat().st_size
    modes = MODES if args.mode == 'all' else (args.mode,)
    print(f'位置索引：{source} ({original_bytes:,} B)')
    summaries = {}
    for mode in modes:
        print(f'正在生成 {mode} ...', flush=True)
        summaries[mode] = encode_variant(source, root / mode, mode, args.size,
                                         args.method, args.block_size, original_bytes)

    if args.mode == 'all':
        reference = summaries['raw']['binary_total_bytes']
        table = {
            'original_jsonl_bytes_for_reference': original_bytes,
            'comparison_baseline': 'raw mode binary (NOT JSONL)',
            'baseline_binary_bytes': reference,
            'term_count': summaries['raw']['term_count'],
            'posting_pairs': summaries['raw']['posting_pairs'],
            'position_occurrences': summaries['raw']['position_occurrences'],
            'variants': {},
        }
        for mode, info in summaries.items():
            current = info['binary_total_bytes']
            table['variants'][mode] = {
                'dict_bytes': info['binary_file_bytes']['dict.bin'],
                'posts_bytes': info['binary_file_bytes']['posts.bin'],
                'index_bytes': info['binary_file_bytes']['index.bin'],
                'total_binary_bytes': current,
                'reduction_vs_raw': round(1 - current / reference, 6),
                'elapsed_seconds_including_io': info['elapsed_seconds_including_io'],
            }
        path = output / f'compression_summary_{args.method}_keep_{args.size}.json'
        path.write_text(json.dumps(table, ensure_ascii=False, indent=2), encoding='utf-8')
        print('\n=== 压缩对照统计 ===')
        print(json.dumps(table, ensure_ascii=False, indent=2))
        print('\n统计文件：', path)
    else:
        print('\n结果：', root / args.mode / 'manifest.json')


if __name__ == '__main__':
    main()
