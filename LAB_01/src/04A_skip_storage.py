"""阶段4A（任务4A-4）：统计不同跳表步长的指针存储开销。

任务4A(4)要求分析跳表步长对"存储性能/检索效率"的影响；检索效率由
04A_boolean_retrieval.py 的基准测试给出，本脚本补充存储侧数据：
按 02_build_index.py 的同一跳表边生成规则，统计每种固定步长在全量
词典上的跳跃边总数与字节数。

使用方法（在项目根目录执行）：
  python src/04A_skip_storage.py --size 100    # 100 篇快速自测
  python src/04A_skip_storage.py --size all    # 50,000 篇全量（核心数据）

输入：output/2/inverted_{method}_keep_{size}.jsonl（阶段2倒排表，逐词项含 df）
输出：output/4/skip_storage_{method}_{size}.json
  每种步长的跳跃边总数、按每边 8 字节（u32 目标文档号 + u32 倒排表内偏移）
  估算的指针存储开销，及其相对未压缩倒排记录（raw posts.bin）的比例。
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage3b_codec import ENTRY  # noqa: F401  仅用于说明元数据布局一致性

ROOT = Path(__file__).resolve().parents[1]
STEPS = ('2', '4', '8', '16', 'sqrt')
# 每条跳跃边的存储成本：目标文档编号(u32) + 该目标在倒排记录文件中的字节偏移(u32)。
BYTES_PER_EDGE = 8


def edges_of(df: int, fixed_step: int) -> int:
    """与 02_build_index.make_skip_edges 完全一致的跳跃边数量。"""
    if df < 4:
        return 0
    step = fixed_step if fixed_step else max(2, math.isqrt(df))
    # links = [[i, i+step] for i in range(0, df - step, step)]
    return len(range(0, df - step, step))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=('fmm', 'hmm'), default='fmm')
    parser.add_argument('--size', choices=('100', '1000', 'all'), default='all')
    args = parser.parse_args()

    output = ROOT / 'output'
    index_path = output / '2' / f'inverted_{args.method}_keep_{args.size}.jsonl'
    if not index_path.is_file():
        raise FileNotFoundError(f'缺少阶段2倒排表：{index_path}')

    started = time.perf_counter()
    dfs: list[int] = []
    with index_path.open('r', encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                dfs.append(json.loads(line)['df'])

    sqrt_edges = sum(edges_of(df, 0) for df in dfs)
    variants = {}
    for step in STEPS:
        if step == 'sqrt':
            edges, step_rule = sqrt_edges, 'floor(sqrt(df)), min 2'
        else:
            fixed = int(step)
            edges, step_rule = sum(edges_of(df, fixed) for df in dfs), f'fixed {fixed}'
        variants[step] = {
            'skip_step': step_rule,
            'skip_edges_total': edges,
            'bytes_per_edge': BYTES_PER_EDGE,
            'pointer_bytes': edges * BYTES_PER_EDGE,
            'pointer_mb_decimal': round(edges * BYTES_PER_EDGE / 1e6, 3),
        }

    # 相对规模参照：未压缩二进制倒排记录 posts.bin 与索引元数据 index.bin。
    raw_posts = output / '3' / f'compressed_{args.method}_keep_{args.size}' / 'raw' / 'posts.bin'
    posts_bytes = raw_posts.stat().st_size if raw_posts.is_file() else None
    summary = {
        'method': args.method, 'size': args.size,
        'term_count': len(dfs),
        'posting_pairs_total': sum(dfs),
        'accounting_rule': '每条跳跃边 = 目标文档号(u32) + 倒排记录内字节偏移(u32)，共 8 B；'
                            '实际提交的压缩索引未物化跳跃边（由 df 与步长推导），此处为等价估算',
        'edge_rule': '与 02_build_index.make_skip_edges 一致：df<4 不建边，'
                     '其余在 [0, df-step) 内每 step 一条',
        'reference_raw_posts_bytes': posts_bytes,
        'variants': variants,
        'elapsed_seconds': round(time.perf_counter() - started, 4),
    }
    if posts_bytes:
        for v in variants.values():
            v['pointer_vs_raw_posts_ratio'] = round(v['pointer_bytes'] / posts_bytes, 6)

    dest = output / '4' / f'skip_storage_{args.method}_{args.size}.json'
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print('统计文件:', dest)


if __name__ == '__main__':
    main()
