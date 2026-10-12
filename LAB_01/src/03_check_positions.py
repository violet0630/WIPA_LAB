"""阶段3A：逐词核对位置索引和阶段2 Posting List，查看标题/正文位置信息（正确性抽查）。

使用方法（在项目根目录执行）：
  python src/03_check_positions.py --size all
  python src/03_check_positions.py --size all --terms 登机牌 话题

输入：output/2/inverted_{method}_keep_{size}.jsonl、output/3/positional_{method}_keep_{size}.jsonl
输出：控制台输出全量词项 df/文档编号/位置的一致性校验结果
"""
import argparse
import json
from pathlib import Path

OUTPUT = Path(__file__).resolve().parents[1] /"output"



def rows(path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", choices=("fmm", "hmm"), default="fmm")
    p.add_argument("--size", choices=("100", "1000", "all"), default="100")
    p.add_argument("--terms", nargs="*", default=["登机牌", "话题"], help="选择查看位置的词项")
    args = p.parse_args()
    old_path = OUTPUT/"2" / f"inverted_{args.method}_keep_{args.size}.jsonl"
    pos_path = OUTPUT /"3"/ f"positional_{args.method}_keep_{args.size}.jsonl"
    wanted = set(args.terms)
    found = {}
    token_count = postings_count = checked = 0
    old_rows = rows(old_path)
    new_rows = rows(pos_path)
    from itertools import zip_longest
    for old, new in zip_longest(old_rows, new_rows):
        if old is None or new is None:
            raise AssertionError("两份词典行数不一致")
        if old["term"] != new["term"] or old["df"] != new["df"]:
            raise AssertionError(f"词项或 df 不一致：{old['term']} / {new['term']}")
        old_docs = old["postings"]
        new_postings = new["postings"]
        if old_docs != [p[0] for p in new_postings]:
            raise AssertionError(f"词项 {old['term']} 的 Posting List 与阶段2不一致")
        for doc_num, titles, contents in new_postings:
            if not titles and not contents:
                raise AssertionError(f"文档 {doc_num} 的位置列表均为空")
            for positions in (titles, contents):
                if any(not isinstance(x, int) or x < 0 for x in positions):
                    raise AssertionError("出现非法位置")
                if any(x >= y for x, y in zip(positions, positions[1:])):
                    raise AssertionError("位置不是严格升序")
            token_count += len(titles) + len(contents)
        postings_count += new["df"]
        checked += 1
        if new["term"] in wanted:
            found[new["term"]] = new

    print(f"验证成功：{checked:,} 词，{postings_count:,} 条不重复倒排记录，{token_count:,} 次词项位置")
    for term in args.terms:
        row = found.get(term)
        if row is None:
            print(f"词项 {term!r} 不存在，跳过")
            continue
        print(f"\n词项 {term!r}: df={row['df']}, 跳表步长={row['skip_step']}")
        for docnum, title, content in row["postings"][:3]:
            print(f"  文档编号={docnum}; 标题位置={title[:15]}; 正文位置={content[:15]}")
    print("说明：位置是字段内的规范化分词下标，不是原文字符偏移量。")


if __name__ == "__main__":
    main()
