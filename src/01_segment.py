"""阶段1（任务1）：分词与规范化——正向最大匹配 (FMM) vs 隐马尔可夫模型 (HMM)。

使用方法（在项目根目录执行）：
  python src/01_segment.py --limit 100    # 100 篇快速自测
  python src/01_segment.py --limit 1000   # 1000 篇验证
  python src/01_segment.py --limit 0      # 50,000 篇全量（核心数据）

输入：data/documents.jsonl、data/stopwords.txt
输出：output/1/tokens_{fmm|hmm}_{100|1000|all}.jsonl   逐篇分词结果（keep/drop 两套）
      output/1/segmentation_stats_{size}.json           分词统计（耗时/词表/停用词比例）
      output/1/segmentation_samples_{size}.json         前 5 篇的分词对照样例

标题与正文分别处理；规范化含 NFKC 全半角统一、英文小写、数字占位、
标点剔除，并保留 keep（保留停用词）与 drop（去停用词）两套词项序列。
"""
from __future__ import annotations

import argparse
import json
import re
import time
import unicodedata
from contextlib import ExitStack
from pathlib import Path

import jieba
from jieba import finalseg

ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS = ROOT / "data" / "documents.jsonl"
STOPWORD_FILE = ROOT / "data" / "stopwords.txt"
OUTPUT_DIR = ROOT / "output"/"1"
ID_FIELDS = ("doc_id", "note_id", "document_id", "id")

# 将汉字、连续英文字母、连续数字（含小数）、其他非空白符号分别作为片段。
FRAGMENT_RE = re.compile(r"[\u4e00-\u9fff]+|[A-Za-z]+|\d+(?:\.\d+)?|[^\s]", re.UNICODE)
HAN_RE = re.compile(r"[\u4e00-\u9fff]+\Z")
VALID_RE = re.compile(r"[\u4e00-\u9fffA-Za-z0-9_]+\Z")
NUMBER_RE = re.compile(r"\d+(?:\.\d+)?\Z")


def normalize_token(token: str) -> str | None:
    """NFKC 全半角统一、英文小写、数字占位符、标点/特殊符号剔除。"""
    token = unicodedata.normalize("NFKC", token).strip().lower()
    if not token:
        return None
    if NUMBER_RE.fullmatch(token):
        return "<NUM>"
    if VALID_RE.fullmatch(token):
        return token
    return None


def load_stopwords() -> set[str]:
    if not STOPWORD_FILE.is_file():
        raise FileNotFoundError(f"缺少停用词表：{STOPWORD_FILE}")
    stopwords = set()
    for word in STOPWORD_FILE.read_text(encoding="utf-8-sig").splitlines():
        word = word.strip()
        if not word or word.startswith("#"):
            continue
        normalized = normalize_token(word)
        if normalized is not None:
            stopwords.add(normalized)
    return stopwords


def fmm_cut(text: str, max_word_len: int, dictionary: dict) -> list[str]:
    """正向最大匹配：从左至右优先匹配词典中最长的词，最短回退到单字。"""
    result = []
    for fragment in FRAGMENT_RE.findall(text):
        if not HAN_RE.fullmatch(fragment):
            result.append(fragment)
            continue
        start = 0
        while start < len(fragment):
            upper = min(max_word_len, len(fragment) - start)
            chosen = fragment[start : start + 1]
            for width in range(upper, 1, -1):
                candidate = fragment[start : start + width]
                if dictionary.get(candidate, 0) > 0:
                    chosen = candidate
                    break
            result.append(chosen)
            start += len(chosen)
    return result


def hmm_cut(text: str) -> list[str]:
    """调用 jieba.finalseg 的 HMM/Viterbi 统计分词器，不调用 jieba.cut 混合模式。"""
    result = []
    for fragment in FRAGMENT_RE.findall(text):
        if HAN_RE.fullmatch(fragment):
            result.extend(finalseg.cut(fragment))
        else:
            result.append(fragment)
    return result


def normalized_versions(raw: list[str], stopwords: set[str]) -> tuple[list[str], list[str]]:
    keep = []
    for token in raw:
        normalized = normalize_token(token)
        if normalized is not None:
            keep.append(normalized)
    drop = [word for word in keep if word not in stopwords]
    return keep, drop


def doc_id_of(doc: dict):
    for name in ID_FIELDS:
        if name in doc and doc[name] is not None:
            return doc[name]
    raise KeyError(f"找不到文档ID，实际字段：{list(doc.keys())}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=100,
                        help="处理前N篇；0 表示全量（默认100）")
    parser.add_argument("--max-word-len", type=int, default=6,
                        help="FMM 最大汉字词长（默认6）")
    args = parser.parse_args()
    if args.limit < 0 or args.max_word_len < 1:
        parser.error("--limit 必须 >= 0，--max-word-len 必须 >= 1")
    return args


def main() -> None:
    args = parse_args()
    if not DOCUMENTS.is_file():
        raise FileNotFoundError(f"请先把 documents.jsonl 放到 {DOCUMENTS}")

    stopwords = load_stopwords()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("正在加载 jieba 词典……")
    jieba.initialize()
    # FREQ 是 jieba 内置词典的词频映射；只允许词频 >0 的候选词。
    dictionary = jieba.dt.FREQ
    if not dictionary:
        raise RuntimeError("jieba 词典加载失败")

    methods = {
        "fmm": lambda text: fmm_cut(text, args.max_word_len, dictionary),
        "hmm": hmm_cut,
    }
    label = "all" if args.limit == 0 else str(args.limit)
    stats = {name: {
        "segmentation_seconds": 0.0,
        "tokens_keep": 0,
        "tokens_drop": 0,
        "vocabulary_keep": set(),
        "vocabulary_drop": set(),
    } for name in methods}
    samples = []
    processed = 0

    with ExitStack() as stack:
        outputs = {
            name: stack.enter_context(
                (OUTPUT_DIR / f"tokens_{name}_{label}.jsonl").open("w", encoding="utf-8")
            ) for name in methods
        }
        source = stack.enter_context(DOCUMENTS.open("r", encoding="utf-8-sig"))
        for line_num, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                doc = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"documents.jsonl 第 {line_num} 行格式错误: {exc}") from exc
            if not isinstance(doc, dict):
                raise ValueError(f"documents.jsonl 第 {line_num} 行不是 JSON 对象")
            doc_id = doc_id_of(doc)
            title = unicodedata.normalize("NFKC", str(doc.get("title") or ""))
            content = unicodedata.normalize("NFKC", str(doc.get("content") or ""))
            sample = {"doc_id": doc_id, "title": title[:150]} if len(samples) < 5 else None

            for name, cut in methods.items():
                t0 = time.perf_counter()
                title_raw = cut(title)
                content_raw = cut(content)
                title_keep, title_drop = normalized_versions(title_raw, stopwords)
                content_keep, content_drop = normalized_versions(content_raw, stopwords)
                stats[name]["segmentation_seconds"] += time.perf_counter() - t0

                record = {
                    "doc_id": doc_id,
                    "title_keep": title_keep,
                    "content_keep": content_keep,
                    "title_drop": title_drop,
                    "content_drop": content_drop,
                }
                outputs[name].write(json.dumps(record, ensure_ascii=False) + "\n")
                keep = title_keep + content_keep
                drop = title_drop + content_drop
                stats[name]["tokens_keep"] += len(keep)
                stats[name]["tokens_drop"] += len(drop)
                stats[name]["vocabulary_keep"].update(keep)
                stats[name]["vocabulary_drop"].update(drop)
                if sample is not None:
                    sample[name] = {
                        "title_raw": title_raw,
                        "title_keep": title_keep,
                        "title_drop": title_drop,
                        "content_raw_first_30": content_raw[:30],
                    }

            if sample is not None:
                samples.append(sample)
            processed += 1
            if processed % 1000 == 0:
                print(f"已处理 {processed:,} 篇")
            if args.limit and processed >= args.limit:
                break

    output_stats = {
        "documents": processed,
        "limit": args.limit,
        "fmm_max_word_len": args.max_word_len,
        "stopword_count": len(stopwords),
        "methods": {},
    }
    for name, value in stats.items():
        output_stats["methods"][name] = {
            "segmentation_seconds": round(value["segmentation_seconds"], 4),
            "tokens_keep": value["tokens_keep"],
            "tokens_drop": value["tokens_drop"],
            "vocabulary_keep_size": len(value["vocabulary_keep"]),
            "vocabulary_drop_size": len(value["vocabulary_drop"]),
            "stopword_removed_ratio": round(
                1 - value["tokens_drop"] / max(1, value["tokens_keep"]), 4
            ),
        }
    stats_path = OUTPUT_DIR / f"segmentation_stats_{label}.json"
    sample_path = OUTPUT_DIR / f"segmentation_samples_{label}.json"
    stats_path.write_text(json.dumps(output_stats, ensure_ascii=False, indent=2), encoding="utf-8")
    sample_path.write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== 分词统计 ===")
    print(json.dumps(output_stats, ensure_ascii=False, indent=2))
    print("\n已保存统计:", stats_path)
    print("已保存案例:", sample_path)
    print("数据文件:", ", ".join(f"tokens_{name}_{label}.jsonl" for name in methods))
    print("注意：后续做短语/位置检索时，不要把去停用词数组的下标直接当作原始词项位置。")


if __name__ == "__main__":
    main()
