"""阶段0：逐行校验 Qilin-IR-50K 的七个 JSONL 文件（数据体检，不属于评分任务）。

使用方法（在项目根目录执行）：
  python src/00_check_data.py

输入：data/ 下七个 JSONL 文件（含 optional/）
输出：output/data_check.json（行数/字段/JSON 错误/文档 ID 重复等检查结果）
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

EXPECTED_FILES = {
    "documents.jsonl": 50000,
    "queries.jsonl": 1941,
    "search_logs.jsonl": 18763,
    "user_history.jsonl": 1941,
    "users.jsonl": 9769,
    "optional/rating_pairs.jsonl": 172950,
    "optional/feedback_events.jsonl": 186486,
}
ID_FIELDS = ("doc_id", "note_id", "document_id", "id")


def inspect_one(relative_path: str, expected: int) -> dict:
    path = DATA_DIR / relative_path
    entry = {"file": relative_path, "expected": expected, "exists": path.is_file()}
    if not path.is_file():
        return entry

    count = 0
    fields = Counter()
    errors = []
    invalid_count = 0
    examples = []
    empty_title = empty_content = 0
    missing_id = 0
    ids = set()
    duplicate_ids = 0

    with path.open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("一行必须是一个 JSON 对象")
            except (json.JSONDecodeError, ValueError) as exc:
                invalid_count += 1
                if len(errors) < 5:
                    errors.append({"line": line_no, "error": str(exc)})
                continue
            count += 1
            fields.update(record.keys())
            if relative_path == "documents.jsonl":
                if len(examples) < 3:
                    examples.append({
                        "keys": list(record.keys()),
                        "title": str(record.get("title") or "")[:90],
                        "content_preview": str(record.get("content") or "")[:140],
                    })
                if not str(record.get("title") or "").strip():
                    empty_title += 1
                if not str(record.get("content") or "").strip():
                    empty_content += 1
                document_id = next((record[k] for k in ID_FIELDS
                                    if k in record and record[k] is not None), None)
                if document_id is None:
                    missing_id += 1
                elif str(document_id) in ids:
                    duplicate_ids += 1
                else:
                    ids.add(str(document_id))

    entry.update({
        "count": count,
        "matches_expected": count == expected,
        "fields": dict(fields.most_common()),
        "json_errors_count": invalid_count,
        "json_errors_first_5": errors,
    })
    if relative_path == "documents.jsonl":
        entry.update({
            "empty_title": empty_title,
            "empty_content": empty_content,
            "missing_id": missing_id,
            "duplicate_ids": duplicate_ids,
            "examples_first_3": examples,
        })
    return entry


def main() -> None:
    print("=== Qilin-IR-50K 数据检查 ===")
    results = []
    for filename, expected in EXPECTED_FILES.items():
        entry = inspect_one(filename, expected)
        results.append(entry)
        if not entry["exists"]:
            print(f"[缺失] {filename}：未找到（预期 {expected} 条）")
        else:
            print(f"[{'OK' if entry['matches_expected'] else '核对'}] "
                  f"{filename}：{entry['count']:,} / {expected:,} 条；"
                  f"解析错误 {entry['json_errors_count']} 条")

    docs = results[0]
    if docs["exists"]:
        print("\n=== documents.jsonl 重要字段及质量 ===")
        print("出现的字段:", ", ".join(docs["fields"]))
        for key in ("empty_title", "empty_content", "missing_id", "duplicate_ids"):
            print(f"{key}: {docs[key]}")
        for i, sample in enumerate(docs["examples_first_3"], 1):
            print(f"\n样本 {i}")
            print("字段:", sample["keys"])
            print("标题:", sample["title"])
            print("正文预览:", sample["content_preview"])
    dest = OUTPUT_DIR / "data_check.json"
    dest.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n详细检查结果已保存: {dest}")


if __name__ == "__main__":
    main()
