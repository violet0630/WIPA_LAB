"""阶段4C（任务4C）：在 04C_compare_semantic.py 导出的 relevance_labels CSV 填写 0/1 后计算 Precision@K。

使用方法（在项目根目录执行，先完成 CSV 中 relevant_0_or_1 列的人工标注）：
  python src/04C_evaluate_semantic.py --size all

输入：output/4/semantic/comparison_{size}_{modelkey}.json、relevance_labels_{size}_{modelkey}.csv
输出：output/4/semantic/precision_{size}_{modelkey}.json（各查询×各方法的 P@K）
"""
import argparse
import csv
import json
from pathlib import Path
import importlib.util

ROOT = Path(__file__).resolve().parents[1]
SEM = ROOT / 'output' /"4"/ 'semantic'
_stage_spec = importlib.util.spec_from_file_location('stage4c_builder', ROOT / 'src' / '04C_build_semantic.py')
STAGE = importlib.util.module_from_spec(_stage_spec)
_stage_spec.loader.exec_module(STAGE)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', choices=['100','1000','all'], default='100')
    parser.add_argument('--model', default='BAAI/bge-small-zh-v1.5')
    args = parser.parse_args()
    key = STAGE.model_key(args.model)
    comparison = json.loads((SEM / f'comparison_{args.size}_{key}.json').read_text(encoding='utf-8'))
    labels_file = SEM / f'relevance_labels_{args.size}_{key}.csv'
    labels = {}
    with labels_file.open('r', encoding='utf-8-sig', newline='') as handle:
        for item in csv.DictReader(handle):
            val = item['relevant_0_or_1'].strip()
            if val not in ('0','1'):
                raise ValueError(f"请先逐项人工标注 0 或 1：query={item['query_id']} doc={item['doc_no']}")
            labels[(item['query_id'], int(item['doc_no']))] = int(val)
    k = comparison['top_k']
    result = []
    for item in comparison['queries']:
        slug = item['query_id']
        methods = {name: r['top_k'] for name,r in item['methods'].items()}
        if 'tfidf_top_k' in item:
            methods['tfidf'] = item['tfidf_top_k']
        for method, rows in methods.items():
            missing = [int(row['doc_no']) for row in rows if (slug, int(row['doc_no'])) not in labels]
            if missing:
                raise KeyError(f'{labels_file.name} 缺少 {slug} 的标注行 doc_no={missing}；'
                               f'comparison JSON 与标注 CSV 不一致，请重新生成两者并补齐标注')
            hits = sum(labels[(slug, int(row['doc_no']))] for row in rows)
            result.append({'query': item['query'], 'method': method, 'retrieved':len(rows),
                           'relevant_in_top':hits, 'precision_at_k':round(hits/k, 4)})
    output = SEM / f'precision_{args.size}_{key}.json'
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2),encoding='utf-8')
    for r in result:
        print(r['query'],r['method'],f'P@{k}={r["precision_at_k"]:.4f}')
    print('保存:',output)

if __name__ == '__main__':
    main()
