# 实验一数据说明

配套实验要求见上级目录《实验一 信息检索实践.docx》。必做任务使用前三个数据文件；其余文件支持选做。

## 文件

| 文件 | 数量 | 用途 |
|---|---:|---|
| documents.jsonl | 50,000 篇 | 文档编号 doc_id、标题 title、正文 content |
| queries.jsonl | 1,941 次请求 | 查询 query、请求 query_id、用户 user_id、原始 split 等 |
| search_logs.jsonl | 18,763 条曝光 | 请求展示的文档及 click、like、collect、comment、share |
| users.jsonl | 9,769 位用户 | 匿名用户特征，供选做探索 |
| user_history.jsonl | 1,941 条 | 每次请求对应的近期点击列表 history_doc_ids |
| optional/rating_pairs.jsonl | 172,950 对 | 聚合的用户—文档行为及参考 split |
| optional/feedback_events.jsonl | 186,486 条事件 | 过滤到本库的原始事件，供自行聚合或划分 |

## 格式与关联

全部文件为 UTF-8 JSONL，一行一个 JSON 对象；缺失数值为 null，行为标记为整数 0/1。ID 不连续，请用字典关联。

- doc_id 关联 documents，user_id 关联 users，query_id 关联 queries、search_logs 和 user_history。
- 同一句查询可以对应多次请求，不能只按查询文字合并反馈。search_logs 保留每次请求的完整曝光；result_index 是原列表序号（从 0 开始），position 是原始展示位置，两者不等同。
- user_history 保留原列表顺序和重复项，所有文档均在本库；未提供历史项的独立时间戳，不能把列表序号当作时间。
- users 的 gender、age、platform、fans_num、follows_num、location 沿用原值；空值或 unknown 不代表具体类别。dense_feat1—40 是上游匿名特征，含义和生成时点未核实，不建议作为默认预测特征。
- feedback_events 的 source/source_split/request_id/result_index 标识事件来源；original_result_count 与 retained_result_count 表示过滤前后数量。本表可能是不完整请求，且与 search_logs 有重叠，不能直接拼接计数。
- rating_pairs 按 (user_id, doc_id) 聚合各行为的最大值，表示是否曾发生。event_count、首末时间戳、source_partitions 均来自该对的全部事件，仅用于核验，不能作为预测该对评分的输入。时间字段保留上游值，page_time 未换算单位。

## 必要使用规则

搜索部分沿用原始标签：train 1,471 次、test 470 次，来自同一批 298 位用户；13 种查询文本跨两边重复，不是用户、查询或会话互斥划分。需要调参时使用训练部分，测试仅作最终比较。

推荐参考划分为 train 138,360 对、test 34,590 对（80%/20%）；同一用户—文档对不跨两边，测试用户和文档均在训练中出现过。它合并原始四个搜索/推荐分区后重新划分，是随机矩阵补全，不是未来预测，与搜索 split 独立。

**使用推荐参考划分时，只从 rating_pairs 的训练行构建行为特征与用户画像。不要混入完整搜索日志、历史或事件表，它们包含推荐测试对的信息。** 自行用事件表划分时，先划分再构造特征；同一对的相关事件应按评测目的统一处理。时间预测须另外定义时点并重建历史。测试标签及由它计算的评分不得作为输入；page_time 是行为发生后的反馈，也不能用于预测当次点击。

可自行定义评分，例如 max(click, 2*like, 3*collect)，两侧规则保持一致。该值是行为等级，不是真实用户打分。未发生行为的曝光对与完全无记录的对要区分；反馈稀疏且类别不均衡，应与简单均值基线比较。

点击只能作为相关性的行为代理，未点击或未曝光均不等于不相关。可以在固定 50K 文档库上报告已点击文档的 Recall@K、MRR@K，或对曝光列表重排序，但须说明候选范围，结合人工案例分析。

## 读取示例

在本目录运行，仅需 Python 标准库：

```python
import json

def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

docs = {r["doc_id"]: r for r in read_jsonl("documents.jsonl")}
users = {r["user_id"]: r for r in read_jsonl("users.jsonl")}
for r in read_jsonl("optional/rating_pairs.jsonl"):
    if r["split"] == "train":
        rating = max(r["click"], 2*r["like"], 3*r["collect"])
        print(r["user_id"], users[r["user_id"]]["age"],
              docs[r["doc_id"]]["title"], rating)
        break
```

## 来源与处理

来源：[THUIR/Qilin](https://huggingface.co/datasets/THUIR/Qilin)，论文：Chen et al., 2025, [Qilin: A Multimodal Information Retrieval Dataset with APP-level User Sessions](https://arxiv.org/abs/2503.00501)。本包只提供文本与行为，不含图片。

从有效搜索请求按用户抽样，保留曝光和历史笔记，再补充背景笔记形成 50K 库。以中文为主，2,876 篇正文少于 50 字符。正文未预分词、未去停用词、未截断；少量空/占位标题、重复文本和原生特殊字符保留。28 篇中的已识别联系方式或直接标识用方括号说明替换，其余文字及行为值保留。

本包偏向长文及重复搜索用户，搜索曝光点击比例为 28.24%（原始数据 20.92%）；不代表原始分布，不能与官方全量成绩直接比较。上游许可在数据卡与正文中分别标作 Apache-2.0 和 MIT，公开再分发前需向发布方确认。
