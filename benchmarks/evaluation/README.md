# benchmarks/evaluation

第六部分的测试集统计与拒答审计。原始题在 `eval/qa-v2.jsonl`；这里只放脚本产物，正文引用这两个 CSV。

## 文件

| 文件 | 内容 | 由谁产生 |
|---|---|---|
| `testset-split.csv` | 四类 × dev/test 的题数、题目长度、答案长度、平均出处数 | `labs/06-evaluation/build_eval.py` |
| `testset-coverage.csv` | 每份语料的可定位条款数、被覆盖条款数、引用次数、覆盖率 | 同上 |
| `refuse-audit.csv` | 75 道拒答题的 top-1/top-2 相似度、命中文档与条款、是否标记、人工复核结论 | `labs/06-evaluation/audit_refuse.py` |
| `refuse-audit-summary.csv` | 可答题与拒答题的相似度分布、标记数 | 同上 |
| `refuse-audit-v1.csv` / `-v1 summary` | v1 的 20 道拒答题审计；R12 的 0.693 是退役依据 | `audit_refuse.py --file eval/qa.jsonl --tag v1` |
| `retrieval-metrics.csv` | 策略 × 相关性口径 × 题型的 Recall@k / MRR@10 / nDCG@10 / joint Recall@5 | `labs/06-evaluation/retrieval_metrics.py` |
| `retrieval-k-curve.csv` | Recall@k 曲线（k=1..20）与 bootstrap 区间 | 同上 |
| `retrieval-paired.csv` | 策略两两比较的 delta、95% CI、McNemar p | 同上 |
| `retrieval-threshold.csv` | 相关性阈值 0.35/0.5/0.65 的敏感性 | 同上 |
| `retrieval-coverage.csv` | test 分片 113 道可答题里，金标在索引里的 105 道 | 同上 |
| `retrieval-excluded.csv` | 被排除的 8 道题与缺失条款 | 同上 |
| `judge-pointwise.csv` | 80 条回答的人工/裁判标签、打分、理由、耗时 | `labs/06-evaluation/judge_calibration.py` |
| `judge-confusion.csv` / `judge-summary.csv` | 三分类混淆矩阵；准确率、κ、F1、偏置翻转率 | 同上 |
| `judge-bias.csv` | 废话偏置与 A/B 顺序偏置 | 同上 |
| `faithfulness.csv` / `faithfulness-summary.csv` | 逐条引用支撑率、内容支撑率、裁判忠实判定 | `labs/06-evaluation/faithfulness_audit.py` |
| `faithfulness-cross.csv` | 忠实度 × 人工正确率的交叉表 | 同上 |
| `generation-v2.jsonl` / `generation-v2-judged.jsonl` | v2 test 的 60 题生成结果与裁判标签 | `labs/06-evaluation/generate_v2.py`、`judge_v2.py` |
| `judge-v2-summary.csv` | v2 上字符串判定与裁判的比率、一致率 | `judge_v2.py` |
| `string-baseline.csv` | 5.8 字符串判定在 60 道可答题上的准确率、κ、精确率、召回率 | `labs/06-evaluation/string_baseline.py` |
| `judge-score-mapping.csv` | 同一批 0-100 分在两种分档边界下的 κ | `labs/06-evaluation/score_mapping.py` |
| `ablation-retrieval.csv` | 六个变体的 Recall@5 / MRR / nDCG / joint Recall@5 | `labs/06-evaluation/ablation.py` |
| `ablation-paired.csv` | 各变体相对 RRF 的配对差值、95% CI、McNemar p | 同上 |
| `ablation-fixed-threshold.csv` | 固定切分在相关性阈值 0.35/0.5 下的指标敏感性 | 同上 |
| `error-attribution.csv` | 18 道错题的阶段归因：索引/检索/误拒/引用/生成 | `labs/06-evaluation/error_attribution.py` |

## 怎么重跑

```bash
cd labs && uv run 06-evaluation/build_eval.py --check    # 核对分片，不重写
cd labs && uv run 06-evaluation/audit_refuse.py          # 需要第五部分的块与向量缓存
cd labs && uv run 06-evaluation/audit_refuse.py --file ../eval/qa.jsonl --tag v1
cd labs && uv run 06-evaluation/retrieval_metrics.py          # 默认跑 test 分片
cd labs && uv run 06-evaluation/judge_calibration.py          # 80 条人工标签上的裁判校准
cd labs && uv run 06-evaluation/faithfulness_audit.py         # 引用支撑与裁判忠实度
cd labs && uv run 06-evaluation/generate_v2.py                # v2 test 的 60 题生成
cd labs && uv run 06-evaluation/judge_v2.py                   # 给 v2 生成结果打裁判标签
cd labs && uv run 06-evaluation/ablation.py                   # 消融：BM25/稠密/RRF/重排/切分/维度
cd labs && uv run 06-evaluation/error_attribution.py           # 18 道错题的五阶段归因
cd labs && uv run 06-evaluation/diagnose.py --set v2 --id X32  # 单题诊断卡
```

`build_eval.py` 的分片是幂等的：按 `id + question` 的哈希选 dev，重跑不会漂移；改了题目才会变。

## 口径

- **条款数按行首条号去重**：安全生产法 119、有限空间规定 22、指南 54，合计 195。指南是汇编，54 个条号里 27 个在正文重复出现（按位置算 94 处）；覆盖率按条号计，检索相关性不能只比条号。
  `corpus/index.json` 里的条目更多（目录、正文引用会重复计），覆盖率只按去重后的条号算；条号重复的段落以正文定位。
- **top-1 相似度**是问题向量与 734 个真实块的余弦最大值，模型为 Qwen3-Embedding-0.6B（Q8_0）。
  可答题的 5% 分位约 0.67，拒答题的中位数 0.44；两条分布有重叠，阈值只用来缩小人工复核范围，不自动判定。
- **拒答审计只查「检索得到」**，不查「模型会不会用」。后者要 6.3 的生成评估。
- **检索相关性按答案原文判**：一个块覆盖某一条答案片段的 4-gram 达到 0.5 才算相关；不按条款号判，因为索引里有漏块与条号错位（见 `retrieval-excluded.csv`）。
- **8 道题被排除**：test 的 113 道可答题里，4 条短条款的正文没进 734 个块（安全法第六条/第四十七条、有限空间规定第十条/第十一条），没有块能覆盖答案的一半。检索指标在 105 道上算。
- **裁判只在 v1 的 80 条上校准**：60 道可答 + 20 道拒答；v2 的 60 条生成结果没有人工标签，只能报「字符串 vs 裁判」的一致率，不能报正确率。
- **消融的基线是 RRF**：所有差值相对 RRF 算，配对 bootstrap 2000 次；切分变体会换一套块与相关性，不能和「只改一个参数」混为一谈。
- **忠实度是句级的**：回答按句切分，句子末尾的 `[n]` 对应第 n 段；句子与第 n 段的 4-gram 覆盖率达到 0.5 才算有支撑。这个口径对改写偏严（X16/X20 被误判），只用于描述引用质量，不用于判定模型对错。
- **v1 的 R12 已退役**：top-1 = 0.693，命中 `pat-inj-02` 的料筒温度，属可答题。保留记录是为了说明 v1 的拒答率口径有缺口。
