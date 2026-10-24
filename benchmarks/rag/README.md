# benchmarks/rag

第五部分的原始测量数据。全部在本机跑出，环境见 `env.txt`（M4 / 24 GB / llama.cpp 0.7.0 build 11669）。

## 文件与来源

| 文件 | 内容 | 由谁产生 |
|---|---|---|
| `parse.csv` | 文本层 PDF 的页数、字符数、表格数、单元格问题比例、与 pypdf 的一致率 | `labs/05-rag/parse_pdf.py` |
| `parse-ground-truth.json` | 两张人工转录的小表（单元格精确率的参照） | 手工 |
| `chunking.csv` | 三种切分的块数、长度、Recall@5/MRR/nDCG、丢表头行数 | `labs/05-rag/chunking.py` |
| `embedding.csv` | 1024/512/256 维 × f16/q8 的检索指标、体积、耗时 | `labs/05-rag/embed_index.py` |
| `embedding-neighbors.txt` | 三个块的最近邻抽查，确认向量不是噪声 | 同上 |
| `ann.csv` | 暴力 vs HNSW；真实 734 块与 2 万随机向量；efSearch 扫描 | `labs/05-rag/index_demo.py` |
| `retrieval.csv` | BM25 / 向量 / RRF：分题型与标识符子集的 Recall/MRR/nDCG | `labs/05-rag/retrieve.py` |
| `rerank.csv` | 重排前后的指标、延迟、top-1 变化数 | `labs/05-rag/rerank.py` |
| `generation.csv` | 带资料 / 闭卷两种模式的正确率、引用命中、拒答率 | `labs/05-rag/generate.py` |
| `generation.jsonl` | 逐题输出，供人工复核 | 同上 |
| `generation-audit.csv` | 60 道可答题的人工标签（correct / partial / wrong）与原因 | 人工逐条复核 |
| `generation-sample.csv` | 20 条人工抽检样本（`manual_label` 留空） | 同上 |
| `long-context.csv` | 三档上下文 × 三个位置的首字延迟与命中 | `labs/05-rag/long_context.py` |
| `metadata.csv` | 权限过滤的越权数/召回/延迟；增量更新与删除 | `labs/05-rag/metadata.py` |

## 怎么重跑

```bash
source labs/env.sh
cd labs && uv sync
labs/05-rag/download-models.sh
labs/05-rag/servers.sh start
uv run 05-rag/parse_pdf.py
uv run 05-rag/chunking.py
uv run 05-rag/embed_index.py
uv run 05-rag/index_demo.py
uv run 05-rag/retrieve.py
uv run 05-rag/rerank.py
uv run 05-rag/generate.py
uv run 05-rag/metadata.py
uv run 05-rag/long_context.py
```

参数在 `labs/05-rag/config.yaml`；改了就重跑，不要手改 CSV。

## 口径

- **延迟只在本机可比。** TTFT、p50/p95 跟机器强相关；跨机器只比相对差异。
- **质量实验用 734 块真实语料**（全量 794 块含 60 块合成件；合成件只进 5.9 的权限与增量实验）。这个规模下召回容易贴顶，5.4 的维度差异被天花板效应盖住；结论不外推到十万级。
- **随机向量只证明机制。** `ann.csv` 的 `random-20k` 不代表真实 embedding 分布；正文用来说明“不能外推”，不用来评价 HNSW。
- **q8 向量是模拟的。** `quantize_rows` 是逐行对称量化再反量化，不是 llama.cpp 的存储格式。
- **合成语料只做链路。** `synthetic: true` 的块不参与检索质量结论；权限与增量实验例外，那里只关心机制。
- **正确率是字面/数字匹配。** 5.8 没有用 LLM-as-judge；60 条可答题的人工标签在 `generation-audit.csv`，自动判定只当回归哨兵。
- **引用精确率按微平均报。** 相关引用数 ÷ 总引用数（66/182 = 0.363）；按题平均是 0.471。两者都只统计 60 道可答题。
- **无 OCR、无双栏。** 见 `corpus/GAPS.md` 与 `corpus/ACQUIRE.md`。
