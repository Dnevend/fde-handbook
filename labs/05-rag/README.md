# labs/05-rag

第五部分（5.1–5.9）的实验脚本，在 M4 / 24GB 上可完整运行。

## 准备

```bash
# 1. 依赖（在 labs/ 下；只装进 labs/.venv/）
cd labs && uv sync

# 2. 模型（约 1.3 GB，首次需要联网）
labs/05-rag/download-models.sh

# 3. 语料：先按 corpus/ACQUIRE.md 把 excerpt 档要用的 PDF 放到 corpus/raw/
corpus/fetch.sh                    # full 档
uv run corpus/synthetic/generate.py --clean --validate

# 4. 起服务：嵌入 :8081、重排 :8082、生成 :8080
labs/05-rag/servers.sh start
```

`servers.sh` 的三个端点：

| 端点 | 模型 | 关键参数 |
|---|---|---|
| `:8081` | Qwen3-Embedding-0.6B Q8 | `--embeddings --pooling last`，1024 维 |
| `:8082` | bge-reranker-v2-m3 Q8 | `--reranking` |
| `:8080` | Qwen3.5-9B Q4_K_M | `--jinja`，默认 ctx 8192 |

`CTX=32768 labs/05-rag/servers.sh start llm` 用于 5.1 的长上下文实验。

## 脚本

| 脚本 | 章节 | 做什么 | 产物 |
|---|---|---|---|
| `raglib.py` | — | 语料加载、切分、向量、重排、生成、指标 | — |
| `parse_pdf.py` | 5.2 | 文本层 PDF 的正文与表格抽取 | `parse.csv`、`parse-tables.json` |
| `chunking.py` | 5.3 | 长度 / 结构 / 语义三种切分 | `chunking.csv` |
| `embed_index.py` | 5.4 | 维度与向量精度对检索的影响 | `embedding.csv`、`embedding-neighbors.txt` |
| `index_demo.py` | 5.5 | 暴力 vs HNSW；真实向量与 2 万随机向量 | `ann.csv` |
| `retrieve.py` | 5.6 | BM25 / 向量 / RRF，含标识符子集 | `retrieval.csv` |
| `rerank.py` | 5.7 | 交叉编码器重排 top-20 | `rerank.csv` |
| `generate.py` | 5.8 | 带出处生成、拒答、闭卷对照 | `generation.csv`、`generation.jsonl`、`generation-sample.csv` |
| `long_context.py` | 5.1 | 长上下文灌入 vs RAG | `long-context.csv` |
| `metadata.py` | 5.9 | 权限过滤与增量更新 | `metadata.csv` |

所有参数在 `config.yaml`：块长/重叠、BM25 的 k1/b、RRF 的 k、top-k、HNSW 的 M/ef/seed、
向量维度与精度、生成参数与拒答标记、权限模型。改参数会让 `benchmarks/rag/` 的数字失效，重跑前记下改动。

## 跑的顺序

```bash
cd labs
uv run 05-rag/parse_pdf.py
uv run 05-rag/chunking.py          # 内部会向量化三种切分
uv run 05-rag/embed_index.py       # 生成 .cache/vectors-1024.npy
uv run 05-rag/index_demo.py
uv run 05-rag/retrieve.py          # 生成 .cache/hybrid-top20.json
uv run 05-rag/rerank.py            # 生成 .cache/reranked.json
uv run 05-rag/generate.py          # 依赖前两步的候选
uv run 05-rag/metadata.py
uv run 05-rag/long_context.py      # 会另起 :8083 的 32k 上下文服务
```

一轮全跑约 40–60 分钟，主要耗在生成（80 题 × 2 种模式）与长上下文预填充。

## 口径与限制

- **延迟只在本机可比。** 同一命令在不同机器上差几倍很正常；正文对比的是相对差异。
- **q8 向量是模拟的。** `quantize_rows` 做逐行对称 8 位量化再反量化，不是 llama.cpp 的存储格式；
  它回答“按 int8 存会不会掉点”，不回答“显存省多少字节”以外的工程细节。
- **合成语料只做链路。** `synthetic: true` 的题与块不参与质量结论，只用于权限、增量、演示。
- **没有 OCR / 双栏。** 见 `corpus/ACQUIRE.md`；5.2 只测文本层 PDF。
- **没有 LLM-as-judge。** 5.8 的正确率是字面/数字匹配，人工抽检 20 条随数据一起提交。
- **PyMuPDF 是 AGPL。** 只在实验脚本里调用，不进入站点构建；正文与数据仍按 CC BY-NC-SA 4.0。
