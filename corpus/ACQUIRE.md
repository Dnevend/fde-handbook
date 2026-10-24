# 语料的人工获取清单

`corpus/fetch.sh` 只自动下载 `redistributable: full` 的条目。`excerpt` 与 `none` 档必须人工取，
因为「节选哪几节」「能不能再分发」要人判断。这份清单只写第五部分实验实际用到的部分。

语料本体一律不入库（`corpus/raw/` 已在 `.gitignore`），这里只记来源与校验值。

## 第五部分需要什么

| 用途 | 语料 id | 档位 | 获取方式 |
|---|---|---|---|
| 5.2 表格抽取 | `eia-injection-01` | excerpt | 从 `corpus/sources.yaml` 的 `url` 手动下载 PDF，放到 `corpus/raw/eia-injection-01/` |
| 5.2 正文完整性 | `gov-ledger-guide` | full | `corpus/fetch.sh --only gov-ledger-guide` |
| 5.3–5.8 检索与生成 | `reg-safety-law`、`reg-mem-order-13`、`gov-checklist-hs`、`gov-responsibility-zs`、`gov-ledger-guide` | full | `corpus/fetch.sh` |
| 5.5–5.9 链路演示 | `synthetic/`（60 份） | 合成 | `uv run corpus/synthetic/generate.py --clean --validate` |
| 可选对照 | `acc-fangsu-0518`、`ipo-hitech`、`pat-inj-01` 等 | excerpt | 按 `sources.yaml` 手动下载；缺了不影响正文结论，脚本会跳过 |

## 校验值

| 文件 | sha256 |
|---|---|
| `eia-injection-01/2578121.pdf` | `28d836e755d2182679b2c8ce08eecd8f90449852c558e7dcc8c4471dd384c7b5` |
| `gov-ledger-guide/北京市生产经营单位安全生产主体责任规定——工作指南(202305).pdf` | `72a0ca0fdc382c03d1b0f1cf7be8a06f0740d91fd7b7ce05c7583a477da481cc` |

## 5.2 用了哪几页

- `eia-injection-01` 第 16 页（公用工程消耗，3×3）与第 46 页（风险物质储存量，4×5）是人工转录的对照表，
  标准答案在 `benchmarks/rag/parse-ground-truth.json`；
- 同一份 PDF 的 52 张表用于统计「空单元格 / 单元格内换行 / 小数点错位」的比例。

缺 `eia-injection-01` 时，`parse_pdf.py` 会跳过它，只留 `gov-ledger-guide` 的统计，正文要相应收回结论。

## 没有覆盖的

- 扫描件 OCR：`acc-meizhu-0116` 已登记但未下载；本书没有 `chi_sim` 语言包，5.2 不做 OCR 实测。
- 双栏版面：现有语料全是单栏。5.2 只引用公开的版面分析基准说明问题，不声称在本书语料上测过。
