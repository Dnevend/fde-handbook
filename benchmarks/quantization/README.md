# benchmarks/quantization

第四部分的原始测量数据。全部在本机跑出，环境见 `env.txt`（M4 / 24 GB / llama.cpp 0.7.0 build 11669）。

## 文件与来源

| 文件 | 内容 | 由谁产生 |
|---|---|---|
| `tensor-types.csv` | 每种量化类型的张量数、参数量、真实位宽 | `labs/04-quantization/inspect_tensors.py` |
| `format-error.csv` | fp16/bf16/int8/int4 在同一批权重上的舍入误差 | 同上 |
| `group-size.csv` | 4/8 位、不同分组大小的归一化 RMSE | 同上 |
| `outliers.csv` | 逐输入通道 absmax 的分布 | 同上 |
| `kv-quant.csv` | KV 类型 × 上下文的困惑度、耗时、KV 体积 | `labs/04-quantization/kv_quant.py` |
| `ppl-domain.csv` | 同一模型换语料域的困惑度 | `labs/04-quantization/ppl_domain.py` |
| `probe-items.csv` | 60 道抽取题与标准答案 | `labs/04-quantization/task_probe.py` |
| `task-probe.csv` | 每轮逐题结果（四轮：三遍 f16 KV + 一遍 q4_0 KV） | 同上 |
| `determinism.txt` | 同一配置重复跑困惑度的输出 | 手工执行，命令见文件头 |

## 怎么重跑

```bash
source labs/env.sh                       # 定位 Qwen3.5-9B-Q4_K_M.gguf
uv run labs/04-quantization/inspect_tensors.py
bash benchmarks/quantization/build-calib.sh /tmp/fde-ppl-calib.txt
uv run labs/04-quantization/kv_quant.py --text /tmp/fde-ppl-calib.txt
uv run labs/04-quantization/ppl_domain.py
uv run labs/04-quantization/task_probe.py --passes 3 --label f16-f16
uv run labs/04-quantization/task_probe.py --passes 1 --label q4_0-q4_0 --ctk q4_0 --ctv q4_0
```

## 口径

- 权重误差的参照值是**反量化后的 Q4 权重**，不是训练时的 fp32；fp16/bf16 的结论不受影响，
  int4 那一列的绝对值偏大（4.1、4.2 节都写明了）。
- 困惑度只在同一段文本内可比；跨语料域的绝对值没有意义（4.6 节）。
- 探针题由合成工单生成，只测"按文档取字段"这一类能力，不代表业务问答质量。
- 语料本体不入库（见 `corpus/README.md`）；`build-calib.sh` 只记录拼装方式，跑的时候现生成。
