# labs/07-gpu-inference

第七部分的云 GPU 实验脚本。**本地（Mac）不要求能跑**；Mac 上只做语法检查和 `--help`，CUDA 部分在 GPU 机器上跑。

租机、装环境、成本记录见正文附录 F。脚本里的密钥一律走环境变量，不入库；模型权重不入库。

## 先租什么

- 单卡：1×4090 24G（7.2–7.6、7.8、7.9）
- 双卡：2×4090 同机（7.7）
- 系统镜像用 Ubuntu 22.04 虚机，不要用容器镜像；7.2 要装 Docker 和 `nvidia-container-toolkit`
- 推荐平台与价格见附录 F；预算上限建议 ¥100

## 数据放哪

所有产物都在 `benchmarks/gpu/` 下，按主题分目录：

```text
benchmarks/gpu/
  env/         启动命令、环境快照
  hardware/    gpu_probe 的带宽/算力
  monitor/     nvidia-smi 时序、拓扑、详细状态
  bench/       并发压测汇总与逐请求 CSV
  quant/       量化质量对比
  prefix/      Prefix Caching / Chunked Prefill
  multi-gpu/   TP=2 vs 2 副本
  capacity/    容量规划校验
  oom/         OOM 与卡顿复现
```

每次测量至少记录：GPU 型号、驱动、CUDA、PyTorch、vLLM 版本、完整命令、重复次数。`gpulib.env_note()` 会把这些打成一段文本；脚本会自动写到对应目录。

## 实验矩阵

| 节 | 脚本 | 主要产物 |
|---|---|---|
| 7.1 硬件 | `gpu_probe.py` | 实测 HBM 带宽、FP16 算力、与规格的比值 |
| 7.2 软件栈 | `serve_vllm.sh` + 附录 F | 驱动/CUDA/Docker/toolkit 版本与容器 GPU 验证 |
| 7.3 监控 | `monitor_gpu.py` | 空载/prefill/decode/显存压力四段时序 |
| 7.4 vLLM | `serve_vllm.sh` + `bench_openai.py` | 并发曲线、TTFT/TPOT、吞吐、显存 |
| 7.5 量化 | `quant_eval.py` | BF16/FP8/AWQ 的抽取正确率、速度、显存 |
| 7.6 加速 | `serve_vllm.sh` + `bench_openai.py` | Prefix Caching、Chunked Prefill、投机解码 |
| 7.7 多卡 | `tp_vs_replica.sh` | TP=2 vs 2 副本的吞吐/延迟 |
| 7.8 OOM | `serve_vllm.sh` + `monitor_gpu.py` | 启动 OOM、运行时 OOM、卡顿现场 |
| 7.9 容量 | `capacity.py` + `bench_openai.py` | 预测 vs 实测、每百万 token 成本 |

## 快速开始

```bash
# 1. 先看脚本会起什么容器（Mac 上也能跑）
bash labs/07-gpu-inference/serve_vllm.sh --dry-run

# 2. GPU 机器上起服务（模型目录先按附录 F 下载）
bash labs/07-gpu-inference/serve_vllm.sh \
  --model-dir models/qwen3.5-9b-awq \
  --model-name qwen3.5-9b-awq \
  --max-model-len 8192 --gpu-mem-util 0.90

# 3. 压测：并发 1/2/4/8/16/32
uv run labs/07-gpu-inference/bench_openai.py \
  --base-url http://127.0.0.1:8000/v1 \
  --concurrency 1,2,4,8,16,32 --requests 64 \
  --prompt-chars 1200 --max-tokens 128 --label qwen-awq

# 4. 采样监控（另开一个终端）
uv run labs/07-gpu-inference/monitor_gpu.py \
  --duration 120 --interval 1 --label qwen-awq-decode

# 5. 容量估算
uv run labs/07-gpu-inference/capacity.py \
  --model qwen3.5-9b-awq --ctx 8192 --concurrency 50 --gpu-price 1.92

# 6. 双卡对照（2×4090 同机）
bash labs/07-gpu-inference/tp_vs_replica.sh \
  --model-dir models/qwen3.5-9b-awq --model-name qwen3.5-9b-awq \
  --concurrency 8 --requests 64
```

## 脚本说明

- `gpulib.py`：共用工具。`stream_once()` 定义唯一的 TTFT/TPOT 口径；`env_note()` 记录环境。
- `serve_vllm.sh`：起/停 vLLM 容器。模型只读挂载，端口只绑 `127.0.0.1`；启动命令写到 `benchmarks/gpu/env/`。
- `bench_openai.py`：并发压测。支持多端点（2 副本轮询）、唯一前缀开关（测/不测 Prefix Caching）、逐请求 CSV。
- `monitor_gpu.py`：按间隔采 `nvidia-smi`，首尾记录 `topo -m` 和详细状态。
- `gpu_probe.py`：HBM 拷贝带宽 + FP16 矩阵乘；PEP 723 自带 `torch` 依赖，也可在 vLLM 容器里跑。
- `quant_eval.py`：复用第四部分的合成工单抽取题，比较 BF16/FP8/AWQ；字符串判定，只做同题对比。
- `tp_vs_replica.sh`：2 卡上先跑 TP=2，再跑 2 副本，同一批请求、同一并发。
- `capacity.py`：显存/带宽/算力/TTFT/成本的容量计算器；输出是估算，实测在 7.4。

## 测量规矩

1. **每个点至少 3 次**，报中位数和 p95，不拿单次数字下结论。
2. **冷/热分开报**。默认加唯一前缀避免 Prefix Caching；测缓存收益时加 `--no-unique-prefix`，并说明命中率假设。
3. **先 warmup 再计时**。脚本默认每个并发点 warmup 一次；不要跳过。
4. **同一组对照只改一个变量**。TP vs 副本，模型、请求、并发、`max_tokens` 必须一致。
5. **记录完整命令**。脚本会把启动命令和环境写进 `benchmarks/gpu/env/`；手跑的命令也补进去。
6. **路径脱敏**。提交前跑 `bash benchmarks/sanitize-paths.sh benchmarks/gpu/**/*.csv benchmarks/gpu/**/*.txt`，只保留文件名。
7. **数据先下载再关机**。云实例的系统盘不是永久备份。

## 成本记录

每次开机在附录 F 的表里记一行：日期、平台/区域、GPU、单价、开机/关机时间、用途、产物路径。关机前确认控制台显示“已关机”，不是只关终端。

## Mac 上能做什么

- `python3 -m py_compile labs/07-gpu-inference/*.py`
- `bash -n labs/07-gpu-inference/*.sh`
- `bash labs/07-gpu-inference/serve_vllm.sh --dry-run`
- `uv run labs/07-gpu-inference/capacity.py --help`
- `uv run labs/07-gpu-inference/bench_openai.py --help`

CUDA、`nvidia-smi`、vLLM 服务、TP=2 都必须在 GPU 机器上跑；Mac 上不要试图模拟。
