# labs/03-deployment

第三部分（3.1–3.7）的实验脚本，在 M4 / 24GB 上可完整运行。

## 准备

```bash
# 在仓库根目录
source labs/env.sh                         # 设置 $MODEL
cd labs && uv sync                         # 装 openai SDK（唯一依赖）
```

`uv sync` 只装到 `labs/.venv/`，不污染系统 Python。

## 脚本

| 脚本 | 对应章节 | 做什么 |
|---|---|---|
| `serve.sh` | 3.5 | 起 `llama-server`，参数走环境变量 `NP` / `CTX` / `PORT` |
| `smoke.py` | 3.5 | 非流式、流式、多轮三个场景的接入验证 |
| `template_check.py` | 3.3 | 正确模板 vs chatml、思考开 vs 关，记录渲染结果与耗时 |
| `sampling_sweep.py` | 3.4 | 固定 seed 扫四组采样参数，检查可复现性 |
| `concurrency.py` | 3.6 | 自动按 NP=1/2/4 起停服务，测并发曲线 |
| `bench_client.py` | 3.7 | 冷热缓存 × 三种 prompt 长度，测首字延迟与噪声底 |
| `serverlib.py` | — | 共用工具：起停服务、流式计时、写 CSV |

## 用法

```bash
# 1. 起服务（前台，日志在眼前）
labs/03-deployment/serve.sh

# 2. 另开终端跑客户端脚本
cd labs
uv run python 03-deployment/smoke.py
uv run python 03-deployment/sampling_sweep.py
uv run python 03-deployment/bench_client.py

# 3. 需要自己起停服务的脚本，别和上面同时跑（端口会撞）
uv run python 03-deployment/template_check.py    # 用 8081
uv run python 03-deployment/concurrency.py       # 用 8082
```

`serve.sh` 默认端口 8080；客户端脚本用 `BASE_URL` 环境变量覆盖，模型别名用 `LLM_MODEL`：

```bash
BASE_URL=http://127.0.0.1:8083/v1 LLM_MODEL=qwen3.5-9b uv run python 03-deployment/smoke.py
```

## 输出

所有脚本把结果写到仓库的 `benchmarks/<主题>/`：

- `template/`：`template.csv` + 每个变体的服务日志
- `sampling/`：`sweep.csv`
- `concurrency/`：`concurrency.csv` + 每个槽位配置的日志
- `client-bench/`：`api-bench.csv`
- 每个目录附一份 `env.txt`（`bash benchmarks/record-env.sh` 的输出）

正文里的「实测」引用这些文件。**换了机器、升级了引擎、改了参数，就要重新跑并替换数据**，不要在新环境里引用旧数字。

## 已知的坑

- `llama-cli` 从 0.7.0 起默认连续对话，脚本里不要用；需要命令行一次性生成时用 `llama-completion`。
- `-np N` 会让每槽位上下文变成 `-c / N`，比较吞吐前先确认上下文一致。
- 符号链接的模型文件要用 `ls -lL` / `os.path.getsize` 取真实大小，否则会算出 0 字节。
- M4 上 `maximum resident set size` 不一定包含 mmap 的权重页，内存实验要交叉核对 `vm_stat`。
- `llama-bench` 的 CSV 会把模型绝对路径写进 `model_filename` 列。提交数据前先跑
  `bash benchmarks/sanitize-paths.sh benchmarks/*/*.csv benchmarks/*/*.txt`，把路径换成文件名——仓库里不留本机路径。
