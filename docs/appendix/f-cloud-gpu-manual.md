---
title: 附录 F 云 GPU 实验手册（租机、环境准备、成本记录）
---

# 附录 F 云 GPU 实验手册（租机、环境准备、成本记录）

> **本章要回答的问题**：第一次租云 GPU，从开机到关机要注意什么，钱主要花在哪？

这份手册按“先选平台、再开机、装环境、跑实验、记成本、关机”的顺序写。价格是 2026-11-06 的公开挂牌价，平台会调价，下单前以控制台为准；第一次 GPU session 跑完后，把真实账单补进本节表格。

## 先选平台：能不能装 Docker 是分水岭

第七部分的 7.2 要在虚机里装驱动、CUDA、Docker 和 `nvidia-container-toolkit`，7.7 要一台机器上的两张 4090。很多便宜平台是“容器实例”，你拿不到宿主机，装不了 Docker。选平台前先确认这两件事。

| 平台 | 1×4090 24G | 2×4090 同机 | 能不能装 Docker | 结论 |
|---|---|---|---|---|
| 优云智算 | ¥1.92/h 起（14C32G） | ¥4.06/h（28C96G） | 系统镜像（Ubuntu 虚机）可以 | **本书首选** |
| AutoDL | 约 ¥2/h | 支持 | 容器实例内**不支持** Docker | 备选；7.2 要另找虚机 |
| 共绩算力 | 约 ¥1.7–2.0/h | 支持 2/4 卡 | 平台帮你跑自定义镜像，装不了宿主机 toolkit | 备选；7.2 不完整 |
| 恒源云 | 约 ¥1.9/h | 支持多卡 | 容器实例内**不能**安装 Docker | 备选 |

价格来自各平台公开价目页：优云见 [\[1\]](#ref-1)，AutoDL 见 [\[10\]](#ref-10)，共绩见 [\[11\]](#ref-11)，恒源云见 [\[12\]](#ref-12)；具体以下单页为准。AutoDL 的 Docker 限制见 [\[6\]](#ref-6)，恒源云的限制见 [\[7\]](#ref-7)，共绩的自定义镜像见 [\[8\]](#ref-8)。

选优云的理由不是单价最低，而是它同时满足：虚机可以装 Docker、有 2×4090 同机、按秒计费、关机不收费、100GB 系统盘免费（引用 [\[1\]](#ref-1)）。对 8 小时 + 3 小时的短任务，这四条比每小时便宜几毛钱重要。

**两个不要**：认准 **4090 24G**，不要 4090D（和书里的基准不一致）；不要 4090 48G（更贵，且改变了显存对比条件）。

## 租什么配置

| 用途 | 配置 | 价格 |
|---|---|---|
| 单卡主体（7.2–7.6、7.8、7.9） | 1×4090 24G，14C32G | ¥1.92/h（引用 [\[1\]](#ref-1)） |
| 双卡（7.7） | 2×4090 24G，28C96G | ¥4.06/h（引用 [\[1\]](#ref-1)） |

- **镜像**：Ubuntu 22.04 系统镜像（虚机），不是容器镜像。系统镜像没有预装 PyTorch/Jupyter，需要自己装；7.2 要讲的正是这个过程（引用 [\[4\]](#ref-4)）。
- **区域**：华北二A / 上海二B 支持系统镜像和“无卡模式”（引用 [\[3\]](#ref-3)）。
- **磁盘**：100GB 系统盘免费；模型 + Docker 镜像大约 50–60GB，够用。不够再加数据盘，按 GB/天计费。
- **计费**：按秒计费，关机不收费（引用 [\[1\]](#ref-1)）。不要买包日/包月；8 小时按量比包日便宜。

## 开机前清单

- [ ] 账号注册、实名认证、充值（按平台要求）。
- [ ] 预算上限：建议 **¥100**；预期 ¥35–45（见下面的成本表）。
- [ ] 模型权重：Qwen3.5-9B 的 AWQ/FP8 版本，确认仓库名和协议；没有现成量化版就准备现场量化。
- [ ] 代码：仓库 clone 到虚机；不要在虚机上改代码，改完推回本地。
- [ ] SSH 公钥：本地生成，公钥贴到平台；私钥不进仓库。
- [ ] 数据目录：`benchmarks/gpu/` 下的子目录先建好。
- [ ] 关机清单：数据下载、容器停止、实例停止、费用记录。

## 从裸机到 vLLM

### 1. 驱动

先看有没有驱动：

```bash
nvidia-smi
```

没有输出，就按 NVIDIA 官方驱动页下载对应 4090 的 `.run` 包，或者按优云的驱动安装文档操作（引用 [\[2\]](#ref-2)）：

```bash
chmod +x NVIDIA-Linux-x86_64-<version>.run
sudo ./NVIDIA-Linux-x86_64-<version>.run
sudo reboot
nvidia-smi
```

如果报 `kernel headers` 不匹配，先装内核头文件再重装驱动：

```bash
sudo apt update
sudo apt install -y linux-headers-$(uname -r)
```

### 2. CUDA Toolkit（可选）

vLLM 容器自带 CUDA runtime，宿主机只要有驱动就能跑容器。7.2 要对比 `nvcc` 版本，所以可以装一个 CUDA Toolkit：

```bash
# 版本按 NVIDIA CUDA Toolkit Archive 选择；下面只是形式
sudo sh cuda_<version>_linux.run --toolkit --silent
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
nvcc --version
```

装完记下 `nvidia-smi` 右上角的 CUDA Version 和 `nvcc --version`，它们不是一回事：前者是驱动支持的最高版本，后者是实际装的 toolkit 版本。

### 3. Docker

```bash
sudo apt update
sudo apt install -y docker.io docker-compose-v2
sudo systemctl enable --now docker
sudo usermod -aG docker "$USER"
# 退出 SSH 重新登录，或 newgrp docker
docker version
```

`usermod` 之后必须重新登录，组权限才生效；只执行命令不重登，`docker` 仍会报 permission denied。

### 4. nvidia-container-toolkit

按 NVIDIA 官方安装指南（引用 [\[5\]](#ref-5)）：

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

验证容器里能看到 GPU：

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

这一步成功，7.2 的核心实验就完成了。如果失败，先看 `docker info` 里有没有 nvidia runtime，再看驱动和 toolkit 版本。

### 5. 起 vLLM

先下载模型到本地目录（用 ModelScope 或 HF 镜像，见下一节），然后用脚本起服务：

```bash
# 在仓库根目录
bash labs/07-gpu-inference/serve_vllm.sh \
  --model-dir models/qwen3.5-9b-awq \
  --model-name qwen3.5-9b-awq \
  --max-model-len 8192 \
  --gpu-mem-util 0.90
```

脚本会挂载模型目录、起 `vllm/vllm-openai` 容器、等 `/health` 就绪，并把启动命令写到 `benchmarks/gpu/env/`。镜像和参数的口径见 vLLM 官方 Docker 文档（引用 [\[9\]](#ref-9)）。下面是最小启动命令；脚本还会补上 `--max-num-seqs`、`--max-num-batched-tokens`、`--dtype`、Prefix Caching/Chunked Prefill 等参数，完整命令以 `benchmarks/gpu/env/*-serve.txt` 为准：

```bash
docker run --rm --gpus all --ipc=host \
  -p 127.0.0.1:8000:8000 \
  -v "$PWD/models/qwen3.5-9b-awq:/models:ro" \
  vllm/vllm-openai:latest \
  --model /models --served-model-name qwen3.5-9b-awq \
  --max-model-len 8192 --gpu-memory-utilization 0.90
```

端口只绑 `127.0.0.1`。本地通过 SSH 隧道访问，不要把 8000 暴露到公网：

```bash
ssh -L 8000:127.0.0.1:8000 ubuntu@<实例地址>
curl http://127.0.0.1:8000/v1/models
```

## 模型怎么上去

优先用 ModelScope（国内直连），或者给 Hugging Face 设镜像：

```bash
# ModelScope
pip install modelscope
modelscope download --model Qwen/Qwen3.5-9B-AWQ --local_dir models/qwen3.5-9b-awq

# 或者 Hugging Face 镜像
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download Qwen/Qwen3.5-9B-AWQ --local-dir models/qwen3.5-9b-awq
```

仓库名以模型卡为准；下载前确认协议。模型权重不进 git，也不要在容器里现场下载——先把权重落到数据盘，再用 `-v` 只读挂载，容器重启不用重下。

## 成本怎么记

每次开机记一行，关机前补完：

| 日期 | 平台/区域 | GPU | 单价 | 开机 | 关机 | 用途 | 产物 |
|---|---|---|---|---|---|---|---|
| 2026-11-xx | 优云/华北二A | 1×4090 | ¥1.92/h | 10:00 | 18:00 | 7.2–7.6 | `benchmarks/gpu/...` |
| 2026-11-xx | 优云/华北二A | 2×4090 | ¥4.06/h | 10:00 | 13:00 | 7.7 | `benchmarks/gpu/multi-gpu/` |

预算表（估算，按 2026-11-06 公开价）：

| 项目 | 配置 | 单价 | 时长 | 小计 |
|---|---|---|---|---|
| 单卡主体 | 1×4090 24G / 14C32G | ¥1.92/h | 8h | ¥15.36 |
| 双卡 7.7 | 2×4090 24G / 28C96G | ¥4.06/h | 3h | ¥12.18 |
| 试错重测 | 单卡 2h + 双卡 1h | | | ¥7.90 |
| 存储/流量 | 100GB 内 | ~¥0 | | <¥1 |
| **合计** | | | | **约 ¥36** |

**预算 = 单卡小时数 × 单价 + 双卡小时数 × 单价 + 存储/流量 + 30–50% 试错缓冲。** 上限设 ¥100，是为了不用中途停下来算钱，不是一定要花完。实际账单出来后，把上面两行换成真实数字。

省钱的两个动作：

1. **无卡模式**：优云支持 ¥0.15/h 的无卡模式（引用 [\[3\]](#ref-3)），用来装 Docker、配镜像加速、下载权重；装完再关机换带 GPU 的实例。
2. **打自定义镜像**：环境装好后做一次镜像，双卡 session 直接用它开机，省掉重复装驱动的时间。镜像存储按 GB/天计费，几十 GB 一天几毛钱。

## 关机前检查

- [ ] 数据已下载到本地：`benchmarks/gpu/` 下的 CSV/JSON/日志。
- [ ] 容器已停止：`docker ps` 没有残留；需要保留的环境已打成镜像。
- [ ] 实例已停止（不是只关终端）；确认控制台显示“已关机”，计费停止。
- [ ] 费用记录已写进上表。
- [ ] 模型权重和客户数据按需要删除；不要留在共享盘上。

## 常见坑

**驱动装不上。** 先看内核头文件是否匹配，再确认 nouveau 已禁用；装完必须重启。优云的驱动文档列了这两个常见原因（引用 [\[2\]](#ref-2)）。

**Docker 权限。** `usermod -aG docker` 后要重新登录；不想重登就每次 `sudo docker`，但别把 `sudo` 写进脚本。

**`/dev/shm` 太小。** vLLM 的 TP 和某些算子需要共享内存；起容器时加 `--ipc=host` 或 `--shm-size=16g`。先 `df -h /dev/shm` 看一眼，小于 1GB 就会出问题。

**端口暴露。** vLLM 默认没有鉴权；只绑 `127.0.0.1`，用 SSH 隧道访问。要多人用，前面加反向代理和鉴权，这是第八部分的内容。

**下载慢。** 优先 ModelScope；HF 用 `HF_ENDPOINT=https://hf-mirror.com`。容器里设 `HF_HUB_OFFLINE=1`，避免启动时联网重试。

**4090 没有 NVLink。** 两张 4090 做 TP=2 走 PCIe，别指望线性加速；7.7 会实测它和 2 个副本的差别。NVLink 的数字引用厂商规格，不写成实测。

**抢占式实例。** 便宜但会被中断；本书的实验用按需实例。跑长任务前先确认计费模式和中断策略。

**数据保留。** 关机后系统盘通常保留一段时间，但不是永久备份；实验结束前把结果下载到本地。释放实例前确认没有唯一副本。

## 安全与合规

- **密钥不入库。** API key、SSH 私钥、平台 token 一律走环境变量；不要写进脚本、日志或 notebook。
- **模型权重不入库。** 仓库的 `.gitignore` 已覆盖权重；不要 `git add -f` 绕过。
- **不暴露服务。** vLLM 只绑本机；需要远程访问用 SSH 隧道。
- **客户数据脱敏。** 实验用公开语料或合成数据；真实工单、合同、身份证号不要上传到共享 GPU 平台。
- **协议确认。** 模型和量化权重的协议各不相同；写进正文前确认是否允许再分发。

## 参考资料

1. <a id="ref-1" style="scroll-margin-top: 96px"></a>优云智算. *计费概览*（4090 24G 按量价、2×4090 配置、按秒计费、100GB 系统盘）. https://www.compshare.cn/docs/operation/charge/bill
2. <a id="ref-2" style="scroll-margin-top: 96px"></a>优云智算. *Linux 安装 Nvidia 驱动和 Cuda*. https://www.compshare.cn/docs/operation/bestpractices/installlinux
3. <a id="ref-3" style="scroll-margin-top: 96px"></a>优云智算. *使用问题 FAQ*（无卡模式、Docker 安装、系统镜像限制）. https://compshare.cn/docs/operation/gpufaq
4. <a id="ref-4" style="scroll-margin-top: 96px"></a>优云智算. *镜像介绍与选择*（容器实例 vs 系统镜像虚机）. https://www.compshare.cn/docs/operation/gpu/community
5. <a id="ref-5" style="scroll-margin-top: 96px"></a>NVIDIA. *NVIDIA Container Toolkit Install Guide*. https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
6. <a id="ref-6" style="scroll-margin-top: 96px"></a>AutoDL. *环境*（容器实例内不支持使用 Docker）. https://www.autodl.com/docs/env/
7. <a id="ref-7" style="scroll-margin-top: 96px"></a>恒源云. *环境（重要）*（容器实例不能安装和使用 Docker）. https://gpushare.com/docs/instance/environment/
8. <a id="ref-8" style="scroll-margin-top: 96px"></a>共绩算力. *运行在线推理服务*（自定义镜像与容器部署）. https://docs.suanli.cn/choose/serverless
9. <a id="ref-9" style="scroll-margin-top: 96px"></a>vLLM Project. *Deployment with Docker*. https://docs.vllm.ai/en/latest/deployment/docker.html
10. <a id="ref-10" style="scroll-margin-top: 96px"></a>AutoDL. *计费说明*. https://www.autodl.com/docs/price/
11. <a id="ref-11" style="scroll-margin-top: 96px"></a>共绩算力. *定价*. https://www.gongjiyun.com/pricing
12. <a id="ref-12" style="scroll-margin-top: 96px"></a>恒源云. *云市场*. https://gpushare.com/
