---
title: 软件栈：驱动、CUDA、PyTorch 的版本对应，容器里调用 GPU（nvidia-container-toolkit）
---

# 软件栈：驱动、CUDA、PyTorch 的版本对应，容器里调用 GPU（nvidia-container-toolkit）

> **本章要回答的问题**：驱动、CUDA、PyTorch 的版本为什么总要对着装？容器里怎么才能用上 GPU？

GPU 软件栈的报错通常长得很吓人：`CUDA driver version is insufficient`、`undefined symbol`、`no CUDA-capable device is detected`。它们大多不是代码问题，而是四层栈里某一层的版本对不上。这一节把四层拆开，每层给一个查看命令，最后用容器把“宿主机驱动 + 容器内运行时”这条链路跑通。

## 四层栈，四个版本号

从下往上：

| 层 | 作用 | 查看命令 | 版本号长什么样 |
|---|---|---|---|
| GPU 驱动 | 内核模块，负责和显卡说话 | `nvidia-smi` | `Driver Version: 550.90.07` |
| CUDA Toolkit | 编译器、头文件、运行时库 | `nvcc --version` | `release 12.4` |
| PyTorch | 深度学习框架，编译时绑定一个 CUDA 版本 | `python -c "import torch; print(torch.__version__, torch.version.cuda)"` | `2.5.1 12.4` |
| vLLM | 推理引擎，依赖 PyTorch 和 CUDA | `python -c "import vllm; print(vllm.__version__)"` | `0.x.y` |

四层里，**只有驱动是装在宿主机内核里的**；CUDA Toolkit、PyTorch、vLLM 都可以装在容器里。容器不需要自己的驱动，它通过 `nvidia-container-toolkit` 把宿主机的驱动设备和库挂进来（引用 [\[2\]](#ref-2)）。

## 驱动支持的最高 CUDA ≠ 实际 CUDA

`nvidia-smi` 右上角那行 `CUDA Version: 12.4`，是**这个驱动支持的最高 CUDA 版本**，不是机器上装了什么 CUDA。真正的 CUDA Toolkit 版本看 `nvcc --version`；PyTorch 实际用的 CUDA 版本看 `torch.version.cuda`。三个数字可以不一样。

规则可以记成两条：

- **新驱动能跑旧 CUDA runtime**：驱动向下兼容，只要不低于 runtime 要求的最低版本。
- **旧驱动不能跑新 CUDA runtime**：驱动太老时，运行时会报 `CUDA driver version is insufficient for CUDA runtime version`。

NVIDIA 的 CUDA Compatibility 文档把这条规则和 minor version compatibility 讲得很清楚（引用 [\[1\]](#ref-1)）。工程上不用背版本表，记住“驱动是天花板，runtime 不能超过它”就够了。

## 版本对应表怎么查

不要凭记忆拼版本，按官方矩阵查：

- **PyTorch**：安装页会给出每个 PyTorch 版本对应的 CUDA 版本和安装命令（引用 [\[3\]](#ref-3)）。
- **vLLM**：安装文档会给出推荐的 PyTorch/CUDA 组合和预编译 wheel 的 CUDA 版本（引用 [\[4\]](#ref-4)）。
- **驱动**：先看 `nvidia-smi` 的天花板，再选不超过它的 CUDA runtime。

一个实用的原则：**先选 vLLM 的版本，再选它要求的 PyTorch/CUDA，最后确认驱动够新。** 反过来“先装最新驱动，再随便装一个 vLLM”也能跑，但一旦报错，排查顺序会很乱。

常见报错和第一怀疑对象：

| 现象 | 第一怀疑 |
|---|---|
| `torch.cuda.is_available()` 返回 `False` | 容器没拿到 GPU，或驱动没装好 |
| `CUDA driver version is insufficient` | 驱动太老，低于 runtime 要求 |
| `undefined symbol: ...` / `libcudart.so` 找不到 | PyTorch 和 CUDA runtime 不匹配 |
| `import vllm` 时报 CUDA/PyTorch 错误 | vLLM 和 PyTorch/CUDA 的组合不对 |
| 容器里 `nvidia-smi` 正常，`torch.cuda` 不正常 | 镜像里的 PyTorch 是 CPU 版，或 `CUDA_VISIBLE_DEVICES` 被屏蔽 |

## 容器里为什么能跑

容器里没有驱动的内核模块，也不应该有驱动内核模块。启动容器时加 `--gpus all`，Docker 通过 `nvidia-container-toolkit` 做三件事：

1. 把宿主机的 GPU 设备节点暴露给容器；
2. 把宿主机的驱动库挂载到容器里；
3. 设置环境变量，让容器里的 CUDA runtime 找到驱动。

所以容器镜像里只需要 CUDA runtime、PyTorch、vLLM，不需要装 `.run` 驱动。镜像的 CUDA 版本可以低于宿主机驱动支持的最高版本，但不能高于它。

最小验证分三步：

```bash
# 1. 宿主机能看到卡
nvidia-smi

# 2. 容器能看到卡（镜像里的 CUDA 版本按驱动天花板选）
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi

# 3. 容器里的 PyTorch 能用 CUDA
docker run --rm --gpus all --entrypoint python vllm/vllm-openai:latest \
  -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

第 2 步失败，先看 `docker info` 里有没有 nvidia runtime，再看 `nvidia-container-toolkit` 是否安装并配置过（引用 [\[2\]](#ref-2)）。第 3 步失败但第 2 步成功，通常是镜像里的 PyTorch 是 CPU 版，或者 `CUDA_VISIBLE_DEVICES` 设成了空。

## 常见坑

**驱动装不上。** 先装内核头文件，再确认 nouveau 已禁用，装完重启。优云的驱动文档列了这两个常见原因（引用 [\[5\]](#ref-5)）。

**Docker 权限。** `usermod -aG docker` 后要重新登录；只执行命令不重登，`docker` 仍会报 permission denied。

**`--gpus` 忘了加。** 容器里 `nvidia-smi` 报 `could not communicate with the NVIDIA driver`，先看启动命令有没有 `--gpus all`。

**`/dev/shm` 太小。** vLLM 的 TP 和某些算子需要共享内存；起容器时加 `--ipc=host` 或 `--shm-size=16g`。先 `df -h /dev/shm` 看一眼。

**容器里 CUDA 版本比驱动新。** 镜像的 CUDA runtime 高于驱动支持的最高版本，会在启动时报 `CUDA driver version is insufficient`。换低版本镜像，或升级驱动。

**把 `nvidia-smi` 的 CUDA Version 当成 toolkit 版本。** 它只是驱动天花板；`nvcc --version` 才是 toolkit。两个数字不一样是正常的，不要为了“对齐”去乱装驱动。

## 工程含义

**版本要钉住，不要追最新。** 交付时写清楚驱动、CUDA、PyTorch、vLLM 四个版本；升级任何一个都要重新跑一遍基准。容器镜像是最好的钉版本单位：镜像 tag 一固定，四层栈就固定了。

**环境记录要自动化。** 每次实验前先 `mkdir -p benchmarks/gpu/env`，再跑 `bash benchmarks/record-env.sh > benchmarks/gpu/env/<label>.txt`；脚本会把 GPU、驱动、CUDA、PyTorch、vLLM 都打进去。手写版本号迟早会写错。

**容器是交付单位，不是调试单位。** 调试时可以在宿主机上装东西；交付时用镜像，把“在我机器上能跑”变成“在这个镜像里能跑”。

## 动手实验

::: tip 动手实验：把四层版本对上一遍
**假设**：`nvidia-smi` 的 CUDA 上限 ≥ 容器镜像的 CUDA 版本；容器里 `torch.cuda.is_available()` 为 `True`；vLLM 能 import 并打印版本（真正的模型加载在 7.4）。

**命令**（GPU 机器上，仓库根目录）：

```bash
# 1. 四层版本
nvidia-smi
nvcc --version
docker run --rm --gpus all --entrypoint python vllm/vllm-openai:latest \
  -c "import torch, vllm; print('torch', torch.__version__, torch.version.cuda, torch.cuda.is_available()); print('vllm', vllm.__version__)"

# 2. 记录环境
mkdir -p benchmarks/gpu/env
bash benchmarks/record-env.sh > benchmarks/gpu/env/software-stack.txt

# 3. 只暴露一张卡，确认 CUDA_VISIBLE_DEVICES 生效
docker run --rm --gpus all -e CUDA_VISIBLE_DEVICES=0 \
  --entrypoint python vllm/vllm-openai:latest \
  -c "import torch; print(torch.cuda.device_count())"
```

**记录**：四个版本号、容器镜像 tag、`nvidia-smi` 的 CUDA 上限、`torch.cuda.is_available()`、`device_count()`。数据存 `benchmarks/gpu/env/`。如果第 2 步的 CUDA 上限低于镜像版本，先换镜像或升级驱动，不要继续往下跑。
:::

## 参考资料

1. <a id="ref-1" style="scroll-margin-top: 96px"></a>NVIDIA. *CUDA Compatibility*（驱动与 CUDA runtime 的兼容规则）. https://docs.nvidia.com/deploy/cuda-compatibility/
2. <a id="ref-2" style="scroll-margin-top: 96px"></a>NVIDIA. *NVIDIA Container Toolkit Install Guide*. https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
3. <a id="ref-3" style="scroll-margin-top: 96px"></a>PyTorch. *Get Started Locally*（PyTorch 与 CUDA 版本对应）. https://pytorch.org/get-started/locally/
4. <a id="ref-4" style="scroll-margin-top: 96px"></a>vLLM Project. *Installation*（vLLM、PyTorch、CUDA 的版本组合）. https://docs.vllm.ai/en/latest/getting_started/installation.html
5. <a id="ref-5" style="scroll-margin-top: 96px"></a>优云智算. *Linux 安装 Nvidia 驱动和 Cuda*. https://www.compshare.cn/docs/operation/bestpractices/installlinux
