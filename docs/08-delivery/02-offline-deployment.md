---
title: 离线部署与断网验证
---

# 离线部署与断网验证

> **本章要回答的问题**：客户现场完全断网，怎么把整套系统搬进去，并验证它真的能跑？

离线部署不是“把代码拷过去，祈祷它能跑”。它是一个**可复现的构建产物**：代码、模型、语料、依赖、配置、校验清单，一次性打包；到现场只做校验、安装、启动、验收。**能离线跑**和**证明它离线跑**是两件事，后者需要断网证据。

## 离线有三层

| 层级 | 要求 | 验证方式 |
|---|---|---|
| 不访问公网 | 服务、模型、依赖都不发外网请求 | 新 net namespace、Docker `--network none`、代理黑洞 |
| 不装包 | 只从 bundle 的 `wheels/` 安装 | `pip install --no-index --find-links wheels`（引用 [\[2\]](#ref-2)） |
| 不下模型 | 模型在 `models/`，路径固定 | `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1` |
| 不重建索引（可选） | `.cache/` 带 chunks/vectors | 首次启动日志里没有重新切分/嵌入 |

“内网能访问一个镜像源”不算离线；“模型能提前下载”也不够——现场可能连 DNS 都没有。**按最坏情况准备。**

## bundle 里有什么

```text
bundle/
  labs/05-rag/          检索/重排/提示词代码
  labs/08-delivery/     服务与离线脚本
  labs/env.sh           定位 LLM 的 $MODEL
  corpus/               sources.yaml、index.json、raw/（原文或节选）
  labs/05-rag/.cache/   chunks-structure-real.jsonl、vectors-1024-real.npy
  eval/qa.jsonl         现场冒烟用的题集
  benchmarks/           运行产物（反馈/验证日志）
  models/               三个 GGUF
  wheels/               Python 依赖
  requirements.txt      uv export 的结果
  config.env            路径/端口/权限
  install.sh            venv + 离线 pip install
  run.sh                起服务
  verify.sh             校验 + 冒烟
  MANIFEST.sha256       逐文件哈希
  bundle.json           构建平台、Python、跳过项
```

三个容易漏的东西：

- **索引缓存**：没有 `chunks/vectors`，现场首次启动会重新切分/嵌入；能跑，但时间不可控。把缓存带上，或者把“重建索引”写进验收时间。
- **系统库**：`llama-server` 依赖的 libstdc++、CUDA 驱动、glibc 不在 bundle 里；它们是目标机基线，必须写进前置条件。
- **配置**：模型路径、端口、权限、日志目录，集中到 `config.env`；不要散落在代码和命令行里。

## 构建一次，带到现场

**平台必须匹配。** wheels 和二进制按平台/架构/Python 版本构建；Mac ARM 的 bundle 不能直接搬到 Linux x86_64 GPU 服务器。给 GPU 服务器构建时，在 Linux x86_64 的联网机器上跑 `bundle.sh`，或把这一步放进 CI。

```bash
# 联网构建机
bash labs/08-delivery/offline/bundle.sh

# 现场
bash verify.sh --mode check     # 清单 + 必需文件
bash install.sh                 # 只从 wheels/ 安装
bash run.sh                     # 起服务
```

`bundle.sh` 会做：复制代码（排除缓存/日志）、复制语料与索引缓存、复制三个模型、`uv export` 出 requirements、`pip download` 出 wheels、生成 `MANIFEST.sha256` 和 `bundle.json`（uv 的锁文件导出见引用 [\[1\]](#ref-1)）。构建产物默认在 `dist/offline-<时间>/`，不入库。

**校验清单是交付物的一部分。** `MANIFEST.sha256` 逐文件哈希；现场先校验，再安装。哈希不对，不要“先跑起来看看”——模型/语料被替换过，跑出来的结果不可信。

## 断网验证

三种方式，证明力从低到高：

| 方式 | 命令 | 证明什么 | 局限 |
|---|---|---|---|
| 代理黑洞 | `HTTP_PROXY=http://127.0.0.1:9` | 走代理的 HTTP(S) 会失败 | 原始 socket/DNS 仍可能出去 |
| Docker `--network none`（引用 [\[3\]](#ref-3)） | `docker run --network none ...` | 容器内无网络 | 需要把模型/服务放进同一容器 |
| 新 net namespace（引用 [\[4\]](#ref-4)） | `sudo unshare -n ...` | 进程树无网络（lo 除外） | 需要 root；Linux only |

`verify.sh` 把三种模式都封装了：

```bash
bash verify.sh --mode check      # 只校验
bash verify.sh --mode proxy      # Mac 上近似断网
bash verify.sh --mode local      # 不隔离，只冒烟
sudo bash verify.sh --mode unshare  # Linux 真断网
```

`--mode unshare` 会新建 net namespace、把 loopback 起起来，再跑 `--mode local`；这样本地 `127.0.0.1` 的 llama-server 仍可用，外网不可达。`verify.sh` 会检查：清单、必需文件、服务健康、已知问题（回答+引用）、语料外问题（拒答）、反馈接口、metrics。

**验收证据要留**：`verify-answer.json`、`verify-refusal.json`、`verify.log`，以及执行时间/执行人。只写“已离线验证”不算。

## 平台差异

| 差异 | 后果 | 处理 |
|---|---|---|
| 架构（arm64 vs x86_64） | wheel/二进制装不上 | 在目标架构上构建 |
| glibc 版本 | 旧系统跑不了新 wheel | 用目标基线镜像/容器 |
| Python 版本 | 依赖不兼容 | `bundle.json` 记录，目标机同 minor |
| CUDA 驱动 | GPU 跑不起来 | 驱动属于目标机基线；先 `nvidia-smi` |
| 路径/权限 | 模型读不到、反馈写不了 | `config.env` + 服务账户权限 |
| 时区/编码 | 日志时间错、中文乱码 | UTF-8 + 明确时区 |

**不要把 Mac 的 wheel 搬到 Linux。** 这是离线部署最常见、也最浪费时间的一种失败。

## 常见失败

| 现象 | 先查 | 处理 |
|---|---|---|
| `pip install` 找不到包 | `wheels/` 是否包含该平台 wheel | 在目标平台重建；或改用容器镜像 |
| `llama-server` 起不来 | `ldd`、CUDA 驱动、模型路径 | 补系统库/驱动；修 `config.env` |
| 首次启动很慢 | 是否在重建 chunks/vectors | 带上 `.cache/`；或接受重建时间 |
| 端口冲突 | `SERVICE_PORT`、8080/8081/8082 | 改端口；`VERIFY_PORT` 覆盖 |
| 回答为空/引用为空 | 检索是否命中、模型是否加载 | 看 `/health`、日志、`verify.log` |
| 反馈写不进去 | 目录权限、磁盘满 | 修权限；反馈目录可配置 |
| 断网后仍能访问外网 | 隔离方式不对 | 用 `unshare -n`/`--network none` 重验 |

## 安全与合规

- **bundle 里不放密钥。** API key、SSH 私钥、平台 token 走现场配置/密钥管理，不进包。
- **校验再安装。** `MANIFEST.sha256` 是防篡改的最低要求；更高要求是签名。
- **数据分级。** 模型和语料可能含客户数据；传输用受控介质，落地后按客户要求加密/删档。
- **协议确认。** `corpus/sources.yaml` 里的 `redistributable` 档位决定语料能不能随包分发；模型协议单独确认。
- **最小权限。** 服务账户只读模型/语料，只写反馈/日志目录；不要用 root 跑服务。

## 最小实现

`labs/08-delivery/offline/` 提供：

- `bundle.sh`：构建 bundle（支持 `--skip-models`/`--skip-wheels`/`--dry-run`）；
- `verify.sh`：check/proxy/local/unshare 四种模式；
- `README.md`：现场清单与常见失败。

本机（Mac ARM）验证了：`--skip-models --skip-wheels` 的 bundle 结构、`MANIFEST.sha256` 校验、`--mode proxy` 的完整冒烟（服务、已知问题、拒答、反馈、metrics）通过。Linux `--mode unshare` 和完整模型拷贝待目标机执行，正文不把它写成已验收。

## 动手实验

::: tip 动手实验：在目标机做一次真断网验收
**假设**：bundle 在目标平台构建；`unshare -n` 下服务仍能启动；已知问题有引用、语料外问题拒答；`verify.log` 无外网错误。

**命令**（Linux 目标机，root）：

```bash
# 1. 构建（联网机器，和目标同架构/Python）
bash labs/08-delivery/offline/bundle.sh

# 2. 带到现场，先校验
bash verify.sh --mode check

# 3. 安装（只用 wheels/）
bash install.sh

# 4. 真断网验收
sudo bash verify.sh --mode unshare
```

**记录**：`verify-answer.json`、`verify-refusal.json`、`verify.log`、`bundle.json`、`MANIFEST.sha256` 校验输出；目标机 `uname -a`、`python3 --version`、`nvidia-smi`。数据存 `benchmarks/delivery/`。**如果 `unshare` 下失败、proxy 下成功，说明还有原始 socket/DNS 路径没堵住，不要放过。**
:::

## 工程含义

**离线部署是构建问题，不是现场问题。** 现场只做校验、安装、启动；凡是需要“现场下载/编译/猜路径”的，都是构建阶段没做好。

**断网证据要可复现。** 写清楚隔离方式、命令、时间、执行人；`proxy` 只能算近似，不能写成“真断网”。

**平台基线写进合同/前置条件。** glibc、Python、CUDA 驱动、磁盘、端口，这些不是 bundle 能解决的；提前确认，避免现场返工。

**bundle 是交付物，不是临时压缩包。** 它有版本、有清单、有构建记录；下次升级可以 diff，出问题可以回滚。

## 参考资料

1. <a id="ref-1" style="scroll-margin-top: 96px"></a>Astral. *uv documentation*. https://docs.astral.sh/uv/
2. <a id="ref-2" style="scroll-margin-top: 96px"></a>pip. *pip install*（`--no-index`、`--find-links`）. https://pip.pypa.io/en/stable/cli/pip_install/
3. <a id="ref-3" style="scroll-margin-top: 96px"></a>Docker. *Run reference: network settings*（`--network none`）. https://docs.docker.com/engine/reference/run/#network-settings
4. <a id="ref-4" style="scroll-margin-top: 96px"></a>Linux man-pages. *unshare(1)*. https://man7.org/linux/man-pages/man1/unshare.1.html
