# 8.2 离线部署与断网验证

目标：在**没有互联网、没有包索引、没有模型下载**的客户现场，把同一套服务装起来并证明它能跑。

## 离线是什么意思

| 层级 | 要求 | 验证方式 |
|---|---|---|
| 不访问公网 | 服务、模型、依赖都不发外网请求 | 新 net namespace / `--network none` / 代理黑洞 |
| 不装包 | 只从 `wheels/` 安装 | `pip install --no-index --find-links wheels` |
| 不下模型 | 模型在 `models/`，路径由 `config.env` 固定 | `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1` |
| 不重建索引（可选） | `labs/05-rag/.cache/` 带 chunks/vectors | 首次启动日志里没有重新切分/嵌入 |

## bundle 里有什么

```text
bundle/
  labs/05-rag/          检索/重排/提示词代码
  labs/08-delivery/     服务与离线脚本
  labs/env.sh           定位 LLM 的 $MODEL
  corpus/               sources.yaml、index.json、raw/（原文或节选）
  labs/05-rag/.cache/   chunks-structure-real.jsonl、vectors-1024-real.npy
  models/               三个 GGUF（--skip-models 时为占位说明）
  wheels/               Python 依赖（离线安装用）
  requirements.txt      uv export 的结果
  config.env            路径/端口/权限
  install.sh            venv + 离线 pip install
  run.sh                起服务
  verify.sh             校验 + 冒烟
  MANIFEST.sha256       逐文件哈希
  bundle.json           构建平台、Python、跳过项
```

## 构建

**平台必须匹配。** wheels 和二进制是按平台/架构/Python 版本构建的；Mac ARM 的 bundle 不能直接搬到 Linux x86_64 GPU 服务器。给 GPU 服务器构建时，在 Linux x86_64 的联网机器上跑：

```bash
bash labs/08-delivery/offline/bundle.sh
```

只验证结构（不拷模型/wheels）：

```bash
bash labs/08-delivery/offline/bundle.sh --skip-models --skip-wheels
```

构建产物默认在 `dist/offline-<时间>/`，不入库。

## 安装与启动

```bash
cd dist/offline-*
bash verify.sh --mode check     # 清单 + 必需文件
bash install.sh                 # 只从 wheels/ 安装
bash run.sh                     # 起服务，默认 127.0.0.1:8090
```

## 断网验证

```bash
# Linux 目标机，真断网（root）
sudo bash verify.sh --mode unshare

# Mac / 不方便建 net namespace 时，近似断网
bash verify.sh --mode proxy

# 不隔离网络，只做功能冒烟
bash verify.sh --mode local
```

`verify.sh` 会做：清单校验 → 起服务 → 问一个已知问题（检查回答和引用）→ 问一个语料外问题（检查拒答）→ 反馈接口 → metrics。失败会打印 `verify.log` 的尾部。

**proxy 模式不是真断网。** 它把 `HTTP_PROXY/HTTPS_PROXY` 指到 `127.0.0.1:9`，只黑洞 HTTP(S) 代理路径；DNS、原始 socket 仍可能出去。真验收用 `unshare -n`（Linux）或 Docker `--network none`。

## 现场验收清单

- [ ] `MANIFEST.sha256` 全部通过，没有缺文件。
- [ ] `python3 --version` 与 `bundle.json` 记录一致。
- [ ] `nvidia-smi`/`llama-server` 能跑（如果目标机用 GPU/llama.cpp）。
- [ ] `bash install.sh` 全程无网络请求。
- [ ] `bash run.sh` 后 `/health` 返回 200，块数与语料一致。
- [ ] 已知问题有答案、有引用；语料外问题正确拒答。
- [ ] `/metrics` 能拉到 `rag_requests_total`。
- [ ] `verify-answer.json`/`verify-refusal.json`/`verify.log` 留存。
- [ ] 断网方式与证据写进交付文档（命令 + 时间 + 执行人）。

## 常见失败

- **glibc/架构不匹配**：Linux x86_64 的 wheel 在 ARM 或旧 glibc 上装不上；在目标同构机器上构建。
- **Python 版本不一致**：`bundle.json` 记了构建版本；目标机用同一个 minor。
- **模型哈希不对**：`MANIFEST.sha256` 会拦住；不要跳过校验。
- **索引缓存缺失**：首次启动会重建 chunks/vectors，需要 embedding 服务；时间会变长，但不是离线失败。
- **端口冲突**：`config.env` 的 `SERVICE_PORT` 改成空闲端口；`verify.sh` 用 `VERIFY_PORT` 覆盖。
- **系统库缺失**：`llama-server` 依赖的 libstdc++/CUDA 驱动不在 bundle 里；这些属于目标机基线，写进前置条件。
- **权限**：容器/服务账户要能读 `models/`、写 `benchmarks/delivery/`；SELinux/AppArmor 可能拦截。
- **时区/编码**：日志和反馈用 UTC+本地时区；数据库/文件系统用 UTF-8。

## 安全

- bundle 里不要放 API key、SSH 私钥、平台 token；`config.env` 只放路径和端口。
- 模型和语料可能含客户数据；传输用受控介质，落地后按客户要求加密/删档。
- 交付前核对模型/语料协议；`corpus/sources.yaml` 里的 `redistributable` 档位决定能不能随包分发。
