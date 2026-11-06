# 部署手册

> 目标：换一个人，照这份文档能在目标环境重装一遍。  
> 每一步都写：命令、预期输出、失败时先看哪里。不要写“视情况而定”。

## 1. 前置条件

| 项 | 要求 | 怎么确认 |
|---|---|---|
| 操作系统 | 【Linux x86_64 / macOS】 | `uname -a` |
| Python | 【3.11/3.12】 | `python3 --version` |
| GPU/驱动 | 【型号/驱动版本】 | `nvidia-smi` |
| 内存/磁盘 | 【最小内存/可用磁盘】 | `free -h`、`df -h` |
| 网络 | 【离线/可访问内网镜像】 | `ping`/`curl` 内网地址 |
| 端口 | 【8090/8080/8081/8082】 | `ss -lntp` |
| 权限 | 【服务账户/目录权限】 | `id`、`ls -ld` |

## 2. 安装

```bash
# 1) 解包并校验
tar -xf 【交付包.tar.gz】 -C 【目标目录】
cd 【目标目录】
bash verify_manifest.sh          # 或 sha256sum -c MANIFEST.sha256

# 2) 安装 Python 依赖（离线）
bash install.sh

# 3) 配置
cp config.env.example config.env
# 修改：模型路径、端口、权限策略、日志目录

# 4) 启动
bash run.sh
```

**预期输出**：`/health` 返回 200，`chunks` 大于 0，`policy.docs` 与语料一致。

**失败时先看**：`install.log`、`/health` 的 HTTP 状态、服务日志的前 50 行。

## 3. 配置项

| 变量 | 必填 | 默认 | 说明 |
|---|---|---|---|
| `FDE_MODEL_DIR` | 是 | — | 模型目录 |
| `MODEL` | 是 | — | LLM GGUF 路径 |
| `SERVICE_HOST/PORT` | 是 | 127.0.0.1:8090 | 只绑本机，外部走反向代理 |
| `BACKEND_MODEL` | 是 | qwen3.5-9b | 后端 served-model-name |
| `ACCESS_POLICY` | 是 | access.yaml | 权限策略 |
| `LOG_FILE` | 否 | — | JSON 日志文件 |
| `ALLOWED_DOCS` | 否 | 空 | 全局白名单，与用户权限取交集 |

## 4. 验证

```bash
curl -s http://127.0.0.1:8090/health
curl -s -X POST http://127.0.0.1:8090/v1/chat/completions \
  -H 'Content-Type: application/json' -H 'X-User-ID: auditor' \
  -d '{"model":"rag-service","messages":[{"role":"user","content":"【验收问题】"}],"max_tokens":64}'
```

检查：回答非空、引用非空、语料外问题拒答、`/metrics` 有 `rag_requests_total`。

## 5. 回滚

1. 停止服务：`bash stop.sh`（或 `systemctl stop`）。
2. 切回上一版目录/镜像 tag。
3. 恢复上一版索引/配置。
4. 启动并重跑第 4 节验证。
5. 记录回滚原因、时间、影响范围。

**回滚必须在交付前演练一次**；只写“可以回滚”不算。

## 6. 离线注意事项

- 模型、语料、wheels 都在包内；不要在客户现场下载。
- 首次启动如果重建索引，记录耗时；能被 `.cache/` 带走的就带走。
- 时间同步、DNS、代理、防火墙由客户环境负责；写进前置条件。

## 7. 变更记录

| 日期 | 版本 | 改了什么 | 验证人 |
|---|---|---|---|
| 【YYYY-MM-DD】 | 【v0.1.0】 | 【初始部署】 | 【姓名】 |
