---
layout: home

hero:
  name: FDE Handbook
  text: 从本地大模型到 RAG 落地
  tagline: 给会写代码、正面临 AI 转行危机的工程师——一条能在自己电脑上走完的落地路径
  actions:
    - theme: brand
      text: 从序章开始
      link: /00-preface/01-why-fde
    - theme: alt
      text: 目录结构
      link: /00-preface/

features:
  - title: 一台 Mac 走完全链路
    details: Apple M4 / 24GB 统一内存为基线，零成本跑通部署、量化、RAG 与评估；GPU 部分租云显卡完成。
  - title: 先跑起来，再讲原理
    details: 每章由上一章动手时冒出的问题引出，原理只讲到能解决问题的深度。
  - title: 数字可复现
    details: 每个数据都写明硬件、软件版本与命令，原始测量数据放在 benchmarks/，并区分实测 🧪、估算 📐 与引用 📚。
---

## 全书结构

### [序](./00-preface/)

- [0.1 写给焦虑的工程师](./00-preface/01-why-fde)
- [0.2 全景图](./00-preface/02-system-overview)
- [0.3 硬件与环境](./00-preface/03-hardware-and-environment)
- [0.4 怎么读这本书](./00-preface/04-how-to-read)

### [第一部分 推理原理（只学到"够用"）](./01-inference-basics/)

- [1.1 Token 与分词](./01-inference-basics/01-tokens)
- [1.2 自回归生成](./01-inference-basics/02-autoregressive)
- [1.3 Prefill 与 Decode](./01-inference-basics/03-prefill-and-decode)
- [1.4 KV Cache](./01-inference-basics/04-kv-cache)
- [1.5 带宽瓶颈](./01-inference-basics/05-bandwidth-bottleneck)
- [1.6 显存估算](./01-inference-basics/06-vram-estimation)

### [第二部分 模型选型](./02-model-selection/)

- [2.1 模型家族与版本](./02-model-selection/01-model-families)
- [2.2 参数量取舍](./02-model-selection/02-parameter-tradeoff)
- [2.3 Dense 与 MoE](./02-model-selection/03-dense-vs-moe)
- [2.4 上下文、协议与中文](./02-model-selection/04-context-and-license)
- [2.5 榜单怎么看](./02-model-selection/05-leaderboards)
- [2.6 选型方法论](./02-model-selection/06-selection-method)

### [第三部分 本地部署](./03-deployment/)

- [3.1 模型格式](./03-deployment/01-model-formats)
- [3.2 推理引擎对比](./03-deployment/02-engine-comparison)
- [3.3 对话模板](./03-deployment/03-chat-template)
- [3.4 采样参数](./03-deployment/04-sampling)
- [3.5 服务化](./03-deployment/05-openai-compatible-api)
- [3.6 并发与批处理](./03-deployment/06-concurrency)
- [3.7 性能测量方法](./03-deployment/07-benchmarking)

### [第四部分 量化](./04-quantization/)

- [4.1 数值表示](./04-quantization/01-number-formats)
- [4.2 量化原理](./04-quantization/02-quantization-principles)
- [4.3 GGUF K-quants](./04-quantization/03-gguf-k-quants)
- [4.4 imatrix](./04-quantization/04-imatrix)
- [4.5 KV Cache 量化](./04-quantization/05-kv-cache-quantization)
- [4.6 评估量化损失](./04-quantization/06-measuring-loss)
- [4.7 三条量化路线](./04-quantization/07-route-selection)

### [第五部分 RAG 全链路](./05-rag/)

- [5.1 为什么用 RAG](./05-rag/01-why-rag)
- [5.2 文档解析](./05-rag/02-document-parsing)
- [5.3 切分](./05-rag/03-chunking)
- [5.4 Embedding](./05-rag/04-embedding)
- [5.5 向量库与 ANN 索引](./05-rag/05-vector-index)
- [5.6 稀疏检索与混合检索](./05-rag/06-hybrid-search)
- [5.7 重排](./05-rag/07-reranking)
- [5.8 生成](./05-rag/08-generation)
- [5.9 元数据与增量更新](./05-rag/09-metadata-and-updates)

### [第六部分 评估与调优](./06-evaluation/)

- [6.1 构建测试集](./06-evaluation/01-building-test-set)
- [6.2 检索指标](./06-evaluation/02-retrieval-metrics)
- [6.3 生成指标](./06-evaluation/03-generation-metrics)
- [6.4 消融实验](./06-evaluation/04-ablation)
- [6.5 错误回答归因](./06-evaluation/05-error-attribution)

### [第七部分 GPU 服务器与推理加速](./07-gpu-inference/)

- [7.1 GPU 硬件常识](./07-gpu-inference/01-gpu-hardware)
- [7.2 软件栈](./07-gpu-inference/02-software-stack)
- [7.3 监控与诊断](./07-gpu-inference/03-monitoring)
- [7.4 vLLM 部署](./07-gpu-inference/04-vllm)
- [7.5 GPU 端量化实操](./07-gpu-inference/05-gpu-quantization)
- [7.6 推理加速](./07-gpu-inference/06-inference-acceleration)
- [7.7 多卡](./07-gpu-inference/07-multi-gpu)
- [7.8 卡顿与 OOM 排查](./07-gpu-inference/08-oom-and-latency)
- [7.9 容量规划](./07-gpu-inference/09-capacity-planning)

### [第八部分 工程化与交付](./08-delivery/)

- [8.1 服务架构与流式输出](./08-delivery/01-service-architecture)
- [8.2 离线部署与断网验证](./08-delivery/02-offline-deployment)
- [8.3 压测与容量验收](./08-delivery/03-load-testing)
- [8.4 可观测性](./08-delivery/04-observability)
- [8.5 数据安全与权限](./08-delivery/05-security-and-permissions)
- [8.6 从 Mac 迁移到 GPU](./08-delivery/06-mac-to-gpu)
- [8.7 交付物](./08-delivery/07-deliverables)

### [附录](./appendix/)

- [A 显存速查表](./appendix/a-vram-cheatsheet)
- [B 术语表（中英对照）](./appendix/b-glossary)
- [C 故障排查手册](./appendix/c-troubleshooting)
- [D 硬件选型参考](./appendix/d-hardware-reference)
- [E 面试题索引](./appendix/e-interview-index)
- [F 云 GPU 实验手册](./appendix/f-cloud-gpu-manual)
- [G 参考文献](./appendix/g-references)
