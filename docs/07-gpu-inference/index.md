---
title: 第七部分 GPU 服务器与推理加速
---

# 第七部分 GPU 服务器与推理加速

> **这一部分要回答的问题**：承接第六部分：Mac 上一个人用没问题，工厂 50 人同时用、要上服务器怎么办？（原理在 Mac 上学，实操租云 GPU）

九节从硬件讲到容量（图 7-1）：先认卡、再认软件栈，能跑起来之后才谈调优，最后把前面所有数字收成一张选卡清单。7.8 是这一部分的出口——卡住了照它排查。

<Figure
  id="7-1"
  src="/figures/7-part7-map.svg"
  alt="第七部分九节的顺序：GPU 硬件常识、软件栈、监控与诊断、vLLM 部署、GPU 端量化、推理加速、多卡、OOM 与卡顿、容量规划"
  caption="前四节把机器跑起来，中间三节榨性能，最后两节回答「出问题怎么办、要买几张卡」。"
/>

- [7.1 GPU 硬件常识](./01-gpu-hardware)
- [7.2 软件栈](./02-software-stack)
- [7.3 监控与诊断](./03-monitoring)
- [7.4 vLLM 部署](./04-vllm)
- [7.5 GPU 端量化实操](./05-gpu-quantization)
- [7.6 推理加速](./06-inference-acceleration)
- [7.7 多卡](./07-multi-gpu)
- [7.8 卡顿与 OOM 排查](./08-oom-and-latency)
- [7.9 容量规划](./09-capacity-planning)
