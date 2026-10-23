import { defineConfig } from 'vitepress'

// 侧边栏即书的目录，顺序与 /Users/dnevend/.opencode/plan/fde-handbook-outline-v1.md 的
// 「四、详细目录」一一对应。增删章节时两边同步。
export default defineConfig({
  lang: 'zh-CN',
  title: 'FDE Handbook',
  description: '从本地大模型到 RAG 落地',
  cleanUrls: true,
  srcExclude: ['**/README.md'],

  themeConfig: {
    nav: [
      { text: '序', link: '/00-preface/' },
      { text: '推理原理', link: '/01-inference-basics/' },
      { text: 'RAG 全链路', link: '/05-rag/' },
      { text: 'GPU 与加速', link: '/07-gpu-inference/' },
      { text: '附录', link: '/appendix/' },
    ],

    sidebar: [
      {
        text: '序',
        collapsed: false,
        items: [
          { text: '序 · 导言', link: '/00-preface/' },
          { text: '0.1 写给焦虑的工程师', link: '/00-preface/01-why-fde' },
          { text: '0.2 全景图', link: '/00-preface/02-system-overview' },
          { text: '0.3 硬件与环境', link: '/00-preface/03-hardware-and-environment' },
          { text: '0.4 怎么读这本书 / 怎么复现', link: '/00-preface/04-how-to-read' },
        ],
      },
      {
        text: '第一部分 推理原理（只学到"够用"）',
        collapsed: false,
        items: [
          { text: '第一部分 推理原理（只学到"够用"） · 导言', link: '/01-inference-basics/' },
          { text: '1.1 Token 与分词', link: '/01-inference-basics/01-tokens' },
          { text: '1.2 自回归生成', link: '/01-inference-basics/02-autoregressive' },
          { text: '1.3 Prefill 与 Decode', link: '/01-inference-basics/03-prefill-and-decode' },
          { text: '1.4 KV Cache', link: '/01-inference-basics/04-kv-cache' },
          { text: '1.5 带宽瓶颈', link: '/01-inference-basics/05-bandwidth-bottleneck' },
          { text: '1.6 显存估算', link: '/01-inference-basics/06-vram-estimation' },
        ],
      },
      {
        text: '第二部分 模型选型',
        collapsed: false,
        items: [
          { text: '第二部分 模型选型 · 导言', link: '/02-model-selection/' },
          { text: '2.1 认识模型家族与版本', link: '/02-model-selection/01-model-families' },
          { text: '2.2 参数量取舍', link: '/02-model-selection/02-parameter-tradeoff' },
          { text: '2.3 Dense 与 MoE', link: '/02-model-selection/03-dense-vs-moe' },
          { text: '2.4 上下文长度、开源协议、中文能力', link: '/02-model-selection/04-context-and-license' },
          { text: '2.5 榜单怎么看，以及为什么要用自己的数据评测', link: '/02-model-selection/05-leaderboards' },
          { text: '2.6 选型方法论', link: '/02-model-selection/06-selection-method' },
        ],
      },
      {
        text: '第三部分 本地部署',
        collapsed: false,
        items: [
          { text: '第三部分 本地部署 · 导言', link: '/03-deployment/' },
          { text: '3.1 模型格式', link: '/03-deployment/01-model-formats' },
          { text: '3.2 推理引擎对比', link: '/03-deployment/02-engine-comparison' },
          { text: '3.3 对话模板', link: '/03-deployment/03-chat-template' },
          { text: '3.4 采样参数', link: '/03-deployment/04-sampling' },
          { text: '3.5 服务化', link: '/03-deployment/05-openai-compatible-api' },
          { text: '3.6 并发与批处理', link: '/03-deployment/06-concurrency' },
          { text: '3.7 性能测量方法', link: '/03-deployment/07-benchmarking' },
        ],
      },
      {
        text: '第四部分 量化',
        collapsed: false,
        items: [
          { text: '第四部分 量化 · 导言', link: '/04-quantization/' },
          { text: '4.1 数值表示', link: '/04-quantization/01-number-formats' },
          { text: '4.2 量化原理', link: '/04-quantization/02-quantization-principles' },
          { text: '4.3 GGUF K-quants', link: '/04-quantization/03-gguf-k-quants' },
          { text: '4.4 imatrix', link: '/04-quantization/04-imatrix' },
          { text: '4.5 KV Cache 量化', link: '/04-quantization/05-kv-cache-quantization' },
          { text: '4.6 评估量化损失', link: '/04-quantization/06-measuring-loss' },
          { text: '4.7 三条量化路线怎么选', link: '/04-quantization/07-route-selection' },
        ],
      },
      {
        text: '第五部分 RAG 全链路',
        collapsed: false,
        items: [
          { text: '第五部分 RAG 全链路 · 导言', link: '/05-rag/' },
          { text: '5.1 为什么用 RAG', link: '/05-rag/01-why-rag' },
          { text: '5.2 文档解析', link: '/05-rag/02-document-parsing' },
          { text: '5.3 切分', link: '/05-rag/03-chunking' },
          { text: '5.4 Embedding', link: '/05-rag/04-embedding' },
          { text: '5.5 向量库与 ANN 索引', link: '/05-rag/05-vector-index' },
          { text: '5.6 稀疏检索与混合检索', link: '/05-rag/06-hybrid-search' },
          { text: '5.7 重排', link: '/05-rag/07-reranking' },
          { text: '5.8 生成', link: '/05-rag/08-generation' },
          { text: '5.9 元数据、权限过滤与增量更新', link: '/05-rag/09-metadata-and-updates' },
        ],
      },
      {
        text: '第六部分 评估与调优',
        collapsed: false,
        items: [
          { text: '第六部分 评估与调优 · 导言', link: '/06-evaluation/' },
          { text: '6.1 构建测试集', link: '/06-evaluation/01-building-test-set' },
          { text: '6.2 检索指标', link: '/06-evaluation/02-retrieval-metrics' },
          { text: '6.3 生成指标', link: '/06-evaluation/03-generation-metrics' },
          { text: '6.4 消融实验', link: '/06-evaluation/04-ablation' },
          { text: '6.5 错误回答归因', link: '/06-evaluation/05-error-attribution' },
        ],
      },
      {
        text: '第七部分 GPU 服务器与推理加速',
        collapsed: false,
        items: [
          { text: '第七部分 GPU 服务器与推理加速 · 导言', link: '/07-gpu-inference/' },
          { text: '7.1 GPU 硬件常识', link: '/07-gpu-inference/01-gpu-hardware' },
          { text: '7.2 软件栈', link: '/07-gpu-inference/02-software-stack' },
          { text: '7.3 监控与诊断', link: '/07-gpu-inference/03-monitoring' },
          { text: '7.4 vLLM 部署', link: '/07-gpu-inference/04-vllm' },
          { text: '7.5 GPU 端量化实操', link: '/07-gpu-inference/05-gpu-quantization' },
          { text: '7.6 推理加速', link: '/07-gpu-inference/06-inference-acceleration' },
          { text: '7.7 多卡', link: '/07-gpu-inference/07-multi-gpu' },
          { text: '7.8 卡顿与 OOM 排查', link: '/07-gpu-inference/08-oom-and-latency' },
          { text: '7.9 容量规划', link: '/07-gpu-inference/09-capacity-planning' },
        ],
      },
      {
        text: '第八部分 工程化与交付',
        collapsed: false,
        items: [
          { text: '第八部分 工程化与交付 · 导言', link: '/08-delivery/' },
          { text: '8.1 服务架构与流式输出', link: '/08-delivery/01-service-architecture' },
          { text: '8.2 离线部署与断网验证', link: '/08-delivery/02-offline-deployment' },
          { text: '8.3 压测与容量验收', link: '/08-delivery/03-load-testing' },
          { text: '8.4 可观测性', link: '/08-delivery/04-observability' },
          { text: '8.5 数据安全与权限', link: '/08-delivery/05-security-and-permissions' },
          { text: '8.6 从 Mac 迁移到 GPU 服务器', link: '/08-delivery/06-mac-to-gpu' },
          { text: '8.7 交付物', link: '/08-delivery/07-deliverables' },
        ],
      },
      {
        text: '附录',
        collapsed: false,
        items: [
          { text: '附录 · 导言', link: '/appendix/' },
          { text: 'A 显存速查表', link: '/appendix/a-vram-cheatsheet' },
          { text: 'B 术语表', link: '/appendix/b-glossary' },
          { text: 'C 故障排查手册', link: '/appendix/c-troubleshooting' },
          { text: 'D 硬件选型参考', link: '/appendix/d-hardware-reference' },
          { text: 'E 面试题索引', link: '/appendix/e-interview-index' },
          { text: 'F 云 GPU 实验手册', link: '/appendix/f-cloud-gpu-manual' },
          { text: 'G 参考文献', link: '/appendix/g-references' },
        ],
      },
    ],

    outline: { label: '本页目录', level: [2, 3] },
    docFooter: { prev: '上一节', next: '下一节' },
    lastUpdated: { text: '最后更新' },
    returnToTopLabel: '回到顶部',
    sidebarMenuLabel: '目录',
    darkModeSwitchLabel: '主题',
    langMenuLabel: '语言',
    externalLinkIcon: true,

    search: {
      provider: 'local',
      options: {
        translations: {
          button: { buttonText: '搜索', buttonAriaLabel: '搜索' },
          modal: {
            noResultsText: '没有找到结果',
            resetButtonTitle: '清空',
            footer: { selectText: '选择', navigateText: '切换', closeText: '关闭' },
          },
        },
      },
    },
  },
})
