# 8.7 交付包模板与自检

交付包不是“把代码打个 zip”，而是让客户能**安装、使用、运维、审计、培训**的一整套东西。

## 模板

`templates/` 下 8 份核心文档：

| 文件 | 用途 |
|---|---|
| `00-README.md` | 交付包门面：版本、范围、联系人、目录 |
| `01-architecture.md` | 架构、数据流、信任边界、版本、SLO、限制 |
| `02-deploy.md` | 前置条件、安装、配置、验证、回滚 |
| `03-operations.md` | 启停、健康检查、日志/追踪、告警、备份、升级、故障 |
| `04-acceptance.md` | 验收标准、测试用例、证据、未决问题、签字 |
| `05-training.md` | 管理员/操作员/用户培训、议程、练习、考核 |
| `06-ai-usage-policy.md` | 数据分类、允许/禁止、引用核验、事件上报 |
| `07-licenses.md` | 模型/语料/软件许可证与合规检查 |

模板里的 `【...】`、`<...>`、`待填` 都是**必须替换**的占位符；不要带着它们交付。

## 自检

```bash
# 模板结构自检（允许占位符）
python3 labs/08-delivery/delivery/check_delivery.py --allow-placeholders --allow-missing-manifest

# 真实交付包自检（占位符会失败）
python3 labs/08-delivery/delivery/check_delivery.py --dir dist/delivery-<客户>-<版本>

# 机器可读
python3 labs/08-delivery/delivery/check_delivery.py --dir ... --json
```

自检只保证“文件齐、标题/表格在、没有漏填/明显密钥、MANIFEST 覆盖并匹配包内文件”，不保证内容正确；内容正确性靠评审和验收。

## 交付前最后三件事

1. **换人重装**：找一个没参与开发的人，只给交付包，让他在目标环境装一遍；卡住的地方就是文档缺口。
2. **跑一遍验收**：至少覆盖已知问题、拒答、权限、离线、备份恢复。
3. **签字**：客户、交付、安全三方签字；未决问题列清楚，不要口头承诺。
