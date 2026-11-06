# benchmarks/delivery

第八部分的运行数据：

- `feedback.jsonl`：8.1 服务的用户反馈原文；可能含用户数据，已 gitignore，只提交汇总。
- 8.3 的压测/验收 CSV。
- 8.2 的离线验证摘要（命令、时间、结果）；完整 `verify.log` 留在 bundle 里，不重复入库。

数据文件应带 `env.txt`（硬件、系统、软件版本），口径同其他 `benchmarks/` 目录。
