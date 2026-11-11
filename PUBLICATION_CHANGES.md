# 公开副本变更记录

公开目录从明确白名单重新构建；原项目只读，未改写或删除。

| 文件/范围 | 公开副本处理 |
|---|---|
| `agents/chat.py`、`agents/triage.py`、`agents/kb.py`、`tools.py`、`llm.py`、`rag.py`、`index.html` | 保留当前修复后版本，逐字节一致，不改变业务流程 |
| `app.py`、`agents/__init__.py`、`eval.py`、`eval_holdout.py`、固定评测 JSON、`docs/`、`data/metrics.json`、`requirements.txt` | 白名单复制，内容保持一致 |
| `data/logs.json` | 将模拟日志中的公网形式 IP 改为文档保留地址 `203.0.113.9`，其他模拟事件不变 |
| `data/changes.json` | 将演示提交者姓名替换为“演示人员01”等编号，时间、服务、版本及变更说明不变 |
| `.gitignore` | 扩展忽略本地凭证、环境配置、缓存、运行结果、数据库、截图/原始报告及开发机器配置；显式允许空白示例 |
| `secret.example.py` | 示例值改为空字符串；复制到本地配置后可启动，首次模型请求明确提示缺少凭证 |
| `requirements-dev.txt` | 新增测试依赖入口，复用原运行依赖并补充 TestClient 所需 `httpx`；未升级原依赖声明 |
| `README.md` | 补全初始安装、配置优先级、本机运行、离线回归、历史证据范围与 AI 辅助/原型边界 |
| `acceptance/check_boundaries.py` | 保留 27 项检查断言，改为仓库相对路径；注入空凭证、禁用向量配置、拦截外部 HTTP；不再写原始结果文件，失败返回非零退出码 |
| `acceptance/README.md` | 重新编写去敏历史摘要与本次公开副本离线验证结果，删除指向未发布材料的链接 |

未复制：真实 `secret.py`、任何 `.git` 历史、Python 缓存、运行结果 CSV、真实 API 报告、页面截图、服务器日志/PID、数据库、本机绝对路径、原始代码备份及历史验收工作目录。没有读取或输出原凭证值，没有联网或产生模型调用费用。

公开包不新增法律许可文件。发布者仍需自行决定是否设置开源许可证。
