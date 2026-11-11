"""工具层：Agent 能调用的四个查询工具 + 一个结束工具。

全部读 data/ 下的本地数据，没有外部依赖 —— 真实环境里换成日志平台 / 监控 / 发布系统的接口即可，
Agent 那一侧的代码一行都不用改，这就是「工具层」的意义。
"""
import inspect
import json
import re
from pathlib import Path

import rag

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"


def _load(name):
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def search_logs(keyword="", service=None, limit=12):
    """按关键字搜日志。关键字可以用空格或逗号分隔多个，命中任意一个就算（OR）。"""
    rows = _load("logs.json")
    kws = [k.strip().lower() for k in re.split(r"[\s,，]+", keyword or "") if k.strip()]
    hits = []
    for row in rows:
        if service and row["service"] != service:
            continue
        msg = row["msg"].lower()
        if kws and not any(k in msg for k in kws):
            continue
        hits.append(row)
    hits.sort(key=lambda r: r["ts"], reverse=True)
    return hits[:limit]


def get_metrics(service=None):
    """看监控指标。不传 service 返回全部服务。"""
    data = _load("metrics.json")
    if service:
        m = data["services"].get(service)
        return {"service": service, "metrics": m} if m else {"error": f"没有服务 {service} 的指标"}
    return data


def get_recent_changes(minutes=180):
    """看最近的发布 / 配置变更记录。"""
    changes = _load("changes.json")
    changes.sort(key=lambda c: c["ts"], reverse=True)
    return {"window_minutes": minutes, "changes": changes}


def search_docs(query, k=3):
    """查运维手册（RAG 检索）。"""
    return rag.get().retrieve(query, k)


FUNCS = {
    "search_logs": search_logs,
    "get_metrics": get_metrics,
    "get_recent_changes": get_recent_changes,
    "search_docs": search_docs,
}

# 给模型看的「工具说明书」：描述写得好不好，直接决定它用得对不对
SPEC = [
    {
        "type": "function",
        "function": {
            "name": "get_metrics",
            "description": "查看某个服务的监控指标：P99 延迟、QPS、错误率、CPU、数据库连接数、缓存命中率、磁盘使用率、句柄数。定位问题的第一步通常是看这里。",
            "parameters": {
                "type": "object",
                "properties": {
                    "service": {"type": "string", "description": "服务名，例如 pay-service、order-service；不传则返回全部服务"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_logs",
            "description": "在日志里搜关键字，返回最近的匹配行。适合搜 error 关键词，如 timeout、Connection、NullPointer、No space、Too many open files、rate limit。",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {"type": "string", "description": "要搜的关键字，不区分大小写"},
                    "service": {"type": "string", "description": "限定某个服务，可不填"},
                },
                "required": ["keyword"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_recent_changes",
            "description": "查看最近的代码发布和配置变更记录。故障时间点附近有发版时，优先怀疑发版。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": "检索内部运维手册（故障预案）。当你需要「这类问题的标准排查步骤和处置动作」时调用。",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "用自然语言描述你要查的问题"}},
                "required": ["query"],
            },
        },
    },
]


def run(name, args):
    """按名字执行工具。

    两个韧性设计：
    1. 参数名写错时，自动丢掉多余的参数而不是直接报错（模型很爱编参数名）。
    2. 任何异常都变成可读的结果 —— Agent 拿到「这一步失败了」也要能继续往下走。
    """
    fn = FUNCS.get(name)
    if not fn:
        return {"error": f"没有名为 {name} 的工具"}

    if not isinstance(args, dict):
        return {"error":"工具参数必须是合法 JSON 对象，请修正后重试。"}
    args = dict(args)
    dropped = []
    sig = inspect.signature(fn)
    accepts = {
        p.name
        for p in sig.parameters.values()
        if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
    }
    if not any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
        dropped = [k for k in args if k not in accepts]
        if dropped:
            return {"error":"存在未允许的参数："+', '.join(dropped)}
    for key in ('service','query','keyword'):
        if key in args and (not isinstance(args[key],str) or (key!='keyword' and not args[key].strip())):
            return {"error":key+' 必须是有效文本。'}
    for key in ('limit','k','minutes'):
        if key in args and (type(args[key]) is not int or not 1<=args[key]<=1000):
            return {"error":key+' 必须是 1 至 1000 的整数。'}

    try:
        out = fn(**args)
    except TypeError as e:
        return {"error": f"参数不对：{e}"}
    except Exception as e:  # noqa: BLE001
        return {"error": f"工具执行失败：{e}"}

    if dropped and isinstance(out, dict):
        out = {**out, "_dropped_args": dropped}
    return out
