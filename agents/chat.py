"""对话 Agent：多轮记忆 + 真正调用工具 + 真流式输出。

和知识库 Agent 的区别：知识库是「你先给我问题，我一定查」；
对话是「模型自己判断要不要查、查哪个」。

这里的两个实现细节值得讲：
1. 工具轮用非流式（要拿到完整 tool_calls 才能执行），最后一段答案才用流式；
2. 流式是模型真的边生成边吐字，不是先生成完再切片假装流式。
"""
import json

import tools

SYSTEM = """你是公司内部的值班助手，回答员工关于系统故障、运维操作的问题。
你有工具可以查日志、查监控、查发布记录、查运维手册。

规则：
1. 不确定的事实先调工具，不要凭印象回答。
2. 回答简短，先给结论再给依据。
3. 如果问题涉及线上正在发生的故障，先查监控再答。
4. 查不到就说查不到，不要编。"""

MEMORY_TURNS = 6
_MAX_SESSIONS = 200
_memory = {}


def history(session_id):
    return list(_memory.get(session_id, []))


def _remember(session_id, role, content):
    arr = _memory.setdefault(session_id, [])
    arr.append({"role": role, "content": content})
    keep = MEMORY_TURNS * 2
    if len(arr) > keep:
        _memory[session_id] = arr[-keep:]
    # 会话太多时清掉最旧的，避免长跑之后内存无限涨
    if len(_memory) > _MAX_SESSIONS:
        for k in list(_memory)[: len(_memory) - _MAX_SESSIONS]:
            _memory.pop(k, None)


def build_messages(session_id, question):
    msgs = [{"role": "system", "content": SYSTEM}]
    msgs.extend(history(session_id))
    msgs.append({"role": "user", "content": question})
    return msgs


def _parse_args(call):
    raw = (call.get("function") or {}).get("arguments") or "{}"
    try:
        args = json.loads(raw)
        return args if isinstance(args, dict) else None
    except json.JSONDecodeError:
        return None


def _tool_round(messages, llm, trace, max_tools=3):
    """问模型：这轮要不要调工具。要调就执行并把结果回填。返回 (是否调了工具, 成本, 耗时)。"""
    msg, cost, latency = llm.chat_with_tools(
        messages, tools.SPEC, mode="fast", temperature=0.2, max_tokens=600
    )
    calls = msg.get("tool_calls") or []
    if not calls:
        # This draft has not been shown to the user. Do not append it as an
        # already-delivered assistant answer and accidentally request a continuation.
        return False, cost, latency

    messages.append(
        {
            "role": "assistant",
            "content": msg.get("content") or "",
            "tool_calls": calls[:max_tools],
        }
    )
    memo = {}
    for call in calls[:max_tools]:
        name = (call.get("function") or {}).get("name", "")
        args = _parse_args(call)
        sig = (name, json.dumps(args, ensure_ascii=False, sort_keys=True))
        if sig in memo:  # 模型偶尔会重复调同一个工具，同轮内复用结果，不再花一次钱
            out = memo[sig]
        else:
            try:
                out = tools.run(name, args)
            except Exception as e:  # noqa: BLE001
                out = {"error": f"工具执行失败：{e}"}
            memo[sig] = out
        trace.append({"tool": name, "args": args, "ok": not (isinstance(out, dict) and out.get("error"))})
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call.get("id", ""),
                "content": json.dumps(out, ensure_ascii=False)[:4000],
            }
        )
    return True, cost, latency


def ask(session_id, question, llm, max_rounds=2):
    """非流式版本：跑工具轮，再一次性取答案。"""
    messages = build_messages(session_id, question)
    cost = latency = 0.0
    trace = []
    for _ in range(max_rounds):
        used, c, l = _tool_round(messages, llm, trace)
        cost += c
        latency += l
        if not used:
            break

    text, c, l = llm.call_model(messages, mode="fast", temperature=0.2, max_tokens=600)
    cost += c
    latency += l
    _remember(session_id, "user", question)
    _remember(session_id, "assistant", text)
    return {"answer": text, "trace": trace, "cost": round(cost, 6), "latency": round(latency, 2)}


def stream(session_id, question, llm, max_rounds=2):
    """流式版本。

    yield 的事件：
        ("stage", {"stage": "tool", "name": ..., "args": ..., "ok": ...})
        ("delta", 文本片段)          ← 模型真的边生成边吐
        ("end",   {"answer", "cost", "latency", "trace"})
    """
    messages = build_messages(session_id, question)
    cost = latency = 0.0
    trace = []

    for _ in range(max_rounds):
        before = len(trace)
        used, c, l = _tool_round(messages, llm, trace)
        cost += c
        latency += l
        if used:
            for item in trace[before:]:
                yield (
                    "stage",
                    {
                        "stage": "tool",
                        "name": item.get("tool", ""),
                        "args": item.get("args", {}),
                        "ok": item.get("ok", True),
                    },
                )
            continue
        break

    full = []
    for ev in llm.stream_model(messages, mode="fast", temperature=0.2, max_tokens=600):
        if ev[0] == "delta":
            full.append(ev[1])
            yield ("delta", ev[1])
        else:
            _, text, c, l = ev
            cost += c
            latency += l
    answer = "".join(full)

    _remember(session_id, "user", question)
    _remember(session_id, "assistant", answer)
    yield (
        "end",
        {"answer": answer, "cost": round(cost, 6), "latency": round(latency, 2), "trace": trace},
    )
