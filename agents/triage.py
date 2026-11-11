"""运维 Agent：Plan → Execute → Replan → 结论。

三个模式的差别就在这里：
- ReAct：边想边做，走一步看一步
- Plan-Execute-Replan：先把计划列出来，执行中发现计划不对再重规划
故障排查天然是「有先后依赖的步骤」，所以用后者。
"""
import json
import re
import math

import llm as llm_mod
import tools

PLAN_SYSTEM = """你是资深 SRE。给你一条告警，你要先列排查计划，再逐步执行。
只输出 JSON，格式：
{
  "hypothesis": "一句话，你最怀疑的原因",
  "steps": [
    {"tool": "工具名", "args": {"参数": "值"}, "why": "这一步想验证什么"}
  ]
}
可用工具（参数名必须严格照抄，多传的参数会被自动丢弃）：
- get_metrics(service)            // service 可选，例如 "pay-service"；不传返回全部服务
- search_logs(keyword, service)   // keyword 必填，例如 "timeout"；service 可选
- get_recent_changes()            // 没有参数
- search_docs(query)              // query 必填，自然语言描述

禁止自创参数名（如 metrics / window / level / group_by），它们会被丢弃。
要求：3~5 步；每步只查一件事；先看指标和变更，再搜日志，最后查手册。"""

PLAN_RETRY_SYSTEM = """你是资深 SRE。上一轮排查没有拿到有效证据，请重新规划。
只输出 JSON，格式同上。这一轮请换不同的工具或不同的关键字，不要重复上一轮。"""

CONCLUDE_SYSTEM = """你是资深 SRE。根据下面的证据给出结论。
只输出 JSON，格式：
{
  "severity": "P0|P1|P2",
  "root_cause": "根因，一句话",
  "evidence_summary": ["证据1", "证据2"],
  "actions": ["处置动作1", "处置动作2"],
  "confidence": 0.0
}
要求：
1. confidence 是 0~1 的小数，表示「这个根因有多确信」。
2. 根因必须能被证据直接支持，不能出现证据里没有的东西。
3. 如果告警信息本身就不足（没有具体现象、服务名、时间），或者证据与告警说的不是同一个服务 /
   同一类故障，你必须：confidence 压到 0.3 以下，并且 root_cause 以「证据不足：」开头，
   后面写清「还缺什么证据」。宁可说不知道，也不要编一个像样的根因。"""


def _json_call(messages, llm, max_tokens=1400):
    """调一次模型并要求 JSON；失败就带着「你上次的 JSON 不合法」重试。

    模型（尤其开了思考模式的）偶尔会输出被截断的 JSON——`Unterminated string` 就是典型。
    所以这里做两件事：
    1) 重试时把 max_tokens 放大一倍（截断的本质是没写够），而不是原样再来一次；
    2) 两次都失败才抛错，交给上层决定是降级还是报错。
    """
    text, cost, latency = llm.call_model(
        messages,
        mode="deep",
        temperature=0.2,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    try:
        return json.loads(llm_mod.clean_json(text)), cost, latency
    except json.JSONDecodeError:
        bigger = min(max_tokens * 2, 3000)
        retry = messages + [
            {"role": "assistant", "content": text},
            {
                "role": "user",
                "content": (
                    "上面的 JSON 不合法（可能被截断或引号没转义）。"
                    "请只输出一个合法的 JSON 对象，不要任何解释文字；字段值尽量简短。"
                ),
            },
        ]
        text2, c2, l2 = llm.call_model(
            retry,
            mode="deep",
            temperature=0.1,
            max_tokens=bigger,
            response_format={"type": "json_object"},
        )
        try:
            return (
                json.loads(llm_mod.clean_json(text2)),
                round(cost + c2, 6),
                round(latency + l2, 2),
            )
        except json.JSONDecodeError as e:
            raise llm_mod.ModelError(f"模型两次都没有返回合法 JSON：{e}") from e


def plan(alert, llm, retry_note=""):
    system = PLAN_RETRY_SYSTEM if retry_note else PLAN_SYSTEM
    user = f"告警：{alert}\n{retry_note}"
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    result,cost,latency = _json_call(messages, llm, max_tokens=1400)
    if (not isinstance(result,dict) or not isinstance(result.get('hypothesis'),str)
            or not isinstance(result.get('steps'),list) or not 1<=len(result['steps'])<=5):
        raise llm_mod.ModelError('模型计划结构无效：需要 hypothesis 和 1 至 5 个步骤。')
    for step in result['steps']:
        if (not isinstance(step,dict) or not isinstance(step.get('tool'),str)
                or not isinstance(step.get('args'),dict) or not isinstance(step.get('why'),str)):
            raise llm_mod.ModelError('模型步骤结构无效：需要 tool、args 和 why。')
    return result,cost,latency


def execute(steps, limit=5, emit=None, round_no=1):
    evidence, used = [], []
    for i, step in enumerate(steps[:limit], 1):
        name = step.get("tool", "")
        args = step.get("args", {}) if isinstance(step.get("args"), dict) else {}
        out = tools.run(name, args)
        ok = "error" not in (out if isinstance(out, dict) else {})
        used.append({"tool": name, "args": args, "why": step.get("why", ""), "ok": ok})
        evidence.append({"tool": name, "args": args, "why": step.get("why", ""), "output": out})
        if emit:
            emit(
                {
                    "stage": "tool",
                    "round": round_no,
                    "index": i,
                    "total": min(len(steps), limit),
                    "tool": name,
                    "args": args,
                    "why": step.get("why", ""),
                    "ok": ok,
                    "preview": str(out)[:300],
                }
            )
    return evidence, used


def _useful(evidence):
    """证据里至少有一条不是空 / 不是 error，才算这轮排查有收获。"""
    for e in evidence:
        out = e.get("output") or {}
        if isinstance(out, dict) and out.get("error"):
            continue
        if isinstance(out, list) and not out:
            continue
        if isinstance(out, dict) and not out:
            continue
        return True
    return False


def conclude(alert, evidence, llm):
    payload = json.dumps(evidence, ensure_ascii=False)[:9000]
    messages = [
        {"role": "system", "content": CONCLUDE_SYSTEM},
        {"role": "user", "content": f"告警：{alert}\n\n证据（JSON）：\n{payload}"},
    ]
    try:
        result,cost,latency = _json_call(messages, llm, max_tokens=1200)
        if (not isinstance(result,dict) or result.get('severity') not in ('P0','P1','P2')
                or not isinstance(result.get('root_cause'),str)
                or type(result.get('confidence')) not in (int,float)
                or not math.isfinite(result['confidence']) or not 0<=result['confidence']<=1
                or any(not isinstance(result.get(k),list) or any(not isinstance(x,str) for x in result[k])
                       for k in ('evidence_summary','actions'))):
            raise llm_mod.ModelError('模型结论字段或类型无效。')
        return result,cost,latency
    except llm_mod.ModelError:
        # 降级：宁给一条「我解读失败」的结论，也不要让整个接口 500
        return (
            {
                "severity": "P2",
                "root_cause": "证据不足：模型返回的结论不是合法 JSON（多半被截断），已降级处理，建议人工复核证据。",
                "evidence_summary": [str(e.get("output", ""))[:120] for e in evidence][:3],
                "actions": ["补充具体服务、实例、时间窗口与异常指标，交由人工核实。"],
                "confidence": 0.2,
                "degraded": True,
            },
            0.0,
            0.0,
        )


SERVICE_RE = re.compile(r"[a-z][a-z0-9-]*-service\b|\bgateway\b")
OPS_WORDS = [
    "超时", "延迟", "错误", "失败", "异常", "磁盘", "连接", "句柄", "缓存", "熔断",
    "限流", "发版", "告警", "日志", "内存", "数据库", "CPU", "QPS", "P0", "P1", "P2", "P3",
]


def _alert_services(alert):
    found=set(SERVICE_RE.findall(alert.lower()))
    if '网关' in alert:found.add('gateway')
    return found


def _evidence_services(evidence):
    found=set()
    def collect(value):
        if isinstance(value,dict):
            if value.get('error'):return
            if isinstance(value.get('service'),str):found.update(SERVICE_RE.findall(value['service'].lower()))
            if isinstance(value.get('services'),dict):
                found.update(k for k,v in value['services'].items() if isinstance(v,dict) and v)
            for key,value in value.items():
                if key not in ('args','why','error','text','note','msg'):collect(value)
        elif isinstance(value,list):
            for item in value:collect(item)
    for item in evidence:collect(item.get('output'))
    return found


def calibrate(alert, evidence, result):
    """确定性的「相关性检查」：告警说的服务，证据里根本没出现，就不该给高置信度。

    模型自己往往意识不到「我查到的这些东西和告警无关」，所以这条用代码兜底，
    不指望 prompt。这是整个项目里最像真实工程的一处：不信任模型的自评。
    """
    alert_svcs = _alert_services(alert)
    ev_svcs = _evidence_services(evidence)
    ops_hits = sum(1 for w in OPS_WORDS if w.lower() in alert.lower())

    reason = ""
    if not _useful(evidence):
        reason = '未取得有效工具证据'
    elif alert_svcs and not alert_svcs.issubset(ev_svcs):
        reason = f"告警提到的服务 {sorted(alert_svcs)} 在证据里没有出现"
    elif not alert_svcs and (ops_hits < 2 or not any(word in alert.lower() for word in
            ('超时','延迟','磁盘','句柄','缓存','熔断','限流','429','504','open files','连接池','内存','cpu','qps'))):
        reason = "告警信息过少，缺少具体服务名和现象"

    uncertain = result.get('confidence',1)<=0.3 or any(word in str(result.get('root_cause','')) for word in ('证据不足','无法确定','无法判断','无法定位','缺少'))
    if not reason and not uncertain:
        return result

    conf = result.get("confidence")
    conf = float(conf) if isinstance(conf, (int, float)) else 0.5
    result["confidence"] = round(min(conf, 0.3), 2)
    rc = str(result.get("root_cause") or "")
    if reason:
        result['root_cause']='证据不足：'+reason+'，请补充信息后再判断根因。'
    elif not rc.startswith("证据不足"):
        result["root_cause"] = f"证据不足：{rc}"
    result['actions']=['补充具体服务与实例、故障时间窗口、异常指标及对应日志，交由人工核实；当前不执行变更操作。']
    result['needs_information']=True
    result["calibration_note"] = '限制为补充信息与人工核实：'+(reason or '模型判断证据不足')
    return result


def run(alert, llm, emit=None):
    """跑一次完整排查。emit 是可选回调，用来把中间过程推给前端（SSE 进度）。

    没有 emit 时行为和以前完全一样，所以 /triage 和 /triage/stream 共用这一份逻辑。
    """
    cost = latency = 0.0
    rounds = []

    def _emit(ev):
        if emit:
            try:
                emit(ev)
            except Exception:  # noqa: BLE001
                pass

    _emit({"stage": "planning", "round": 1, "note": "正在生成排查计划…"})
    p1, c1, l1 = plan(alert, llm)
    cost += c1
    latency += l1
    _emit(
        {
            "stage": "plan",
            "round": 1,
            "hypothesis": p1.get("hypothesis", ""),
            "steps": [s for s in (p1.get("steps") or [])][:5],
        }
    )
    ev1, used1 = execute(p1.get("steps", []), emit=_emit, round_no=1)
    rounds.append({"round": 1, "hypothesis": p1.get("hypothesis", ""), "steps": used1, "replanned": False})

    evidence = ev1
    replanned = False
    if not _useful(ev1):
        note = "上一轮所有工具都返回空结果或报错，请换工具和关键字重新规划。"
        _emit({"stage": "replan", "note": "上一轮没拿到有效证据，正在重新规划…"})
        p2, c2, l2 = plan(alert, llm, retry_note=note)
        cost += c2
        latency += l2
        _emit(
            {
                "stage": "plan",
                "round": 2,
                "hypothesis": p2.get("hypothesis", ""),
                "steps": [s for s in (p2.get("steps") or [])][:5],
            }
        )
        ev2, used2 = execute(p2.get("steps", []), emit=_emit, round_no=2)
        rounds.append({"round":2,"hypothesis":p2.get("hypothesis",""),"steps":used2,
                       "replanned":True,"useful":_useful(ev2)})
        replanned=True
        if _useful(ev2):
            evidence = ev2

    _emit({"stage": "concluding", "note": "正在核对已取得的证据并生成结论…"})
    final, c3, l3 = conclude(alert, evidence, llm)
    cost += c3
    latency += l3
    final = calibrate(alert, evidence, final)

    return {
        "alert": alert,
        "plan_rounds": rounds,
        "replanned": replanned,
        "evidence": [
            {"tool": e["tool"], "args": e["args"], "why": e["why"], "preview": str(e["output"])[:400]}
            for e in evidence
        ],
        "result": final,
        "cost": round(cost, 6),
        "latency": round(latency, 2),
    }
