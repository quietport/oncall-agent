"""调用层：一个函数管完「调模型 + 算钱 + 缓存 + 限流 + 预算」。

上层（agents/）不需要知道 HTTP、重试、计费细节，只管拿文本和成本。
"""
import hashlib
import json
import os
import time
import threading
from functools import wraps

import requests

URL = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-flash"

try:  # 本地密钥文件，不进仓库
    from secret import API_KEY
except ImportError:
    API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")

PRICE_IN = 2.0 / 1_000_000
PRICE_OUT = 8.0 / 1_000_000
THINKING_OFF = {"type": "disabled"}

DAILY_BUDGET = float(os.environ.get("DAILY_BUDGET", "1.0"))
_SPEND = {}
_CALL_LOCK = threading.RLock()
_CACHE = {}
CACHE_TTL = 1800
CACHE_MAX = 256


class BudgetExceeded(Exception):
    pass


class ModelError(Exception):
    pass


def _today():
    return time.strftime("%Y-%m-%d")


def today_cost():
    return round(_SPEND.get(_today(), 0.0), 6)


def budget_left():
    return round(DAILY_BUDGET - today_cost(), 6)


def ensure_budget():
    if today_cost() >= DAILY_BUDGET:
        raise BudgetExceeded(f"今日预算已用完（上限 {DAILY_BUDGET} 元）")


def charge(cost):
    d = _today()
    _SPEND[d] = _SPEND.get(d, 0.0) + cost


def cache_key(messages, mode, temperature, max_tokens=1200, response_format=None):
    raw = json.dumps({'messages':messages,'model':MODEL,'url':URL,
                     'max_tokens':max_tokens,'response_format':response_format}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(f"{mode}|{temperature}|{raw}".encode()).hexdigest()


def cache_get(key):
    item = _CACHE.get(key)
    if not item:
        return None
    if time.time() - item["ts"] > CACHE_TTL:
        _CACHE.pop(key, None)
        return None
    return item["value"]


def cache_put(key, value):
    if len(_CACHE) > CACHE_MAX:
        for k in sorted(_CACHE, key=lambda x: _CACHE[x]["ts"])[:64]:
            _CACHE.pop(k, None)
    _CACHE[key] = {"ts": time.time(), "value": value}


def cache_size():
    return len(_CACHE)


def _body(messages, mode, temperature, max_tokens, response_format, stream=False, tools=None):
    body = {
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if mode == "fast":
        body["thinking"] = THINKING_OFF
    if response_format:
        body["response_format"] = response_format
    if tools:
        body["tools"] = tools
    if stream:
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
    return body


def _post(body, stream=False):
    if not API_KEY:
        raise ModelError("没有配置 API key（secret.py 或 DEEPSEEK_API_KEY）")
    try:
        return requests.post(
            URL,
            headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
            json=body,
            timeout=90,
            stream=stream,
        )
    except requests.RequestException as e:
        raise ModelError(f"网络错误：{e}") from e


def _usage_cost(usage):
    pin = usage.get("prompt_tokens", 0) if usage else 0
    pout = usage.get("completion_tokens", 0) if usage else 0
    return pin * PRICE_IN + pout * PRICE_OUT


def _budgeted(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        with _CALL_LOCK:
            ensure_budget()
            return fn(*args, **kwargs)
    return wrapped


def _stream_budgeted(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        with _CALL_LOCK:
            ensure_budget()
            yield from fn(*args, **kwargs)
    return wrapped


@_budgeted
def call_model(messages, mode="deep", temperature=0.2, max_tokens=1200, response_format=None):
    """返回 (文本, 成本, 耗时秒)。命中缓存时成本 0、耗时 0。"""
    key = cache_key(messages, mode, temperature, max_tokens, response_format)
    hit = cache_get(key)
    if hit:
        return hit, 0.0, 0.0

    t0 = time.time()
    r = _post(_body(messages, mode, temperature, max_tokens, response_format))
    latency = time.time() - t0
    if r.status_code != 200:
        raise ModelError(f"接口返回 {r.status_code}：{r.text[:200]}")
    data = r.json()
    text = data["choices"][0]["message"]["content"]
    cost = _usage_cost(data.get("usage"))
    charge(cost)
    cache_put(key, text)
    return text, round(cost, 6), round(latency, 2)


@_stream_budgeted
def stream_model(messages, mode="deep", temperature=0.2, max_tokens=1200):
    """逐块吐出文本。最后 yield ("end", 全文, 成本, 耗时)。"""
    t0 = time.time()
    r = _post(_body(messages, mode, temperature, max_tokens, None, stream=True), stream=True)
    if r.status_code != 200:
        raise ModelError(f"接口返回 {r.status_code}：{r.text[:200]}")

    chunks, usage = [], None
    for line in r.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        payload = line[6:]
        if payload.strip() == "[DONE]":
            break
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if obj.get("usage"):
            usage = obj["usage"]
        for choice in obj.get("choices", []):
            piece = (choice.get("delta") or {}).get("content")
            if piece:
                chunks.append(piece)
                yield ("delta", piece)
    full = "".join(chunks)
    cost = _usage_cost(usage)
    charge(cost)
    yield ("end", full, round(cost, 6), round(time.time() - t0, 2))


@_budgeted
def chat_with_tools(messages, tools, mode="fast", temperature=0.2, max_tokens=600):
    """带工具的一次调用，返回 (message 字典, 成本, 耗时)。

    message 里可能有 tool_calls，也可能只有 content。
    带工具的请求不缓存——工具调用的中间状态缓存没有意义。
    """
    t0 = time.time()
    r = _post(_body(messages, mode, temperature, max_tokens, None, tools=tools))
    latency = time.time() - t0
    if r.status_code != 200:
        raise ModelError(f"接口返回 {r.status_code}：{r.text[:200]}")
    data = r.json()
    msg = data["choices"][0]["message"]
    cost = _usage_cost(data.get("usage"))
    charge(cost)
    return msg, round(cost, 6), round(latency, 2)


def clean_json(text):
    """模型有时会包一层 ```json ... ```，剥掉再解析。"""
    s = (text or "").strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[-1]
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3]
    start, end = s.find("{"), s.rfind("}")
    if start >= 0 and end > start:
        s = s[start : end + 1]
    return s


def parse_json(text, retry_messages=None, mode="deep", temperature=0.2, max_tokens=1200):
    """解析 JSON；失败时（可选）重试一次，返回 (对象, 成本, 耗时)。"""
    cost = latency = 0.0
    try:
        return json.loads(clean_json(text)), cost, latency
    except json.JSONDecodeError:
        if not retry_messages:
            raise ModelError("模型没有返回合法 JSON")
    text2, c2, l2 = call_model(
        retry_messages,
        mode=mode,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    try:
        return json.loads(clean_json(text2)), round(cost + c2, 6), round(latency + l2, 2)
    except json.JSONDecodeError as e:
        raise ModelError("模型两次都没有返回合法 JSON") from e


def stats():
    return {
        "model": MODEL,
        "cache_size": cache_size(),
        "today_cost": today_cost(),
        "daily_budget": DAILY_BUDGET,
        "budget_left": budget_left(),
    }
