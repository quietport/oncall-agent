"""FastAPI 入口：三个 Agent 各一个接口。

    POST /kb/ask        知识库 Agent（RAG，带引用）
    POST /chat/ask      对话 Agent（多轮记忆）
    GET  /chat/stream   对话 Agent（SSE 流式）
    POST /triage        运维 Agent（Plan-Execute-Replan）
    GET  /stats         今日成本 / 预算 / 缓存

本地跑：
    python -m uvicorn app:app --reload --port 8001
"""
import json
import os
import queue
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

import llm
import rag
import tools
from agents import chat, kb, triage

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="智能 OnCall Agent", version="0.1")


class KBIn(BaseModel):
    question: str = Field(..., min_length=2, max_length=2000)


class ChatIn(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    session_id: str = Field("default", max_length=64)


class TriageIn(BaseModel):
    alert: str = Field(..., min_length=4, max_length=2000)


def _guard():
    try:
        llm.ensure_budget()
    except llm.BudgetExceeded as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/")
def index():
    return FileResponse(BASE_DIR / "index.html", media_type="text/html; charset=utf-8")


@app.get("/health")
def health():
    return {"ok": True, "docs_indexed": len(rag.get().chunks), "retrieval_mode": rag.get().mode}


@app.post("/kb/ask")
def kb_ask(body: KBIn):
    _guard()
    try:
        return kb.run(body.question, llm)
    except llm.ModelError as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.post("/chat/ask")
def chat_ask(body: ChatIn):
    _guard()
    try:
        return chat.ask(body.session_id, body.question, llm)
    except llm.ModelError as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.get("/chat/stream")
def chat_stream(session_id: str = "default", question: str = ""):
    if len(question.strip()) < 1:
        raise HTTPException(status_code=422, detail="question 不能为空")
    _guard()

    def gen():
        try:
            for kind, payload in chat.stream(session_id, question, llm):
                if kind == "delta":
                    yield f"data: {json.dumps({'stage': 'delta', 'text': payload}, ensure_ascii=False)}\n\n"
                elif kind == "stage":
                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                else:
                    yield f"data: {json.dumps({'stage': 'done', 'payload': payload}, ensure_ascii=False)}\n\n"
        except llm.ModelError as e:
            yield f"data: {json.dumps({'stage': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"
        except Exception as e:  # noqa: BLE001
            yield f"data: {json.dumps({'stage': 'error', 'message': f'内部错误：{e}'}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/triage/stream")
def triage_stream(alert: str = ""):
    """排障的流式版本：把「生成计划 / 每一步工具结果 / 重规划 / 下结论」逐个推给前端。

    同步的 triage.run() 跑在后台线程里，通过 queue 把事件交给 SSE 生成器。
    这样前端不用盯着转圈等 12 秒，能看到它每一步在干什么。
    """
    if len(alert.strip()) < 4:
        raise HTTPException(status_code=422, detail="alert 太短")
    _guard()

    def gen():
        q: "queue.Queue[dict]" = queue.Queue()

        def work():
            try:
                res = triage.run(alert, llm, emit=lambda ev: q.put(ev))
                q.put({"stage": "done", "payload": res})
            except llm.BudgetExceeded as e:
                q.put({"stage": "error", "message": str(e)})
            except llm.ModelError as e:
                q.put({"stage": "error", "message": str(e)})
            except Exception as e:  # noqa: BLE001
                q.put({"stage": "error", "message": f"内部错误：{e}"})

        threading.Thread(target=work, daemon=True).start()
        while True:
            ev = q.get()
            yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
            if ev.get("stage") in ("done", "error"):
                break
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/triage")
def triage_alert(body: TriageIn):
    _guard()
    try:
        return triage.run(body.alert, llm)
    except llm.ModelError as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.get("/stats")
def stats():
    return {
        **llm.stats(),
        "retrieval": {"mode": rag.get().mode, "chunks": len(rag.get().chunks), "note": rag.get().note},
    }


@app.get("/logs")
def list_logs(service: str = "", keyword: str = ""):
    return tools.search_logs(keyword=keyword, service=service or None)


@app.get("/metrics")
def metrics(service: str = ""):
    return tools.get_metrics(service or None)


@app.get("/changes")
def changes():
    return tools.get_recent_changes()


@app.exception_handler(HTTPException)
def http_error(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
