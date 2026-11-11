"""知识库 Agent：先检索、再回答、必须给出处。

没有检索的话，模型会凭记忆瞎编运维步骤；有了检索，它只能基于手册里的话回答。
"""
import rag

SYSTEM = """你是公司内部的运维知识库助手。
规则：
1. 只根据【参考资料】回答，不要用你自己的通用知识补充。
2. 每一条结论后面用 [文档名·小节] 标出来源。
3. 如果参考资料里没有相关内容，说「本次检索到的资料不足」，不能声称整个手册或知识库没有这一条。仅在故障场景与前提成立时给对应处置建议，不能把能回滚当作应该回滚。
4. 回答分三步：现象判断 → 排查顺序 → 处置动作。用中文，不要长篇大论。"""


def build_prompt(question, items):
    blocks = []
    for i, it in enumerate(items, 1):
        blocks.append(f"[{i}] {it['doc']}·{it['section']}\n{it['text']}")
    refs = "\n\n".join(blocks) or "（没有检索到相关内容）"
    return f"【参考资料】\n{refs}\n\n【问题】\n{question}"


def run(question, llm, k=3):
    hit = rag.get().retrieve(question, k)
    if not hit["items"]:
        return {
            "answer": "手册里没有检索到相关内容。",
            "sources": [],
            "retrieval": hit,
            "cost": 0.0,
            "latency": 0.0,
        }
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": build_prompt(question, hit["items"])},
    ]
    answer, cost, latency = llm.call_model(messages, mode="deep", temperature=0.2, max_tokens=900)
    return {
        "answer": answer,
        "sources": [
            {"doc": it["doc"], "section": it["section"], "score": it["score"]}
            for it in hit["items"]
        ],
        "retrieval": hit,
        "cost": cost,
        "latency": latency,
    }
