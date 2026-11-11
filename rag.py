"""检索层：把 docs/ 下的运维手册切成块，找出和问题最相关的几段。

两种实现，自动切换：
- 默认 BM25（纯 Python，零依赖，靠词频相关性）
- 设置环境变量 EMBED_API_KEY 后走向量检索（OpenAI 兼容的 embeddings 接口）

用法：
    python rag.py "支付一直在超时"
"""
import json
import math
import os
import re
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent
DOC_DIR = BASE / "docs"
VEC_FILE = BASE / "docs_vec.json"

EMBED_URL = os.environ.get("EMBED_URL", "https://api.siliconflow.cn/v1/embeddings")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-m3")
EMBED_API_KEY = os.environ.get("EMBED_API_KEY", "")


def tokens(text):
    """英文按词切，中文按单字 + 二元组切。二元组是为了让「超时」「连接池」这类词能命中。"""
    low = text.lower()
    words = re.findall(r"[a-z0-9_.]+", low)
    han = re.findall(r"[\u4e00-\u9fff]", low)
    bigrams = [a + b for a, b in zip(han, han[1:])]
    return words + han + bigrams


def load_chunks():
    """一个 md 文件按 ## 小节切块，每块记下出自哪个文件、哪一节。"""
    chunks = []
    for path in sorted(DOC_DIR.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        title = raw.splitlines()[0].lstrip("# ").strip()
        parts = re.split(r"\n(?=## )", raw)
        if len(parts) == 1:
            parts = [raw]
        for part in parts:
            if not part.strip() or not any(line.strip() and not line.startswith('#') for line in part.splitlines()):
                continue
            head = part.splitlines()[0].lstrip("# ").strip()
            chunks.append({
                "doc": path.name,
                "title": title,
                "section": head,
                "text": part.strip(),
                "tokens": tokens(title+'\n'+part),
            })
    return chunks


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.docs = docs
        self.k1 = k1
        self.b = b
        self.N = max(1, len(docs))
        self.avg = sum(len(d["tokens"]) for d in docs) / self.N
        self.df = Counter()
        for d in docs:
            for t in set(d["tokens"]):
                self.df[t] += 1
        self.idf = {t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for t, n in self.df.items()}

    def _score(self, q_tokens, doc):
        tf = Counter(doc["tokens"])
        length = len(doc["tokens"])
        total = 0.0
        for t in q_tokens:
            f = tf.get(t, 0)
            if not f:
                continue
            total += self.idf.get(t, 0.0) * f * (self.k1 + 1) / (
                f + self.k1 * (1 - self.b + self.b * length / self.avg)
            )
        return total

    def top(self, query, k=3):
        q = tokens(query)
        scored = [(self._score(q, d), d) for d in self.docs]
        scored.sort(key=lambda x: -x[0])
        return scored[:k]


def _embed(texts):
    import requests

    r = requests.post(
        EMBED_URL,
        headers={"Authorization": f"Bearer {EMBED_API_KEY}"},
        json={"model": EMBED_MODEL, "input": texts},
        timeout=30,
    )
    r.raise_for_status()
    return [x["embedding"] for x in r.json()["data"]]


def _cos(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb + 1e-9)


class Retriever:
    def __init__(self):
        self.chunks = load_chunks()
        self.bm25 = BM25(self.chunks)
        self.mode = "bm25"
        self.note = "未配置 EMBED_API_KEY，使用词频检索（BM25）"
        self._vec = None
        if EMBED_API_KEY:
            try:
                self._vec = self._load_vec()
                self.mode = "vector"
                self.note = f"向量检索：{EMBED_MODEL}"
            except Exception as e:  # 向量服务不可用时静默退回词频
                self.note = f"向量检索不可用（{e}），退回词频"

    def _load_vec(self):
        if VEC_FILE.exists():
            data = json.loads(VEC_FILE.read_text(encoding="utf-8"))
            if len(data) == len(self.chunks):
                return data
        vecs = _embed([c["text"] for c in self.chunks])
        VEC_FILE.write_text(json.dumps(vecs), encoding="utf-8")
        return vecs

    def retrieve(self, query, k=3):
        if self.mode == "vector" and self._vec:
            qv = _embed([query])[0]
            scored = [(_cos(qv, v), c) for v, c in zip(self._vec, self.chunks)]
            scored.sort(key=lambda x: -x[0])
        else:
            scored = self.bm25.top(query, k)
        return {
            "mode": self.mode,
            "note": self.note,
            "items": [
                {
                    "doc": c["doc"],
                    "section": c["section"],
                    "score": round(s, 2),
                    "text": c["text"][:600],
                }
                for s, c in scored[:k]
            ],
        }


_inst = None


def get():
    global _inst
    if _inst is None:
        _inst = Retriever()
    return _inst


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "支付一直在超时"
    out = get().retrieve(q, 3)
    print(f"模式：{out['mode']}  （{out['note']}）")
    for it in out["items"]:
        print(f"  [{it['score']}] {it['doc']} · {it['section']}")
