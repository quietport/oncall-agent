"""评测：12 组故障场景，看运维 Agent 给出的根因有没有命中预期关键词。

这是「关键词命中率」，不是严格的语义评测 —— 但它便宜、可重复、能对比版本，
比"我觉得它答得挺好"有用得多。

    python eval.py
"""
import json
import time
from pathlib import Path

import llm
from agents import triage

BASE = Path(__file__).resolve().parent
CASES = json.loads((BASE / "eval_cases.json").read_text(encoding="utf-8"))


def hit(result, expects):
    """根因 + 证据摘要里，是否出现任意一个预期关键词（不区分大小写）。"""
    text = json.dumps(result, ensure_ascii=False).lower()
    for kw in expects:
        if kw.lower() in text:
            return True, kw
    return False, ""


def main():
    rows = []
    ok = 0
    t0 = time.time()
    for case in CASES:
        try:
            out = triage.run(case["alert"], llm)
            res = out.get("result") or {}
            passed, kw = hit(res, case["expect"])
            rows.append(
                {
                    "id": case["id"],
                    "ok": passed,
                    "kw": kw,
                    "severity": res.get("severity", ""),
                    "root_cause": (res.get("root_cause") or "")[:80],
                    "confidence": res.get("confidence", ""),
                    "cost": out.get("cost", 0),
                    "latency": out.get("latency", 0),
                    "replanned": out.get("replanned", False),
                }
            )
            if passed:
                ok += 1
        except Exception as e:  # noqa: BLE001
            rows.append({"id": case["id"], "ok": False, "kw": "", "root_cause": f"出错：{e}"})
        mark = "PASS" if rows[-1]["ok"] else "MISS"
        print(
            f"{mark}  {rows[-1]['id']:<4} {rows[-1].get('severity',''):<3} "
            f"{(rows[-1].get('root_cause') or '')[:46]}  [{rows[-1].get('kw','')}]"
        )

    total = len(CASES)
    cost = sum(r.get("cost") or 0 for r in rows)
    lat = [r.get("latency") or 0 for r in rows]
    print("\n" + "=" * 60)
    print(f"命中 {ok}/{total} = {ok/total:.0%}")
    print(f"总成本 {cost:.4f} 元，平均延迟 {sum(lat)/len(lat):.2f} 秒，总耗时 {time.time()-t0:.0f} 秒")
    print("=" * 60)

    out_csv = BASE / "eval_result.csv"
    with out_csv.open("w", encoding="utf-8-sig") as f:
        f.write("id,ok,keyword,severity,root_cause,confidence,cost,latency,replanned\n")
        for r in rows:
            f.write(
                f"{r['id']},{r['ok']},{r.get('kw','')},{r.get('severity','')},"
                f"\"{(r.get('root_cause') or '').replace(chr(34), '')}\","
                f"{r.get('confidence','')},{r.get('cost',0)},{r.get('latency',0)},{r.get('replanned',False)}\n"
            )
    print(f"明细已写入 {out_csv.name}")


if __name__ == "__main__":
    main()
