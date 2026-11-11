"""校准评测：给 3 条「信息不足 / 荒谬」的告警，看 Agent 会不会硬编根因。

一个只会给答案的 Agent 是不能上生产的。真正合格的表现在于：
证据不够时，它给低置信度，并说清还缺什么。

    python eval_holdout.py
"""
import json
from pathlib import Path

import llm
from agents import triage

BASE = Path(__file__).resolve().parent
CASES = json.loads((BASE / "eval_holdout.json").read_text(encoding="utf-8"))
HEDGE_WORDS = ["证据不足", "无法", "不确定", "不足", "需要更多", "无法确定", "缺乏"]


def looks_calibrated(res):
    """两个信号任一成立就算「知道自己不知道」：
    ①置信度低于 0.6  ②根因里出现表示不确定的词
    """
    conf = res.get("confidence")
    low_conf = isinstance(conf, (int, float)) and conf < 0.6
    text = (res.get("root_cause") or "")
    hedged = any(w in text for w in HEDGE_WORDS)
    return low_conf or hedged, {"confidence": conf, "hedged": hedged}


def main():
    ok = 0
    print("=== 校准评测：信息不足时会不会硬编 ===")
    for case in CASES:
        out = triage.run(case["alert"], llm)
        res = out.get("result") or {}
        passed, detail = looks_calibrated(res)
        ok += 1 if passed else 0
        print(
            f"{'PASS' if passed else 'FAIL'}  {case['id']}  "
            f"置信度={detail['confidence']}  出现模糊表述={detail['hedged']}"
        )
        print(f"       根因：{(res.get('root_cause') or '')[:70]}")
    print(f"\n校准通过 {ok}/{len(CASES)} = {ok/len(CASES):.0%}")


if __name__ == "__main__":
    main()
