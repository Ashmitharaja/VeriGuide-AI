"""8. Evaluation: accuracy, faithfulness, hallucination rate, abstention.

Run:  python -m veriguide.evaluate
Each case lists the document that should be retrieved and keywords a correct answer must contain.
Cases with "answerable": false check that the system abstains instead of hallucinating.
"""

import json

from . import config
from .rag import VeriGuide

CASES = [
    {"q": "My motor is overheating. What could be the reason?", "doc": "induction_motor_manual.md",
     "keywords": ["overload", "fan"]},
    {"q": "How often should the IM-300 bearings be regreased?", "doc": "induction_motor_manual.md",
     "keywords": ["4000"]},
    {"q": "What is the minimum insulation resistance for the motor?", "doc": "induction_motor_manual.md",
     "keywords": ["1 megaohm"]},
    {"q": "The pump is running but there is no flow", "doc": "centrifugal_pump_guide.md",
     "keywords": ["prime"]},
    {"q": "How much seal leakage is acceptable on the CP-50 pump?", "doc": "centrifugal_pump_guide.md",
     "keywords": ["5 drops"]},
    {"q": "What does fault code F07 mean on the VFD?", "doc": "vfd_fault_codes.txt",
     "keywords": ["heatsink"]},
    {"q": "The drive trips with overvoltage when stopping", "doc": "vfd_fault_codes.txt",
     "keywords": ["deceleration"]},
    {"q": "How do I bake a chocolate cake?", "answerable": False},
    {"q": "What is the warranty period for the IM-300 motor?", "answerable": False},
]


def main():
    vg = VeriGuide()
    print(f"Mode: {vg.mode} | chunks: {vg.store.size}\n")
    rows = []
    for c in CASES:
        r = vg.ask(c["q"])
        answerable = c.get("answerable", True)
        retrieved = [s["doc"] for s in r["sources"]]
        row = {
            "q": c["q"], "answerable": answerable, "abstained": r["abstained"],
            "confidence": r["confidence"], "supported_ratio": r["verification"]["supported_ratio"],
            "retrieval_hit": (c.get("doc") in retrieved) if answerable else None,
            "correct": (not r["abstained"] and all(k.lower() in r["answer"].lower() for k in c["keywords"]))
            if answerable else r["abstained"],
            "answer": r["answer"],
        }
        rows.append(row)
        mark = "PASS" if row["correct"] else "FAIL"
        print(f"[{mark}] {c['q']}\n       conf={r['confidence']} abstained={r['abstained']} "
              f"faithfulness={row['supported_ratio']}")

    ans = [r for r in rows if r["answerable"]]
    unans = [r for r in rows if not r["answerable"]]
    answered = [r for r in rows if not r["abstained"]]
    claims = sum(len(vg.verify(r["answer"], vg.retrieve(r["q"]))["claims"]) for r in answered) or 1
    unsupported = sum(
        sum(not c["supported"] for c in vg.verify(r["answer"], vg.retrieve(r["q"]))["claims"]) for r in answered
    )
    metrics = {
        "accuracy": round(sum(r["correct"] for r in ans) / len(ans), 3),
        "retrieval_hit_rate": round(sum(bool(r["retrieval_hit"]) for r in ans) / len(ans), 3),
        "faithfulness": round(sum(r["supported_ratio"] for r in answered) / max(1, len(answered)), 3),
        "hallucination_rate": round(unsupported / claims, 3),
        "correct_abstention_rate": round(sum(r["abstained"] for r in unans) / len(unans), 3),
    }
    print("\n" + json.dumps(metrics, indent=2))
    out = config.STORE_DIR / "eval_results.json"
    out.write_text(json.dumps({"mode": vg.mode, "metrics": metrics, "cases": rows}, indent=2))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
