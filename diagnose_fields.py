"""逐字段对比：模型输出 vs 参考实现金标准（带符号偏差）。

用来看"模型是随机偏"还是"系统性偏"——
如果某个字段的 signed delta 均值明显不为零，说明它和公式的尺度/口径对不上，
那就该改提示词或改参考实现，而不是放宽容差掩盖。

用法::

    python diagnose_fields.py                       # 默认 DeepSeek-V3.2
    python diagnose_fields.py <model> 2             # 跑 2 场
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mdh import agents, llm, tasks  # noqa: E402
from mdh import reference as ref  # noqa: E402

NUMERIC = list(ref.JUDGED_NUMERIC)


def main(argv: list[str]) -> int:
    model = argv[1] if len(argv) > 1 else "deepseek-ai/DeepSeek-V3.2"
    n_scenes = int(argv[2]) if len(argv) > 2 else 1

    backend = llm.OpenAICompat(model=model, temperature=0.3, max_tokens=2048)
    all_tasks = tasks.build_tasks(n_generated=12)[:n_scenes]

    print(f"模型 {backend.model}\n")

    for scaffold in ("card_injected", "card_formula"):
        deltas: dict[str, list[float]] = {f: [] for f in NUMERIC}
        styles = gazes = 0
        style_hit = gaze_hit = 0
        print("=" * 78)
        print(f"scaffold = {scaffold}")
        for task in all_tasks:
            golds = {ln["id"]: ref.gold_shot(ln, task["characters"]) for ln in task["lines"]}
            out = agents.run_scaffold(task, scaffold, backend, max_repair=0, repeat=0)
            if out["output"] is None:
                print(f"  [{task['scene_id']}] JSON 解析失败")
                continue
            by_id = {s.get("line_id"): s for s in out["output"].get("shots", [])
                     if isinstance(s, dict)}
            for ln in task["lines"]:
                g = golds[ln["id"]]
                s = by_id.get(ln["id"])
                if not isinstance(s, dict):
                    print(f"  [{ln['id']}] 模型没给这条")
                    continue
                print(f"  [{ln['id']}] 「{ln['text'][:16]}」")
                for f in NUMERIC:
                    gv, sv = g.get(f), s.get(f)
                    if isinstance(gv, (int, float)) and isinstance(sv, (int, float)):
                        d = sv - gv
                        deltas[f].append(d)
                        flag = "" if abs(d) <= judge_tol(f) else "  ← 超容差"
                        print(f"      {f:<16} 金标 {gv:>8.2f}  模型 {sv:>8.2f}  "
                              f"偏差 {d:>+7.2f}{flag}")
                gs, ss = g["expression"]["style"], (s.get("expression") or {}).get("style")
                gg, sg = g["expression"]["gaze"], (s.get("expression") or {}).get("gaze")
                styles += 1
                style_hit += int(gs == ss)
                gazes += 1
                gaze_hit += int(gg == sg)
                if gs != ss or gg != sg:
                    print(f"      {'style/gaze':<16} 金标 {gs}/{gg}  模型 {ss}/{sg}")

        print(f"\n  --- {scaffold} 汇总：带符号偏差均值（正=模型偏高）")
        for f in NUMERIC:
            ds = deltas[f]
            if not ds:
                continue
            mean = sum(ds) / len(ds)
            mae = sum(abs(d) for d in ds) / len(ds)
            print(f"      {f:<16} n={len(ds):<3} 均值 {mean:>+7.2f}  MAE {mae:>6.2f}  "
                  f"容差 {judge_tol(f)}")
        if styles:
            print(f"      style 命中 {style_hit}/{styles}   gaze 命中 {gaze_hit}/{gazes}")
        print()
    return 0


def judge_tol(field: str) -> float:
    from mdh import judge
    return judge.DEFAULT_TOL[field]


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
