"""真实剧本上的三方对比：模型 vs 人 vs 规则表。

这是整条路线里唯一无法替代的一步——前两批实验的金标都是关键词表，
而它在真实台词上已经塌缩（36 行多数落在默认桶 5.0）。
只有人工标注能回答：模型在真实剧本上到底还行不行。

用法：
    python eval_real.py            # 读 annotations/real_form.csv 已填的行
    python eval_real.py --arm judge_cue
"""

from __future__ import annotations

import csv
import json
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mdh import agents, llm  # noqa: E402
from mdh import reference as ref  # noqa: E402
from stage_adapter import NEUTRAL_CARD  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
FORM = os.path.join(ROOT, "annotations", "real_form.csv")


def load() -> list[dict]:
    rows = [r for r in csv.DictReader(open(FORM, encoding="utf-8-sig", newline=""))
            if (r.get("arousal(1-10)") or "").strip()]
    return rows


def to_tasks(rows: list[dict]) -> list[dict]:
    """按 (出处, 场次) 分组成场景，这样模型能拿到同场上下文。"""
    groups: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        groups.setdefault((r["出处"], r["场次"]), []).append(r)

    tasks = []
    for i, ((tag, scene), rs) in enumerate(sorted(groups.items())):
        cast = []
        for r in rs:
            if r["说话人"] not in cast:
                cast.append(r["说话人"])
        if len(cast) < 2:                     # 独角戏补一个"对方"，只为让关系成立
            cast.append("对手")
        sid = f"real-S{i:02d}"
        tasks.append({
            "scene_id": sid,
            "work_id": f"real-{tag}",
            "title": f"{tag}·{scene}",
            "characters": {c: NEUTRAL_CARD for c in cast},
            "lines": [{
                "id": r["line_id"],
                "character": r["说话人"],
                "scene": scene,
                "scene_type": "日常",
                "listener": cast[(cast.index(r["说话人"]) + 1) % len(cast)],
                "relation": "陌生人",
                "text": r["台词"],
                "subtext": "",
            } for r in rs],
        })
    return tasks


def main(argv: list[str]) -> int:
    arm = "judge_plain"
    if "--arm" in argv:
        arm = argv[argv.index("--arm") + 1]

    rows = load()
    if not rows:
        print(f"[等待] {FORM} 还没有填好的行")
        return 1
    human = {r["line_id"]: float(r["arousal(1-10)"]) for r in rows}
    tasks = to_tasks(rows)
    print(f"真实台词 {len(rows)} 行 / {len(tasks)} 场　臂：{arm}")

    backend = llm.OpenAICompat(model=os.environ.get("MDH_MODEL", "deepseek-ai/DeepSeek-V3.2"))
    got: dict[str, float] = {}
    for t in tasks:
        out = agents.run_scaffold(t, arm, backend, repeat=0)
        # 判读臂的原始输出在 out["appraisal"]，结构是 {"scene_id":..,"lines":[{line_id,arousal,..}]}
        for a in ((out.get("appraisal") or {}).get("lines") or []):
            if isinstance(a, dict) and isinstance(a.get("arousal"), (int, float)):
                got[a.get("line_id")] = float(a["arousal"])

    print(f"模型返回 {len(got)} 行")

    pairs = [(r["line_id"], human[r["line_id"]], got.get(r["line_id"]),
              ref.appraise(r["台词"], "")["arousal"])
             for r in rows if isinstance(got.get(r["line_id"]), (int, float))]
    if not pairs:
        print("[失败] 没有可比的模型输出；检查 appraisal 字段名")
        return 1

    h = [p[1] for p in pairs]
    m = [p[2] for p in pairs]
    k = [p[3] for p in pairs]

    def mae(a, b):
        return sum(abs(x - y) for x, y in zip(a, b)) / len(a)

    print(f"\n可比 {len(pairs)} 行")
    print(f"  人标 均值 {st.mean(h):.2f}  标准差 {st.pstdev(h):.2f}")
    print(f"  模型 均值 {st.mean(m):.2f}  标准差 {st.pstdev(m):.2f}")
    print(f"  规则 均值 {st.mean(k):.2f}  标准差 {st.pstdev(k):.2f}")
    print(f"\n  MAE(模型, 人) = {mae(m, h):.2f}")
    print(f"  MAE(规则, 人) = {mae(k, h):.2f}")
    print(f"  MAE(模型, 规则) = {mae(m, k):.2f}")

    out = os.path.join(ROOT, "outputs", "real_eval.json")
    json.dump({"arm": arm, "n": len(pairs),
               "human": h, "model": m, "rule": k,
               "rows": [(p[0], p[1], p[2], p[3]) for p in pairs]},
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n明细：{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
