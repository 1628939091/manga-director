"""模型自身的稳定性：同一批真实台词，多次采样，MAE 抖多少？

为什么必须先做这个：
    如果 judge_plain 的 0.93 是某一次采样的运气，那么「cue 比 plain 差 0.71」
    这个结论也可能只是采样噪声碰巧。
    先证明「同一个臂重复跑，MAE 波动远小于臂间差异」，跨臂比较才成立。
"""

from __future__ import annotations

import csv
import json
import os
import random
import statistics as st
import sys

sys.path.insert(0, r"D:\develop\2026-10-21-12-21-27\projects\manga-director")

from mdh import agents, llm  # noqa: E402
from mdh import reference as ref  # noqa: E402
from stage_adapter import NEUTRAL_CARD  # noqa: E402

ROOT = r"D:\develop\2026-10-21-12-21-27\projects\manga-director"
FORM = os.path.join(ROOT, "annotations", "real_form.csv")
OUT = os.path.join(ROOT, "outputs", "real_stability.json")

REPEATS = 3
ARM = "judge_plain"


def load_rows():
    return [r for r in csv.DictReader(open(FORM, encoding="utf-8-sig", newline=""))
            if (r.get("arousal(1-10)") or "").strip()]


def to_tasks(rows):
    groups = {}
    for r in rows:
        groups.setdefault((r["出处"], r["场次"]), []).append(r)
    tasks = []
    for i, ((tag, scene), rs) in enumerate(sorted(groups.items())):
        cast = []
        for r in rs:
            if r["说话人"] not in cast:
                cast.append(r["说话人"])
        if len(cast) < 2:
            cast.append("对手")
        sid = f"real-S{i:02d}"
        tasks.append({
            "scene_id": sid, "work_id": f"real-{tag}", "title": f"{tag}·{scene}",
            "characters": {c: NEUTRAL_CARD for c in cast},
            "lines": [{
                "id": r["line_id"], "character": r["说话人"], "scene": scene,
                "scene_type": "日常",
                "listener": cast[(cast.index(r["说话人"]) + 1) % len(cast)],
                "relation": "陌生人", "text": r["台词"], "subtext": "",
            } for r in rs],
        })
    return tasks


def main() -> int:
    rows = load_rows()
    human = {r["line_id"]: float(r["arousal(1-10)"]) for r in rows}
    tasks = to_tasks(rows)
    print(f"真实台词 {len(rows)} 行 / {len(tasks)} 场　臂 {ARM}　重复 {REPEATS} 次\n")

    runs = []
    for k in range(REPEATS):
        temp = 0.3 + 0.4 * k          # 0.3 / 0.7 / 1.1：同时测温度敏感性
        backend = llm.OpenAICompat(
            model=os.environ.get("MDH_MODEL", "deepseek-chat"),
            temperature=temp, seed=1000 + k)
        got = {}
        for t in tasks:
            out = agents.run_scaffold(t, ARM, backend, repeat=k)
            for a in ((out.get("appraisal") or {}).get("lines") or []):
                if isinstance(a, dict) and isinstance(a.get("arousal"), (int, float)):
                    got[a["line_id"]] = float(a["arousal"])
        pairs = [(human[lid], got[lid]) for lid in got if lid in human]
        mae = sum(abs(a - b) for a, b in pairs) / len(pairs)
        mean = st.mean([b for _, b in pairs])
        runs.append({"temp": temp, "mae": mae, "mean": mean, "n": len(pairs)})
        print(f"  run{k}  temp={temp:.1f}  n={len(pairs)}  MAE={mae:.2f}  均值={mean:.2f}")

    maes = [r["mae"] for r in runs]
    print(f"\nMAE 范围 {min(maes):.2f} ~ {max(maes):.2f}　跨度 {max(maes) - min(maes):.3f}")
    print(f"臂间差异（cue−plain = +0.71）与本次采样跨度之比："
          f"{0.71 / max(max(maes) - min(maes), 1e-6):.1f}×")
    json.dump(runs, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n明细：{OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
