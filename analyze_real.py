"""真实剧本三方对比的汇总：MAE + bootstrap CI + Spearman，一次算完、只落一处。

为什么需要这个脚本
------------------
此前 `outputs/analysis_real_scripts.md` 里的 CI 与 ρ 是临时脚本算的，没提交。
后果是同一份 140 句数据在不同时间能得出两个 ρ（0.55 / 0.54）：`judge_plain`
在 11-01 14:36 重跑过一次，md 还停在上一轮，页面又自己从明细现算一遍——
三处数字并存，评审者一对账就会怀疑整份评测。

这个脚本把三件事固定下来：
  输入   outputs/real_eval.json（plain）、outputs/real_eval_cue.json（cue）、
         annotations/real_form.csv（只读取台词与出处）
  方法   bootstrap 2000 次、固定种子 7、平均秩 Spearman（与 scipy 同口径）
  输出   outputs/real_eval_summary.json（页面与 README 的唯一真值）
         + 可直接粘贴进 md 的两张表（--md）

用法：
    python analyze_real.py          # 落 json，打印两张表
    python analyze_real.py --md     # 只打印 markdown
"""

from __future__ import annotations

import csv
import json
import os
import random
import sys
from typing import Any

ROOT = os.path.dirname(os.path.abspath(__file__))
PLAIN = os.path.join(ROOT, "outputs", "real_eval.json")
CUE = os.path.join(ROOT, "outputs", "real_eval_cue.json")
FORM = os.path.join(ROOT, "annotations", "real_form.csv")
OUT = os.path.join(ROOT, "outputs", "real_eval_summary.json")

BOOT = 2000
SEED = 7


def mae(a: list[float], b: list[float]) -> float:
    return sum(abs(x - y) for x, y in zip(a, b)) / len(a)


def ranks(x: list[float]) -> list[float]:
    """平均秩（并列取均值）——与 scipy.stats.rankdata 的 'average' 一致。"""
    order = sorted(range(len(x)), key=lambda i: x[i])
    out = [0.0] * len(x)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and x[order[j + 1]] == x[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for t in range(i, j + 1):
            out[order[t]] = avg
        i = j + 1
    return out


def spearman(a: list[float], b: list[float]) -> float:
    ra, rb = ranks(a), ranks(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return num / den if den else 0.0


def boot_ci(a: list[float], b: list[float]) -> list[float]:
    """对「逐行绝对误差」做有放回重采样，取 2.5% / 97.5% 分位。"""
    rnd = random.Random(SEED)
    n = len(a)
    vals = []
    for _ in range(BOOT):
        idx = [rnd.randrange(n) for _ in range(n)]
        vals.append(sum(abs(a[i] - b[i]) for i in idx) / n)
    vals.sort()
    return [round(vals[int(0.025 * BOOT)], 2), round(vals[int(0.975 * BOOT) - 1], 2)]


def load_rows(path: str) -> dict[str, tuple]:
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return {r[0]: tuple(r) for r in d["rows"]}


def main(argv: list[str]) -> int:
    md_only = "--md" in argv
    plain = load_rows(PLAIN)
    cue = load_rows(CUE)
    ids = [i for i in plain if i in cue]
    if not ids:
        print("[失败] 两份 json 没有共同行；先跑 python eval_real.py")
        return 1

    human = [plain[i][1] for i in ids]
    model = [plain[i][2] for i in ids]
    cue_model = [cue[i][2] for i in ids]
    rule = [plain[i][3] for i in ids]

    arms = [
        ("judge_plain", model),
        ("judge_cue", cue_model),
        ("规则表", rule),
    ]
    rows = []
    for name, vals in arms:
        rows.append({
            "arm": name,
            "mae": round(mae(vals, human), 2),
            "ci": boot_ci(vals, human),
            "rho": round(spearman(vals, human), 2),
        })

    # 分片：按影片算 plain / cue / rule 的 MAE，以及 cue−plain
    form = {r["line_id"]: r for r in csv.DictReader(
        open(FORM, encoding="utf-8-sig", newline=""))}
    by_work: dict[str, list[str]] = {}
    for i in ids:
        by_work.setdefault(form.get(i, {}).get("出处", "未知"), []).append(i)
    works = []
    for w, ix in by_work.items():
        p = round(mae([plain[i][2] for i in ix], [plain[i][1] for i in ix]), 2)
        c = round(mae([cue[i][2] for i in ix], [cue[i][1] for i in ix]), 2)
        k = round(mae([plain[i][3] for i in ix], [plain[i][1] for i in ix]), 2)
        works.append({"work": w, "n": len(ix), "plain": p, "cue": c,
                      "rule": k, "delta": round(c - p, 2)})

    data: dict[str, Any] = {
        "n": len(ids), "boot": BOOT, "seed": SEED,
        "arms": rows, "by_work": works,
        "sources": {"plain": "outputs/real_eval.json",
                    "cue": "outputs/real_eval_cue.json",
                    "form": "annotations/real_form.csv"},
    }

    if not md_only:
        with open(OUT, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
        print(f"[落盘] {OUT}（n={len(ids)}）")

    print("\n| 尺子 | MAE vs 人 | 95% CI | 秩相关 ρ |")
    print("|---|---|---|---|")
    for r in rows:
        print(f"| `{r['arm']}` | {r['mae']:.2f} | [{r['ci'][0]:.2f}, {r['ci'][1]:.2f}] | "
              f"**{r['rho']:+.2f}** |" if r["arm"] == "judge_plain"
              else f"| `{r['arm']}` | {r['mae']:.2f} | [{r['ci'][0]:.2f}, {r['ci'][1]:.2f}] | "
                   f"{r['rho']:+.2f} |")
    print("\n| 影片 | n | plain | cue | 规则 | cue−plain |")
    print("|---|---|---|---|---|---|")
    for w in works:
        print(f"| {w['work']} | {w['n']} | {w['plain']:.2f} | {w['cue']:.2f} | "
              f"{w['rule']:.2f} | {w['delta']:+.2f} |")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
