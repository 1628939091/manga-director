"""尺子敏感性：判读层指标有多少是「金标常数拍出来的」。

由来
----
潜台词改成逐句配对后，judge_plain 的 arousal 命中率从 46% 掉到 39%，
偏差从 −0.13 掉到 −0.68。最直接的反驳是：「你的 tense 潜台词加 1.2 分
是拍脑袋，模型不加 1.2 是你的尺子错了，不是模型错了。」

这个脚本把尺子的两个旋钮都拧一遍，看结论稳不稳：
  1. tense 潜台词加分幅度 bump ∈ {0, 0.6, 1.2, 1.8}
  2. arousal 容差 tol ∈ {0.5, 1.0, 1.5}

如果某个臂的命中率在 bump 从 0 扫到 1.8 的过程中几乎不动，
说明它的失分与这个常数无关，是判读本身的问题。

用法
----
    python ruler_sensitivity.py                          # 默认读两层记录
    python ruler_sensitivity.py outputs/records_xxx.jsonl
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mdh import analysis, reference as ref, tasks  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
DEFAULT_REC = os.path.join(OUT_DIR, "records_two_layer_subfix.jsonl")

BUMPS = (0.0, 0.6, 1.2, 1.8)
TOLS = (0.5, 1.0, 1.5)


def build_lut(n_scenes: int = 50) -> dict:
    """(scene_id, line_id) -> (text, subtext)。生成器固定 seed，可复现。"""
    lut = {}
    for t in tasks.build_tasks(n_generated=n_scenes):
        for line in t["lines"]:
            lut[(t["scene_id"], line["id"])] = (line["text"], line["subtext"])
    return lut


def collect(records: list[dict], arm: str, lut: dict) -> list[tuple]:
    """每行取 (模型给的 arousal, 不含加分的基准 arousal, 是否被加分)。"""
    rows = []
    for r in records:
        if r["scaffold"] != arm:
            continue
        for a in (r.get("appraisal") or []):
            key = (r["scene_id"], a["line_id"])
            if key not in lut:
                continue
            text, sub = lut[key]
            base = ref.appraise(text, "")["arousal"]
            full = ref.appraise(text, sub)["arousal"]
            rows.append((a["value"]["arousal"], base, abs(full - base) > 0.05))
    return rows


def sweep(rows: list[tuple]) -> dict:
    out = {}
    for bump in BUMPS:
        for tol in TOLS:
            hit = 0.0
            mae = 0.0
            for m, base, bumped in rows:
                gold = min(10.0, base + (bump if bumped else 0.0))
                hit += abs(m - gold) <= tol
                mae += abs(m - gold)
            n = len(rows) or 1
            out[(bump, tol)] = (hit / n * 100, mae / n)
    return out


def render(records: list[dict], arms: list[str], lut: dict) -> str:
    L = ["# 尺子敏感性：判读层指标有多少是金标常数拍出来的", ""]
    n_rows = len(collect(records, arms[0], lut)) if arms else 0
    bumped = sum(1 for _, _, b in collect(records, arms[0], lut) if b) if arms else 0
    L.append(f"记录：{len(records)} 条 / 判读行 {n_rows} / 其中被 tense 加分 {bumped} 行"
             f"（{bumped / (n_rows or 1) * 100:.0f}%）")
    L.append("")
    L.append("`reference.appraise()` 里有个常数：潜台词含「心跳/其实/忍住/怕/紧张」时 "
             "arousal +1.2。这个 1.2 不是拟合出来的，是导演直觉。")
    L.append("下面把它换成 0 / 0.6 / 1.2 / 1.8，同时把 arousal 容差从 ±0.5 扫到 ±1.5，"
             "看各臂的命中率会不会翻盘。")
    L.append("")

    for arm in arms:
        rows = collect(records, arm, lut)
        if not rows:
            continue
        tbl = sweep(rows)
        L.append(f"## `{arm}`")
        L.append("")
        L.append("arousal 命中率（行级，±tol 内算对）：")
        L.append("")
        L.append("| tense 加分 | " + " | ".join(f"±{t}" for t in TOLS) + " |")
        L.append("|---" * (len(TOLS) + 1) + "|")
        for bump in BUMPS:
            cells = " | ".join(f"{tbl[(bump, t)][0]:.1f}%" for t in TOLS)
            L.append(f"| +{bump:.1f} | {cells} |")
        L.append("")
        L.append("arousal MAE：")
        L.append("")
        L.append("| tense 加分 | " + " | ".join(f"±{t}" for t in TOLS) + " |")
        L.append("|---" * (len(TOLS) + 1) + "|")
        for bump in BUMPS:
            cells = " | ".join(f"{tbl[(bump, t)][1]:.2f}" for t in TOLS)
            L.append(f"| +{bump:.1f} | {cells} |")
        L.append("")

        # 跨度：把「尺子能解释多少」和「尺子解释不了多少」分开
        hits = [tbl[(b, 1.0)][0] for b in BUMPS]
        span = max(hits) - min(hits)
        best = BUMPS[hits.index(max(hits))]
        L.append(f"**±1.0 容差下，加分从 0 扫到 1.8，命中率变动 {span:.1f} 个百分点**"
                 f"（{min(hits):.1f}% ~ {max(hits):.1f}%，最优在 +{best:.1f}）。")
        if span < 5:
            L.append("")
            L.append("> 变动 <5 个百分点：**这个臂的失分与 1.2 这个常数无关**。"
                     "换任何合理的加分幅度都救不回来，问题在判读本身。")
        else:
            L.append("")
            L.append("> 变动 ≥5 个百分点：这个臂的分数**对金标常数敏感**，"
                     "报数时必须连带报出它是在哪个常数下得到的——"
                     "否则「我给模型打了多少分」这句话本身不成立。")
        L.append("")

    L.append("---")
    L.append("")
    L.append("## 怎么读这张表")
    L.append("")
    L.append("金标不是自然常数，是有人写下来的规则。所以判读层指标天然有两种成分：")
    L.append("**模型读得准不准**，和**尺子的刻度是不是模型能对齐的刻度**。")
    L.append("这个表把后者拧到极端，剩下的差距才是前者。")
    L.append("")
    L.append("要真正把两者分开，只有一条路：换成人工标注的金标"
             "（`python annotate.py score`），让尺子不再是我写的规则表。")
    L.append("")
    return "\n".join(L)


def main(argv: list[str]) -> int:
    path = argv[1] if len(argv) > 1 else DEFAULT_REC
    if not os.path.isabs(path):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
    if not os.path.exists(path):
        print(f"找不到记录：{path}")
        return 2

    records = analysis.load(path)
    arms = [a for a in analysis.scaffolds_of(records) if any(r.get("appraisal") for r in records)]
    if not arms:
        print("这份记录里没有判读层结果（appraisal 为空），只适用于两层记录。")
        return 1

    md = render(records, arms, build_lut())
    out = os.path.join(OUT_DIR, "ruler_sensitivity.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(md)
    print(md)
    print(f"\n已写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
