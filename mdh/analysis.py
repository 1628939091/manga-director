"""真实运行的分析层：连续口径 + 反事实 + 失败归因。

为什么要单独一层
----------------
`judge.DEFAULT_TAU = 0.8` 是在 Mock 上校准出来的：Mock 的缺陷模型是
"随机坏 1–2 个字段"，所以 8 个字段命中 6–7 个就能过 0.8。
真实模型平均坏 4.2 个，"每行 ≥0.8 且每场每行都过"在两处同时不可达，
pass@1 恒为 0——指标下限饱和，和项目一"120/120 全过"的上限饱和是同一类问题。

本层做四件事：
1. **主指标换成连续量**：场景平均命中率 + 每行平均错字段数，保留 τ 口径作副表；
2. **反事实**：把 style/gaze 视为全对，看还剩多少——判断是不是只卡在这两个字段；
3. **条件命中率**：AU 在 style 对/错时的命中差，暴露失败传播（级联）；
4. **错因分解**：style 错的行里 intensity 是否也错，区分"判读错"与"分支选错"。

口径纪律：所有区间一律按**场景**聚类 Bootstrap，k 次重复不计入样本量。
"""

from __future__ import annotations

import json
import os
import statistics as st

from . import metrics
from . import reference as ref

FIELDS = list(ref.JUDGED_FIELDS)  # intensity/f0_offset/f0_range/rate/pause/style/gaze/au

FIELD_CN = {
    "intensity": "强度",
    "f0_offset": "音高偏移",
    "f0_range": "音高范围",
    "rate": "语速",
    "pause_after_ms": "停顿",
    "style": "表演风格",
    "gaze": "视线",
    "au": "面部 AU",
}


# --------------------------------------------------------------------------
# 载入
# --------------------------------------------------------------------------
def load(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def scaffolds_of(records: list[dict]) -> list[str]:
    seen: list[str] = []
    for r in records:
        if r["scaffold"] not in seen:
            seen.append(r["scaffold"])
    return seen


# --------------------------------------------------------------------------
# 基本量
# --------------------------------------------------------------------------
def _rows(records: list[dict], scaffold: str):
    for r in records:
        if r["scaffold"] != scaffold:
            continue
        for row in r["rows"]:
            yield r, row


def field_hits(records: list[dict], scaffold: str) -> dict[str, tuple[int, int]]:
    out = {f: [0, 0] for f in FIELDS}
    for _, row in _rows(records, scaffold):
        for f in FIELDS:
            if f in row["hits"]:
                out[f][1] += 1
                out[f][0] += int(bool(row["hits"][f]))
    return {f: (h, n) for f, (h, n) in out.items()}


def field_hit_ci(records: list[dict], scaffold: str, field: str) -> tuple[float, float, float]:
    """按场景聚类的 95% CI：同一场戏的多行共用角色卡与场景规范，不独立。"""
    by_scene: dict[str, list[float]] = {}
    for r, row in _rows(records, scaffold):
        if field in row["hits"]:
            by_scene.setdefault(r["scene_id"], []).append(1.0 if row["hits"][field] else 0.0)
    return metrics.cluster_bootstrap_ci(by_scene)


def scene_scores(records: list[dict], scaffold: str) -> dict[str, list[float]]:
    """scene_id -> 每次运行的场景平均命中率（k 个值）。"""
    out: dict[str, list[float]] = {}
    for r in records:
        if r["scaffold"] == scaffold:
            out.setdefault(r["scene_id"], []).append(r["hit_rate"])
    return out


def scene_mean_ci(records: list[dict], scaffold: str) -> tuple[float, float, float]:
    """每场先对 k 次取均值，再按场景聚类 Bootstrap。"""
    per_scene = {k: [sum(v) / len(v)] for k, v in scene_scores(records, scaffold).items()}
    return metrics.cluster_bootstrap_ci(per_scene)


def avg_wrong_fields(records: list[dict], scaffold: str) -> float:
    """每行平均错几个字段（8 个受判字段里）。比 pass 敏感得多的一等指标。"""
    acc = []
    for _, row in _rows(records, scaffold):
        if row["hits"]:
            acc.append(sum(1 for v in row["hits"].values() if not v))
    return sum(acc) / len(acc) if acc else 0.0


def line_pass(records: list[dict], scaffold: str, tau: float) -> float:
    ok = n = 0
    for _, row in _rows(records, scaffold):
        if not row["hits"]:
            continue
        n += 1
        ok += int(row["l1_clean"] and sum(row["hits"].values()) / len(row["hits"]) >= tau)
    return ok / n if n else 0.0


def scene_mean_above(records: list[dict], scaffold: str, thr: float) -> float:
    """场景平均命中率 ≥ thr 的场次占比（按每次运行计，不是按场去重）。"""
    vals = [r["hit_rate"] for r in records if r["scaffold"] == scaffold]
    return sum(1 for v in vals if v >= thr) / len(vals) if vals else 0.0


def quantiles(values: list[float]) -> tuple[float, float, float, float, float]:
    if not values:
        return (0.0, 0.0, 0.0, 0.0, 0.0)
    if len(values) < 4:
        return (min(values), min(values), st.median(values), max(values), max(values))
    q = st.quantiles(values, n=4)
    return (min(values), q[0], q[1], q[2], max(values))


# --------------------------------------------------------------------------
# 反事实：style/gaze 若全对，还剩多少
# --------------------------------------------------------------------------
def counterfactual_style_gaze(records: list[dict], scaffold: str,
                              tau: float = 0.8) -> dict:
    rates = []
    for _, row in _rows(records, scaffold):
        if not row["hits"]:
            continue
        h = dict(row["hits"])
        h["style"] = h["gaze"] = True
        rates.append(sum(h.values()) / len(h))
    if not rates:
        return {"mean": 0.0, "line_pass": 0.0, "n": 0}
    return {
        "mean": sum(rates) / len(rates),
        "line_pass": sum(1 for x in rates if x >= tau) / len(rates),
        "n": len(rates),
    }


def required_hit_rate(n_fields: int = 8, tau: float = 0.8) -> int:
    """在 τ 下每行至少要命中几个字段。"""
    import math
    return math.ceil(tau * n_fields - 1e-9)


# --------------------------------------------------------------------------
# 失败传播与错因分解
# --------------------------------------------------------------------------
def au_cascade(records: list[dict], scaffold: str) -> dict:
    """AU 命中在 style 对/错两种条件下的差异——级联的直接证据。"""
    ok = [0, 0]
    bad = [0, 0]
    for _, row in _rows(records, scaffold):
        h = row["hits"]
        if "au" not in h or "style" not in h:
            continue
        bucket = ok if h["style"] else bad
        bucket[1] += 1
        bucket[0] += int(bool(h["au"]))
    rate = lambda b: (b[0] / b[1]) if b[1] else None
    return {"style_ok": (ok[0], ok[1], rate(ok)), "style_bad": (bad[0], bad[1], rate(bad))}


def style_error_split(records: list[dict], scaffold: str) -> dict:
    """style 错的行里，intensity 是否也错——区分「判读错」与「分支选错」。

    注意：这是**相关**不是因果。intensity 对也可能来自被上限截断，
    此时 arousal 仍可能错、withheld 仍可能错。所以本表只能说
    「多数 style 错误伴随上游数值错误」，不能说「其中 X% 可被函数化救回」。
    """
    both = only = 0
    for _, row in _rows(records, scaffold):
        h = row["hits"]
        if "style" not in h or h["style"]:
            continue
        if h.get("intensity") is False:
            both += 1
        elif h.get("intensity") is True:
            only += 1
    tot = both + only
    return {"n": tot, "both_wrong": both, "only_style": only,
            "both_ratio": both / tot if tot else 0.0}


def l1_breakdown(records: list[dict], scaffold: str) -> dict[str, int]:
    """L1 错误按类型计数，看结构性问题还是内容问题。"""
    cnt: dict[str, int] = {}
    for r, row in _rows(records, scaffold):
        for e in row.get("errors") or []:
            key = e.split("（")[0][:24]
            cnt[key] = cnt.get(key, 0) + 1
    return cnt


# --------------------------------------------------------------------------
# 报告
# --------------------------------------------------------------------------
def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def split_breakdown(records: list[dict], scaffolds: list[str], tau: float) -> list[dict]:
    """按 train / test 分组统计（test = 按 work_id 留出的作品）。

    为什么要单列一表：如果只有 train 指标，任何"臂比臂好"的结论都无法排除
    "在见过的作品上调好了提示词"。按作品留出才能回答"换一批戏还成不成立"。
    同一部作品的角色卡与说话风格相同，按场景随机切分会泄漏——
    这是评测基准建设的基本纪律。
    """
    splits = [s for s in ("train", "test")
              if any(r.get("split") == s for r in records)]
    out = []
    for sc in scaffolds:
        for sp in splits:
            sub = [r for r in records if r["scaffold"] == sc and r.get("split") == sp]
            if not sub:
                continue
            st_ = appraisal_stats(sub, sc)
            out.append({
                "scaffold": sc, "split": sp,
                "scenes": len({r["scene_id"] for r in sub}),
                "median": quantiles([r["hit_rate"] for r in sub])[2],
                "ci": scene_mean_ci(sub, sc),
                "wrong": avg_wrong_fields(sub, sc),
                "mae": st_["arousal_mae"] if st_["n"] else None,
                "pass1": sum(1 for r in sub if r["passed"]) / len(sub),
            })
    return out


def _ci(ci, pct: bool = True) -> str:
    lo, hi, pt = ci
    if pct:
        return f"{pt * 100:.1f}% [{lo * 100:.1f}, {hi * 100:.1f}]"
    return f"{pt:.3f} [{lo:.3f}, {hi:.3f}]"


def render_real(records: list[dict], meta: dict) -> str:
    scs = meta.get("scaffolds") or scaffolds_of(records)
    k = meta.get("k", 0)
    tau = meta.get("tau", 0.8)
    need = required_hit_rate(len(FIELDS), tau)
    n_scene = len({r["scene_id"] for r in records if r["scaffold"] == scs[0]}) if scs else 0
    L: list[str] = []

    L.append("# 真实模型运行分析（连续口径 + 失败归因）")
    L.append("")
    L.append(f"> 模型：`{meta.get('model', '-')}`　k = {k}　"
             f"记录 {len(records)} 次运行 / "
             f"{sum(len(r['rows']) for r in records if r['scaffold'] == scs[0])} 行 × {len(scs)} 臂")
    L.append("")

    # ---------------- 0 ----------------
    L.append("## 0. 结论先行")
    L.append("")
    L.append("1. **原始判据不可达，不是模型差。** τ=0.8 要求每行至少命中 "
             f"**{need}/{len(FIELDS)}** 个字段，且每场每一行都要过。"
             "真实模型平均每行错 4 个字段左右，几何上就不可能有场次通过。"
             "这个阈值是在 Mock 上校准的——Mock 只随机坏 1–2 个字段，"
             "所以 0.8 对它是合适的；搬到真实模型上立刻失去分辨率。")
    L.append("2. **公式确实被模型执行了，但没被执行对。** 算术类字段（音高范围、停顿）"
             "随公式显著上升，绝对值仍停在 4–8 成；"
             "需要语义推理的字段（风格、视线）停在四选一的随机水平附近。")
    L.append("3. **更精确的指令反而多了一条失败路径。** 给公式等于给了一张"
             "「按风格查 AU 的固定表」；风格一错，AU 的命中率断崖式下跌。"
             "这是本次运行里证据最强的一条。")
    L.append("")

    # ---------------- 1 ----------------
    L.append("## 1. 为什么 pass@1 = 0：判据不可达的定量证据")
    L.append("")
    L.append("| scaffold | 每行平均错字段 | 反事实：style/gaze 全对后的平均命中率 | 反事实后行级 ≥τ 的占比 |")
    L.append("|---|---|---|---|")
    for s in scs:
        cf = counterfactual_style_gaze(records, s, tau)
        w = avg_wrong_fields(records, s)
        L.append(f"| `{s}` | {w:.2f} | {cf['mean']:.3f} | {_pct(cf['line_pass'])} |")
    L.append("")
    L.append("**读法**：即使把两个语义字段直接当作全对（这是上界，实际做不到），"
             "行级达标率仍只有个位数到两成——说明瓶颈不止 style/gaze，"
             "算术字段本身也没进容差。所以「把公式做成函数」只能救一部分，"
             "救不了全部：真正的堵点在**判读**，不在计算。")
    L.append("")

    # ---------------- 2 ----------------
    L.append("## 2. 主表（连续口径，有分辨率）")
    L.append("")
    L.append("| scaffold | 场景平均命中率（中位数 [p25, p75]） | 聚类 95% CI | 场景均值 ≥0.50 | ≥0.60 | 每行错字段 |")
    L.append("|---|---|---|---|---|---|")
    for s in scs:
        vals = [r["hit_rate"] for r in records if r["scaffold"] == s]
        mn, q1, med, q3, mx = quantiles(vals)
        L.append(f"| `{s}` | {med:.2f} [{q1:.2f}, {q3:.2f}] | {_ci(scene_mean_ci(records, s))} | "
                 f"{_pct(scene_mean_above(records, s, 0.50))} | "
                 f"{_pct(scene_mean_above(records, s, 0.60))} | "
                 f"{avg_wrong_fields(records, s):.2f} |")
    L.append("")
    L.append("> 场景平均命中率＝该场所有行、8 个字段的命中比例，对 k 次运行取均值。"
             "它不再需要「每行都过」这种连乘条件，因此能看出臂与臂的差别。"
             "「≥0.50 / ≥0.60」两列的阈值是**显式设计选择**，不是实证结论——"
             "写进简历时必须同时给出，否则换个阈值故事就变。")
    L.append("")

    # ---------------- 3 ----------------
    L.append("## 3. 逐字段命中率（带按场景聚类的 95% CI）")
    L.append("")
    head = "| scaffold | " + " | ".join(FIELD_CN[f] for f in FIELDS) + " |"
    L.append(head)
    L.append("|---" * (len(FIELDS) + 1) + "|")
    for s in scs:
        fh = field_hits(records, s)
        cells = []
        for f in FIELDS:
            h, n = fh[f]
            cells.append(f"{h / n * 100:.0f}%" if n else "—")
        L.append(f"| `{s}` | " + " | ".join(cells) + " |")
    L.append("")
    base = scs[0] if scs else None
    L.append("**字段级 CI（取表现最好的臂）**")
    L.append("")
    target = max(scs, key=lambda s: metrics.mean_hit_rate(records, s)) if scs else None
    if target:
        L.append("| 字段 | 命中率 (95% CI) |")
        L.append("|---|---|")
        for f in FIELDS:
            L.append(f"| {FIELD_CN[f]} | {_ci(field_hit_ci(records, target, f))} |")
        L.append("")
        L.append(f"> 上表是 `{target}`。区间按场景聚类（{n_scene} 个簇），"
                 "所以宽度反映的是「换一批戏还成不成立」，不是「行数够不够」。")
        L.append("")

    # ---------------- 4 ----------------
    L.append("## 4. 失败传播：AU 随 style 级联")
    L.append("")
    L.append("| scaffold | style 对 → AU 命中 | style 错 → AU 命中 | 比值 |")
    L.append("|---|---|---|---|")
    for s in scs:
        c = au_cascade(records, s)
        o, b = c["style_ok"], c["style_bad"]
        if o[2] is None or b[2] is None:
            continue
        ratio = (o[2] / b[2]) if b[2] else float("inf")
        L.append(f"| `{s}` | {_pct(o[2])} (n={o[1]}) | {_pct(b[2])} (n={b[1]}) | {ratio:.1f}× |")
    L.append("")
    L.append("**这是本次运行最重要的一条**：AU 在参考实现里是「按 style 分支查表」，"
             "所以 style 一错，AU 必然跟着错。把公式写进提示词，等于把这条依赖关系"
             "也交给了模型；模型的 style 判断不稳，AU 就从一个相对容易的字段"
             "（不给公式时也有 7 成以上）退化成最难对的字段。"
             "**更精确的指令不会减少失败，只会改变失败的传播路径。**")
    L.append("")

    # ---------------- 5 ----------------
    L.append("## 5. style 的错因分解")
    L.append("")
    L.append("| scaffold | style 错的行数 | 其中 intensity 也错 | 仅 style 错（分支选错） |")
    L.append("|---|---|---|---|")
    for s in scs:
        d = style_error_split(records, s)
        if not d["n"]:
            continue
        L.append(f"| `{s}` | {d['n']} | {d['both_wrong']}（{_pct(d['both_ratio'])}） | "
                 f"{d['only_style']}（{_pct(1 - d['both_ratio'])}） |")
    L.append("")
    L.append("> ⚠️ 这是相关，不是因果。intensity 对也可能来自被上限截断，"
             "此时 arousal 仍然可能是错的。所以只能读成"
             "「多数 style 错误伴随上游数值错误」，不能读成"
             "「其中固定比例可以靠函数化救回」。")
    L.append("")

    # ---------------- 6 ----------------
    L.append("## 6. L1 结构健康度")
    L.append("")
    L.append("| scaffold | L1 全清率 | 主要错误类型 |")
    L.append("|---|---|---|")
    for s in scs:
        rs = [r for r in records if r["scaffold"] == s]
        clean = sum(1 for r in rs if r["l1_clean"]) / len(rs) if rs else 0.0
        errs = {}
        for r in rs:
            for e in r.get("errors") or []:
                key = e.split("（")[0][:20]
                errs[key] = errs.get(key, 0) + 1
        top = "；".join(f"{k}×{v}" for k, v in sorted(errs.items(), key=lambda x: -x[1])[:2]) or "—"
        L.append(f"| `{s}` | {_pct(clean)} | {top} |")
    L.append("")
    L.append("> L1 全清率低说明大量运行在结构层就已经废掉（缺字段、越界、"
             "evidence 不是原文），这些行的 8 字段命中率记为 0，会把均值往下拽。"
             "看第 3 节逐字段表时要把这部分和「字段算错」分开。")
    L.append("")

    # ---------------- 7 ----------------
    L.append("## 7. 局限（必须一起报）")
    L.append("")
    n_lines = sum(len(r["rows"]) for r in records if r["scaffold"] == scs[0])
    L.append(f"- **样本太小**：{n_scene} 场戏。一场 = 约 {100 / n_scene:.0f} 个百分点，"
             "所以臂与臂之间 8 个点以内的差距现在不能下结论。")
    L.append(f"- **有效刺激数**：{n_lines} 行台词，唯一文本由 `mdh/lines.py` 的库决定"
             "（当前 139 句）。同文本的金标准相同，所以有效自由度按唯一文本数算，"
             "不按行数算——这是本报告所有 MAE 的隐含分母。")
    L.append("- **金标准本身是规则**：参考实现的 arousal 来自关键词表，"
             "所以「与金标准一致」不等于「判对了」，只等于「和我的规则表一致」。"
             "要把它变成能力结论，必须补人工标注 + κ。")
    L.append("- **τ 与容差是设计变量**：本报告用连续口径绕开了它们，"
             "但第 2 节的两个阈值列同样是设计变量，换阈值排名会变。")
    L.append("")
    return "\n".join(L)


def save(md: str, path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(md)
    return path


# ==========================================================================
# 两层架构实验（judge_plain / judge_cue）
# ==========================================================================
APPR_FIELDS = ("arousal", "valence", "intent", "evidence")


def appraisal_stats(records: list[dict], scaffold: str) -> dict:
    """判读层的基本量。rows 粒度是"一条台词的判读"。"""
    mae, nd, exact = [], 0, 0
    bias: list[float] = []
    hits = {f: [0, 0] for f in APPR_FIELDS}
    per_scene: dict[str, list[float]] = {}
    for r in records:
        if r["scaffold"] != scaffold or not r.get("appraisal"):
            continue
        scene_ok = []
        for row in r["appraisal"]:
            if "arousal" in row["deltas"]:
                mae.append(row["deltas"]["arousal"])
                nd += 1
                exact += int(row["deltas"]["arousal"] == 0.0)
            s = (row.get("signed") or {}).get("arousal")
            if s is not None:
                bias.append(s)
            for f in APPR_FIELDS:
                if f in row["hits"]:
                    hits[f][1] += 1
                    hits[f][0] += int(bool(row["hits"][f]))
                    if f == "arousal":
                        scene_ok.append(1.0 if row["hits"][f] else 0.0)
        if scene_ok:
            per_scene.setdefault(r["scene_id"], []).append(sum(scene_ok) / len(scene_ok))
    return {
        "arousal_mae": sum(mae) / len(mae) if mae else 0.0,
        "arousal_bias": sum(bias) / len(bias) if bias else 0.0,
        "arousal_exact": exact / nd if nd else 0.0,
        "hits": {f: (h, n) for f, (h, n) in hits.items()},
        "per_scene": per_scene,
        "n": nd,
    }


def arousal_bias_after_offset(records: list[dict], scaffold: str, offset: float) -> float:
    """扣掉常数偏差后的 MAE —— 判断"偏差是系统性的还是随机的"。"""
    acc = []
    for r in records:
        if r["scaffold"] != scaffold or not r.get("appraisal"):
            continue
        for row in r["appraisal"]:
            s = (row.get("signed") or {}).get("arousal")
            if s is not None:
                acc.append(abs(s - offset))
    return sum(acc) / len(acc) if acc else 0.0


def end_to_end_by_arousal(records: list[dict], scaffold: str) -> dict:
    """端到端行命中率，按「该行 arousal 是否判对」分两组。

    这是本实验的**核心证据**：如果判对时端到端 ≈ 100%，
    说明标定层是精确的、残差全部来自判读；那"模型算不准"就不再是问题，
    因为根本不需要它算。
    """
    ok, bad = [], []
    for r in records:
        if r["scaffold"] != scaffold or not r.get("appraisal"):
            continue
        a_by_line = {row["line_id"]: row["hits"].get("arousal") for row in r["appraisal"]}
        for row in r["rows"]:
            if not row["hits"]:
                continue
            rate = sum(row["hits"].values()) / len(row["hits"])
            a = a_by_line.get(row["line_id"])
            if a is True:
                ok.append(rate)
            elif a is False:
                bad.append(rate)
    f = lambda v: (sum(v) / len(v)) if v else 0.0
    return {"arousal_ok": (f(ok), len(ok)), "arousal_bad": (f(bad), len(bad))}


def render_two_layer(records: list[dict], meta: dict, baseline: list[dict] | None = None) -> str:
    scs = meta.get("scaffolds") or scaffolds_of(records)
    k = meta.get("k", 0)
    tau = meta.get("tau", 0.8)
    need = required_hit_rate(len(FIELDS), tau)
    L: list[str] = []

    L.append("# 两层架构实验：判读交给模型，标定交给代码")
    L.append("")
    L.append(f"> 模型：`{meta.get('model', '-')}`　k = {k}　"
             f"臂：{' / '.join('`%s`' % s for s in scs)}　"
             f"记录 {len(records)} 次运行")
    L.append("")
    L.append("架构：模型只输出 `arousal / valence / intent / evidence`，"
             "其余 7 个表演参数由 `reference.calibrate` 确定性计算。"
             "判卷仍是同一个 `judge_scene`，所以端到端数字与单层实验可比。")
    L.append("")

    # ---------------- 0 ----------------
    L.append("## 0. 结论先行")
    L.append("")
    for s in scs:
        st_ = appraisal_stats(records, s)
        e2e = end_to_end_by_arousal(records, s)
        L.append(f"- `{s}`：arousal MAE **{st_['arousal_mae']:.2f}**"
                 f"（带符号偏差 {st_['arousal_bias']:+.2f}），"
                 f"精确命中 {_pct(st_['arousal_exact'])}，"
                 f"意图准确率 {_pct(_ratio(st_['hits']['intent']))}；"
                 f"端到端行命中率在「arousal 判对」组为 **{e2e['arousal_ok'][0]:.3f}**、"
                 f"「判错」组为 {e2e['arousal_bad'][0]:.3f}。")
    L.append("")
    L.append("> 若两者差距接近满命中与零命中，说明**标定层是精确的，全部残差来自判读**。"
             "这就是把标定移出模型的直接依据：不是模型不需要更努力，"
             "而是这层工作本来就不该由它做。")
    L.append("")

    # ---------------- 1 ----------------
    L.append("## 1. 判读层准确度")
    L.append("")
    L.append("| 臂 | arousal MAE | arousal 带符号偏差 | 扣掉偏差后 MAE | 命中@±1.0 | 精确命中 | valence 命中@±0.3 | 意图准确率 | 归因合法率 |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for s in scs:
        st_ = appraisal_stats(records, s)
        h = st_["hits"]
        b = st_["arousal_bias"]
        L.append(f"| `{s}` | {st_['arousal_mae']:.2f} | {b:+.2f} | "
                 f"{arousal_bias_after_offset(records, s, b):.2f} | "
                 f"{_pct(_ratio(h['arousal']))} | "
                 f"{_pct(st_['arousal_exact'])} | {_pct(_ratio(h['valence']))} | "
                 f"{_pct(_ratio(h['intent']))} | {_pct(_ratio(h['evidence']))} |")
    L.append("")
    L.append("> **带符号偏差先看**：MAE 相同但「系统性偏高 1.5 分」与「随机抖动 ±1.5」是两回事。"
             "前者扣一个常数就能救回大半（右列），后者只能靠换模型或补上下文。"
             "所以本表把偏差与扣偏差后的 MAE 一起给，读者能自己判断这个误差是可校正的还是不可校正的。")
    L.append("")
    if "judge_cue" in scs:
        L.append("### ⚠️ judge_cue 的分数不能当能力证据")
        L.append("")
        L.append("`judge_cue` 把参考实现的关键词表原样写进了提示词，"
                 "所以模型只要正确抄表就能拿满分。它的 arousal 命中率是**上界**，"
                 "证明的是「这张表能被正确执行」，不是「模型会读情绪」。"
                 "真正有意义的是 `judge_plain`：不给表，它能把 arousal 读到什么程度。")
        L.append("")
        L.append("这不是设计缺陷，是刻意设置的对照：如果 `judge_cue` 都上不去，"
                 "说明连抄表都做不到，那时才该怀疑提示词；它上去了，"
                 "就把问题干净地收敛到判读能力本身。")
        L.append("")
    for s in scs:
        st_ = appraisal_stats(records, s)
        if st_["per_scene"]:
            per = {kk: [sum(vv) / len(vv)] for kk, vv in st_["per_scene"].items()}
            L.append(f"- `{s}` 的 arousal 行级命中率（按场景聚类 95% CI）："
                     f"{_ci(metrics.cluster_bootstrap_ci(per))}")
    L.append("")
    L.append("> ⚠️ 金标准 arousal 来自 `reference.appraise()` 的**关键词表**（9 个桶 + 默认值）。"
             "所以「命中」的意思是「和我的规则表一致」，不是「判对了」。"
             "要把它变成能力结论，必须补人工标注 + κ。这是本实验最大的外部效度缺口。")
    L.append("")

    # ---------------- 2 ----------------
    L.append("## 2. 端到端对比（同一判卷、同一金标准）")
    L.append("")
    rows = []
    if baseline:
        for s in ("card_injected", "card_formula"):
            if any(r["scaffold"] == s for r in baseline):
                rows.append(("单层 · " + s, baseline, s))
    for s in scs:
        rows.append(("两层 · " + s, records, s))
    L.append("| 配置 | 场景平均命中率 | 聚类 95% CI | 行级 pass@τ | 每行错字段 | 原始 pass@1 |")
    L.append("|---|---|---|---|---|---|")
    for label, src, s in rows:
        vals = [r["hit_rate"] for r in src if r["scaffold"] == s]
        if not vals:
            continue
        _, _, med, _, _ = quantiles(vals)
        L.append(f"| {label} | {med:.2f} | {_ci(scene_mean_ci(src, s))} | "
                 f"{_pct(line_pass(src, s, tau))} | {avg_wrong_fields(src, s):.2f} | "
                 f"{_pct(sum(1 for r in src if r['scaffold'] == s and r['passed']) / max(1, len([r for r in src if r['scaffold'] == s])))} |")
    L.append("")
    L.append(f"> 「原始 pass@1」是 τ={tau}、每场每行都过那个口径——"
             "在第 2 节它变成了一列参考，而不是主指标。")
    L.append("")

    # ---------------- 3 ----------------
    L.append("## 3. 残差归因：判对 vs 判错")
    L.append("")
    L.append("| 臂 | arousal 判对的行：端到端命中率 | arousal 判错的行：端到端命中率 | 行数 (对/错) |")
    L.append("|---|---|---|---|")
    for s in scs:
        e = end_to_end_by_arousal(records, s)
        L.append(f"| `{s}` | {e['arousal_ok'][0]:.3f} | {e['arousal_bad'][0]:.3f} | "
                 f"{e['arousal_ok'][1]}/{e['arousal_bad'][1]} |")
    L.append("")
    L.append("**读法**：判对时端到端接近 1.0，说明 7 个派生参数没有引入任何新误差——"
             "它们不是「模型预测得准」，而是「按公式算出来的」。"
             "这是构造正确性，不该被报成准确率；报成准确率是这类项目最常见的口径作弊。")
    L.append("")

    # ---------------- 4 ----------------
    sb = split_breakdown(records, scs, tau)
    if sb:
        L.append("## 4. 留出集检验（按作品切分）")
        L.append("")
        L.append("| 臂 | 划分 | 场景数 | 场景平均命中率 | 聚类 95% CI | 每行错字段 | arousal MAE | 原始 pass@1 |")
        L.append("|---|---|---|---|---|---|---|---|")
        for row in sb:
            mae = f"{row['mae']:.2f}" if row["mae"] is not None else "—"
            L.append(f"| `{row['scaffold']}` | {row['split']} | {row['scenes']} | "
                     f"{row['median']:.2f} | {_ci(row['ci'])} | {row['wrong']:.2f} | "
                     f"{mae} | {_pct(row['pass1'])} |")
        L.append("")
        L.append("> 留出按 `work_id` 切：同一部作品的角色卡与说话风格相同，"
                 "按场景随机切必然泄漏。test 组是模型在提示词里没见过的作品。"
                 "两组差距小，说明结论没有过拟合到具体戏；差距大则要查提示词是否在背样本。")
        L.append("")

    # ---------------- 5 ----------------
    L.append("## 5. 局限")
    L.append("")
    n_scene = len({r["scene_id"] for r in records if r["scaffold"] == scs[0]})
    n_lines = sum(len(r["rows"]) for r in records if r["scaffold"] == scs[0])
    L.append("- **判读层的金标准是规则，不是人。** 结论只能表述为"
             "「与参考实现的关键词表一致率」，不能表述为「情绪判读准确率」。"
             "补人工标注 + κ 才能变成能力结论。")
    L.append(f"- **样本仍不算大**：{n_scene} 场 / {n_lines} 行。"
             "一场约等于几个百分点，臂间小于这个量级的差距不要下结论。")
    L.append("- **台词库是模板偏置的**：台词按线索桶设计，措辞比真实剧本更「线索密集」，"
             "所以绝对准确率会偏高。它测的是「能不能按线索对齐」，不是「能不能读真实剧本」。")
    L.append("- **valence 不进任何下游计算**（`calibrate` 只用 arousal），"
             "所以它只影响导演备注；把它算进「判读综合分」会虚高。")
    L.append("")
    return "\n".join(L)


def _ratio(pair: tuple[int, int]) -> float:
    h, n = pair
    return h / n if n else 0.0
