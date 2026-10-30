"""报告层：把 records 变成一份可读、可复核的 Markdown。

报告的第一原则：**每个数字都要能被追问到来源。**
所以这里固定输出四类"防身"内容：
1. 口径声明（k、tau、容差、模型、切分方式）；
2. 主表带聚类 CI（不是裸点估计）；
3. tau 与容差敏感性（回答"换个阈值结论还成立吗"）；
4. 自动诊断（从数字推出来的话，不是预先写好的话）。
"""

from __future__ import annotations

import datetime as _dt
from collections import Counter

from . import judge, metrics, schema
from . import reference as ref


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _fmt_ci(ci, as_pct: bool = True) -> str:
    lo, hi, pt = ci
    if as_pct:
        return f"{pt * 100:.1f}% [{lo * 100:.1f}, {hi * 100:.1f}]"
    return f"{pt:.3f} [{lo:.3f}, {hi:.3f}]"


def render(records: list[dict], meta: dict) -> str:
    scaffolds = meta["scaffolds"]
    k = meta["k"]
    tau = meta.get("tau", judge.DEFAULT_TAU)
    L: list[str] = []

    L.append(f"# 漫剧导演 Agent · 可靠性评测报告")
    L.append("")
    L.append(f"> 生成时间：{_dt.datetime.now().strftime('%Y-%m-%d %H:%M')}　"
             f"模式：**{meta.get('mode', 'mock')}**　模型：`{meta.get('model', '-')}`")
    if meta.get("mode") == "mock":
        L.append(">")
        L.append("> ⚠️ **本报告是 Mock 校准运行，数字不是实验结果。** 它只用来验证"
                 "判卷、指标与编排链路是否正确：Mock 被注入了四类已知缺陷，"
                 "若这里跑不出下文「自动诊断」里的已知结论，说明管线有 bug。")
    L.append("")

    # ---------------- 口径 ----------------
    L.append("## 0. 口径声明")
    L.append("")
    L.append(f"- 任务数：**{meta['n_tasks']}** 场戏 / **{meta['n_lines']}** 条台词；"
             f"重复次数 k = **{k}**（每场戏独立跑 k 次）")
    L.append(f"- 场景级 pass 判据：L1 全清 **且** 参数命中率 ≥ τ = **{tau}**")
    L.append("- 参数命中率 = 8 个受判字段的命中比例："
             "intensity / f0_offset / f0_range / rate / pause_after_ms / style / gaze / AU")
    L.append(f"- 数值容差：{', '.join(f'{k}={v}' for k, v in judge.DEFAULT_TOL.items())}")
    L.append("- 区间估计：按**场景**聚类的 Bootstrap 95% CI（B=2000）；"
             "k 次重复不当作样本量")
    if meta.get("split"):
        L.append(f"- 数据切分：{meta['split']}")
    L.append("")

    # ---------------- 主表 ----------------
    L.append("## 1. 主结果")
    L.append("")
    L.append("| scaffold | pass@1 (95%CI) | pass^k | 行级 pass | 平均命中率 | L1 全清率 | 禁令违反率 | tokens/场 | 平均轮数 |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for s in scaffolds:
        ci = metrics.pass_ci(records, s)
        L.append(
            f"| `{s}` | {_fmt_ci(ci)} | {_pct(metrics.pass_pow_k(records, s))} | "
            f"{_pct(metrics.line_level_pass(records, s, tau))} | "
            f"{metrics.mean_hit_rate(records, s):.3f} | "
            f"{_pct(metrics.l1_clean_rate(records, s))} | "
            f"{_pct(metrics.violation_rate(records, s))} | "
            f"{metrics.mean_tokens(records, s):.0f} | "
            f"{metrics.mean_attempts(records, s):.1f} |"
        )
    L.append("")
    L.append("> `pass^k` 是每场戏 k 次全过才算过；`行级 pass` 不看场景连乘。"
             "两者与 pass@1 的落差，就是「单次能力」与「生产可靠性」的距离。")
    L.append("")

    # ---------------- 逐字段误差 ----------------
    L.append("## 2. 逐字段误差（相对参考实现）")
    L.append("")
    fields = list(ref.JUDGED_NUMERIC)
    L.append("| scaffold | " + " | ".join(fields) + " | AU-MAE |")
    L.append("|---" * (len(fields) + 2) + "|")
    for s in scaffolds:
        m = metrics.mae_by_field(records, s)
        au = [r["rows"] and [row.get("deltas", {}).get("au", 0.0) for row in r["rows"]] for r in records
              if r["scaffold"] == s]
        flat_au = [x for sub in au for x in (sub or [])]
        au_mae = sum(flat_au) / len(flat_au) if flat_au else 0.0
        L.append(f"| `{s}` | " + " | ".join(f"{m[f]:.2f}" for f in fields) + f" | {au_mae:.2f} |")
    L.append("")
    L.append(f"> 容差阈值：{', '.join(f'{f}≤{judge.DEFAULT_TOL[f]}' for f in fields)}。"
             "误差超过容差即该字段不算命中。")
    L.append("")

    # ---------------- 敏感性 ----------------
    L.append("## 3. 敏感性：结论是不是被我调阈值调出来的")
    L.append("")
    L.append("### 3.1 τ（命中率门槛）")
    L.append("")
    taus = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
    L.append("| scaffold | " + " | ".join(f"τ={t}" for t in taus) + " |")
    L.append("|---" * (len(taus) + 1) + "|")
    for s in scaffolds:
        ts = metrics.tau_sensitivity(records, s, taus)
        L.append(f"| `{s}` | " + " | ".join(f"{ts[t]['pass@1'] * 100:.0f}%" for t in taus) + " |")
    L.append("")

    L.append("### 3.2 容差缩放")
    L.append("")
    scales = (0.5, 0.75, 1.0, 1.5, 2.0)
    L.append("| scaffold | " + " | ".join(f"×{s}" for s in scales) + " |")
    L.append("|---" * (len(scales) + 1) + "|")
    for s in scaffolds:
        row = [metrics.tol_scale_sensitivity(records, s, sc)["pass@1"] for sc in scales]
        L.append(f"| `{s}` | " + " | ".join(f"{v * 100:.0f}%" for v in row) + " |")
    L.append("")

    # ---------------- 禁令分解 ----------------
    L.append("## 4. 禁令违反分解")
    L.append("")
    cnt = Counter()
    for r in records:
        for v in r["violations"]:
            cnt[(r["scaffold"], v["constraint"])] += 1
    if cnt:
        L.append("| scaffold | 禁令 | 违反行数 |")
        L.append("|---|---|---|")
        for (s, c), n in sorted(cnt.items(), key=lambda x: (-x[1], x[0])):
            L.append(f"| `{s}` | {c}（{schema.CONSTRAINT_LABEL.get(c, c)}） | {n} |")
    else:
        L.append("本次运行没有出现禁令违反。")
    L.append("")

    # ---------------- 自动诊断 ----------------
    L.append("## 5. 自动诊断")
    L.append("")
    L.extend(_diagnose(records, scaffolds, k, tau))
    L.append("")

    L.append("---")
    L.append("")
    L.append("## 附：失败样例（随机抽 3 条，供人工复核判卷是否判对了）")
    L.append("")
    # 挑样例时刻意分散到不同 scaffold 与不同场景，否则会看到三遍同一格
    cands: list[tuple[dict, dict]] = []
    for r in records:
        if r["passed"]:
            continue
        bad = [row for row in r["rows"] if row["errors"] or row["violations"] or row["hit_rate"] < tau]
        if bad:
            cands.append((r, bad[0]))

    picked: list[tuple[dict, dict]] = []
    used_sc, used_sf = set(), set()
    for r, row in cands:
        if r["scene_id"] in used_sc or r["scaffold"] in used_sf:
            continue
        picked.append((r, row))
        used_sc.add(r["scene_id"])
        used_sf.add(r["scaffold"])
        if len(picked) >= 3:
            break
    for c in cands:
        if len(picked) >= 3:
            break
        if c not in picked:
            picked.append(c)

    for r, row in picked[:3]:
        L.append(f"**{r['scaffold']} / {r['scene_id']} / rep{r['repeat']}** "
                 f"（该行命中率 {row['hit_rate']:.2f}）")
        if row["errors"]:
            L.append(f"- L1 错误：{'；'.join(row['errors'][:4])}")
        if row["violations"]:
            L.append(f"- 禁令违反：{'；'.join(v['detail'] for v in row['violations'][:2])}")
        if row.get("deltas"):
            L.append(f"- 字段误差：{', '.join(f'{kk}={vv}' for kk, vv in row['deltas'].items())}")
        L.append("")

    return "\n".join(L)


def _diagnose(records: list[dict], scaffolds: list[str], k: int, tau: float) -> list[str]:
    """从数字推结论。每条都带"触发条件"，便于复核。"""
    out: list[str] = []

    # 1) 可靠性缺口：只在"单次已经像能用"的配置里找，否则措辞会失真
    gaps = []
    for s in scaffolds:
        p1 = metrics.pass_at_1(records, s)
        pk = metrics.pass_pow_k(records, s)
        gaps.append((s, p1, pk, p1 - pk))
    usable = [g for g in gaps if g[1] >= 0.5] or gaps
    s, p1, pk, gap = max(usable, key=lambda x: x[3])
    if gap > 0.10:
        if p1 >= 0.7:
            out.append(f"1. **可靠性缺口（`{s}`）**：pass@1 = {_pct(p1)} 看起来可以在灰度里用，"
                       f"但 pass^{k} = {_pct(pk)}，缺口 {gap * 100:.0f} 个百分点。"
                       f"一次过和每次过是两个产品；上线门槛应该按 pass^k 定，不是 pass@1。")
        else:
            out.append(f"1. **可靠性缺口（`{s}`）**：pass@1 = {_pct(p1)}，"
                       f"pass^{k} = {_pct(pk)}。单次就已经不稳，重复 k 次只会把不稳放大。")

    # 2) 场景连乘
    worst = max(scaffolds, key=lambda s: metrics.line_level_pass(records, s, tau)
                - metrics.pass_at_1(records, s))
    lp = metrics.line_level_pass(records, worst, tau)
    p1w = metrics.pass_at_1(records, worst)
    if lp - p1w > 0.10:
        out.append(f"2. **场景级连乘（`{worst}`）**：行级 pass {_pct(lp)}，"
                   f"场级 pass 只有 {_pct(p1w)}（差 {(lp - p1w) * 100:.0f} 个百分点）。"
                   f"参照算一下：行级 95%、一场 4 行 → 场级约 {(0.95 ** 4) * 100:.0f}%。"
                   f"这就是必须按场报可靠性的理由。")

    # 3) 禁令
    v = {s: metrics.violation_rate(records, s) for s in scaffolds}
    if v:
        worst_v = max(v, key=v.get)
        if v[worst_v] > 0.05:
            out.append(f"3. **禁令违反（`{worst_v}`：{_pct(v[worst_v])} 的行）**："
                       f"说明「不要喊」这类约束写在自然语言里是无效的，"
                       f"必须升级为结构化字段 + 硬阈值才可执行、可计数。")

    # 4) 公式消融
    if "card_injected" in v and "card_formula" in v:
        a = metrics.pass_at_1(records, "card_injected")
        b = metrics.pass_at_1(records, "card_formula")
        if b - a > 0.15:
            out.append(f"4. **给出公式才对齐（{_pct(a)} → {_pct(b)}）**：同一份角色卡下，"
                       f"把参考算法原样交出，通过率跳升 {(b - a) * 100:.0f} 个百分点。"
                       f"说明模型的短板不在「写参数」，而在「从台词读出内心情绪」——"
                       f"方法论含义是：这层能力该用检索/规则补，而不是指望 prompt。")

    # 5) 修复的性价比
    if "card_repair" in v and "card_injected" in v:
        a = metrics.pass_at_1(records, "card_injected")
        b = metrics.pass_at_1(records, "card_repair")
        ta, tb = metrics.mean_tokens(records, "card_injected"), metrics.mean_tokens(records, "card_repair")
        if b > a and ta > 0:
            out.append(f"5. **修复轮的性价比**：通过率 {_pct(a)} → {_pct(b)}，"
                       f"代价是 tokens/场 {ta:.0f} → {tb:.0f}"
                       f"（+{(tb / ta - 1) * 100:.0f}%）。"
                       f"修复反馈只回传 L1 规则错误，不回传金标准数值——否则就是对着答案改错。")

    # 6) 敏感性：挑跨度最大的那个 scaffold 说
    spans = []
    for s in scaffolds:
        ts = metrics.tau_sensitivity(records, s)
        vals = [v["pass@1"] for v in ts.values()]
        spans.append((s, max(vals) - min(vals)))
    s_max, span = max(spans, key=lambda x: x[1])
    out.append(f"6. **阈值敏感性**：τ 从 0.5 扫到 1.0，`{s_max}` 的 pass@1 "
               f"跨度 {span * 100:.0f} 个百分点。报告结论时必须给这条误差带，"
               f"否则「我做到 85%」这种说法在换一个 τ 之后就会被打穿。")

    return out
