"""判卷器：L1 规则层 + L2 容差层（部分分）。

分层原则（面试可讲的第一个设计决策）
----------------------------------
**能被规则确定性判定的，绝不交给模型或金标准。**
- L1 只看"契约本身"：结构、区间、归因（evidence 是否原文）、禁则是否被执行。
  这一层不需要金标准，因此不受"金标准是不是你自己"这一质疑影响。
- L2 才和金标准比：数值落容差、分类判等。L2 天然是"部分分"，
  用参数级命中率而不是 0/1，这样"模型大部分字段对、只是强度偏高"能被看见。

一遍 pass 的定义（pass^k 的原子事件）
------------------------------------
    pass = L1 全清 AND 参数命中率 >= tau

tau 是**显式设计变量**，不是 0.8 这个"看起来合理"的数。
因此报告必须给 tau 与容差的敏感性曲线——结论要能扛住"你换个阈值还成立吗"。
"""

from __future__ import annotations

from . import reference as ref
from . import schema

# --------------------------------------------------------------------------
# 容差：显式设计变量。注释里给的是"可辩护的理由"，不是实证结论。
# --------------------------------------------------------------------------
DEFAULT_TOL = {
    "intensity": 0.8,       # 10 分制下约"半档"，导演术语里 1 档 ≈ 1 分
    "f0_offset": 1.0,       # 1 个半音，约等于人耳对音高偏移的可辨阈
    "f0_range": 1.0,        # 1 个半音
    "rate": 0.5,            # 字/秒，约 8% 相对误差
    "pause_after_ms": 200,  # 0.2 秒，短于感知上"明显停顿"的差异
    "au": 0.25,             # AU 强度平均绝对误差
}

DEFAULT_TAU = 0.8

# 判读层容差：与 8 字段容差同理，是显式设计变量。
# arousal 1.0 = 10 分制下的"一档"，与 intensity 的 0.8 同量级；
# valence 0.3 是因为金标准的 valence 只有 9 个离散取值，0.3 约等于相邻两档。
APPRAISAL_TOL = {"arousal": 1.0, "valence": 0.3}


def _num_hit(actual: float, gold: float, tol: float) -> bool:
    return abs(actual - gold) <= tol


def judge_appraisal(appr: dict, line: dict, gold: dict,
                    tol: dict | None = None) -> dict:
    """判一条**判读**结果（两层架构的第一层）。

    只比 3 个语义量 + 归因有效性，不比任何派生参数——
    派生参数在两层架构里由代码算出，比它等于比自己的函数，没有信息量。

    金标准取 `gold["_internal"]`（reference.appraise 的输出），
    注意它本身是关键词表的产物，所以这里的"命中"只意味着
    "与我的规则表一致"，不等于"判对了"。报告里必须说明。
    """
    tol = {**APPRAISAL_TOL, **(tol or {})}
    internal = gold.get("_internal") or {}
    hits: dict[str, bool] = {}
    deltas: dict[str, float] = {}
    signed: dict[str, float] = {}

    if not isinstance(appr, dict):
        return {"hits": {}, "deltas": {}, "signed": {}, "errors": ["判读结果不是对象"]}

    for k in ("arousal", "valence"):
        g = internal.get(k)
        v = appr.get(k)
        if g is None or not isinstance(v, (int, float)) or isinstance(v, bool):
            hits[k] = False
            continue
        d = abs(float(v) - float(g))
        deltas[k] = round(d, 4)
        # 带符号偏差：MAE 相同但"系统性偏高 1.5 分"与"随机抖动 ±1.5"是两回事，
        # 前者可以靠常数校正，后者不能。必须分开报。
        signed[k] = round(float(v) - float(g), 4)
        hits[k] = d <= tol[k]

    hits["intent"] = appr.get("intent") == internal.get("intent")
    hits["evidence"] = not schema.check_evidence(appr.get("evidence", ""), line["text"])

    return {"hits": hits, "deltas": deltas, "signed": signed, "errors": []}


def judge_line(shot: dict, line: dict, gold: dict, tol: dict | None = None) -> dict:
    """判一条台词。返回逐字段命中、L1 错误、禁令违反。"""
    tol = {**DEFAULT_TOL, **(tol or {})}
    errors: list[str] = []
    violations: list[dict] = []
    hits: dict[str, bool] = {}

    # ---------------- L1a：结构 ----------------
    errors.extend(schema.validate_shot(shot))
    structural_ok = not errors

    if structural_ok:
        # ---------------- L1b：身份对齐 ----------------
        if shot["line_id"] != line["id"]:
            errors.append(f"line_id 应为 {line['id']}，得到 {shot['line_id']}")
        if shot["character"] != line["character"]:
            errors.append(f"character 应为 {line['character']}，得到 {shot['character']}")

        # ---------------- L1c：归因（幻觉检测） ----------------
        errors.extend(schema.check_evidence(shot["evidence"], line["text"]))

        # ---------------- L1d：禁令覆盖与执行 ----------------
        required = ref.required_constraints(line.get("scene_type", "日常"))
        declared = set(shot["constraints"])
        for c in required:
            if c not in declared:
                errors.append(f"漏声明禁令 {c}")

        for c in required:
            if c == "no_shout":
                bad = []
                if shot["intensity"] > ref.SHOUT_CAP_INTENSITY:
                    bad.append(f"intensity={shot['intensity']} > {ref.SHOUT_CAP_INTENSITY}")
                if shot["f0_range"] > ref.SHOUT_CAP_F0_RANGE:
                    bad.append(f"f0_range={shot['f0_range']} > {ref.SHOUT_CAP_F0_RANGE}")
                style = shot["expression"]["style"]
                if style not in ref.NON_SHOUT_STYLES:
                    bad.append(f"style={style}（该场景不许外放）")
                if bad:
                    violations.append({
                        "line_id": line["id"],
                        "constraint": c,
                        "detail": "；".join(bad),
                    })

    # ---------------- L2：容差层（结构不合法就不判 L2） ----------------
    deltas: dict[str, float] = {}
    if structural_ok:
        for k in ref.JUDGED_NUMERIC:
            d = abs(float(shot[k]) - float(gold[k]))
            deltas[k] = round(d, 4)
            hits[k] = d <= tol[k]
        hits["style"] = shot["expression"]["style"] == gold["expression"]["style"]
        hits["gaze"] = shot["expression"]["gaze"] == gold["expression"]["gaze"]

        au_a, au_g = shot["expression"]["au"], gold["expression"]["au"]
        mae = sum(abs(float(au_a.get(k, 0.0)) - au_g[k]) for k in schema.AU_KEYS) / len(schema.AU_KEYS)
        deltas["au"] = round(mae, 4)
        hits["au"] = mae <= tol["au"]

    hit_rate = sum(hits.values()) / len(ref.JUDGED_FIELDS) if hits else 0.0

    return {
        "line_id": line["id"],
        "errors": errors,
        "violations": violations,
        "hits": hits,
        "deltas": deltas,
        "hit_rate": round(hit_rate, 4),
        "l1_clean": not errors and not violations,
        "shot": shot,
        "gold": gold,
    }


def judge_scene(output: dict, task: dict, golds: dict[str, dict],
                tol: dict | None = None, tau: float = DEFAULT_TAU) -> dict:
    """判一整场戏。

    场景级 pass = **每一行**都 L1 全清且命中率 >= tau。
    这里刻意不用"平均命中率 >= tau"：平均会让一条演崩的台词
    被其他几条好台词稀释掉——而观众看到的恰恰是那一条。
    pass^k 要度量的是"每次都对"，任何形式的平均都是在给可靠性注水。
    """
    errors: list[str] = []
    lines = task["lines"]

    shots = output.get("shots") if isinstance(output, dict) else None
    if not isinstance(shots, list):
        return {
            "scene_id": task["scene_id"],
            "passed": False,
            "tau": tau,
            "hit_rate": 0.0,
            "min_hit_rate": 0.0,
            "errors": ["输出缺少 shots 数组"],
            "rows": [],
            "violations": [],
            "l1_clean": False,
        }

    by_id = {}
    for s in shots:
        if isinstance(s, dict) and isinstance(s.get("line_id"), str):
            by_id.setdefault(s["line_id"], s)

    rows = []
    expected_ids = [ln["id"] for ln in lines]
    for ln in lines:
        if ln["id"] not in by_id:
            errors.append(f"缺镜头 {ln['id']}")
            rows.append({
                "line_id": ln["id"], "errors": ["缺镜头"], "violations": [],
                "hits": {}, "deltas": {}, "hit_rate": 0.0, "l1_clean": False,
                "shot": None, "gold": golds[ln["id"]],
            })
            continue
        rows.append(judge_line(by_id[ln["id"]], ln, golds[ln["id"]], tol))

    extra = set(by_id) - set(expected_ids)
    for e in sorted(extra):
        errors.append(f"多余镜头 {e}")

    rates = [r["hit_rate"] for r in rows]
    hit_rate = sum(rates) / len(rates) if rates else 0.0
    min_hit = min(rates) if rates else 0.0
    l1_clean = all(r["l1_clean"] for r in rows) and not errors
    every_line_ok = all(r["l1_clean"] and r["hit_rate"] >= tau for r in rows)
    violations = [v for r in rows for v in r["violations"]]

    return {
        "scene_id": task["scene_id"],
        "passed": bool(every_line_ok and not errors),
        "tau": tau,
        "hit_rate": round(hit_rate, 4),
        "min_hit_rate": round(min_hit, 4),
        "errors": errors,
        "rows": rows,
        "violations": violations,
        "l1_clean": l1_clean,
    }
