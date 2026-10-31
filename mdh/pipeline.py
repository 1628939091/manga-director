"""两层流水线：LLM 只做判读，标定交回确定性代码。

架构
----
    Agent 输出：  arousal / valence / intent / evidence      ← 判读（模型做）
    代码计算：    intensity / suppression / f0_* / rate / pause / style / AU / gaze
                  ← 标定（reference.calibrate 做）

为什么这么切
------------
8 个受判字段里，只有 arousal 是"读台词读出来的"，其余 7 个都是
`arousal × 性格四维 × 场景上限 × 关系修正` 的确定性函数。
把这个函数交给 LLM 执行，等于让语言模型当计算器：
- 算术字段能到 4–8 成，但永远进不了容差；
- 语义字段（style/gaze）停在四选一的随机水平；
- 而且 AU 是"按 style 分支查表"，style 一错，AU 从 7 成退化到 4 成
  （见 outputs/analysis_real_v32.md 第 4 节，2.6 倍级联）。

这是"给更精确的指令反而多了一条失败路径"的机制解释，
也是把标定层移出模型的直接依据。

纪律
----
1. **不修改 reference.calibrate**：金标准保持单一实现，
   否则判卷基准会跟着架构漂移，前后不可比。
2. **组合出来的镜头仍走同一个 judge_scene**：所以端到端指标可比，
   差值只来自"谁在执行标定"，不来自判卷口径变化。
3. **越界一律夹紧并计数**，不静默修正——静默修正会让 L1 通过率虚高。
"""

from __future__ import annotations

from . import reference as ref


def _clamp(v: float, lo: float, hi: float) -> tuple[float, bool]:
    if v < lo:
        return lo, True
    if v > hi:
        return hi, True
    return v, False


def compose_shot(appr: dict, line: dict, characters: dict) -> tuple[dict, list[str]]:
    """把一条判读结果合成为一个契约格式的镜头。返回 (shot, 夹紧记录)。"""
    notes: list[str] = []
    ch = characters[line["character"]]
    baseline = ref.personality_baseline(ch.get("personality", {}))

    scene_type = line.get("scene_type", "日常")
    scene = ref.SCENE_PROFILE.get(scene_type, ref.SCENE_PROFILE["日常"])
    relation = line.get("relation", "陌生人")

    arousal, clamped = _clamp(float(appr.get("arousal", 5.0)), 0.0, 10.0)
    if clamped:
        notes.append(f"arousal 夹紧到 {arousal}")
    valence, clamped = _clamp(float(appr.get("valence", 0.0)), -1.0, 1.0)
    if clamped:
        notes.append(f"valence 夹紧到 {valence}")
    intent = appr.get("intent", "陈述")

    internal = {"arousal": arousal, "valence": valence, "intent": intent}
    cal = ref.calibrate(internal, baseline, scene, relation)

    # calibrate 里 suppression = intensity / max(arousal, 0.1)。
    # arousal 极小时 intensity 被 max(1.0, ...) 抬到 1.0，比值可能越过 1.0，
    # 这会撞上 schema.RANGES 导致整行作废——夹紧并留痕，不静默放过。
    suppression, clamped = _clamp(cal["suppression"], 0.0, 1.0)
    if clamped:
        notes.append(f"suppression 夹紧到 {suppression}")

    shot = {
        "line_id": line["id"],
        "character": line["character"],
        "intensity": cal["intensity"],
        "suppression": round(suppression, 3),
        "f0_offset": cal["f0_offset"],
        "f0_range": cal["f0_range"],
        "rate": cal["rate"],
        "pause_after_ms": cal["pause_after_ms"],
        "expression": cal["expression"],
        "constraints": ref.required_constraints(scene_type),
        "evidence": appr.get("evidence", ""),
        "director_note": ref.director_note(
            cal["intensity"], suppression, cal["withheld"], scene, intent
        ),
    }
    return shot, notes


def compose_scene(appraisal: dict, task: dict) -> dict:
    """把一场戏的判读结果合成为完整输出（与 agents 的单层输出同构）。

    返回的 dict 里额外带 `_appraisal`（模型的原始判读）与 `_clamps`，
    供 runner 记录、供分析层比较判读误差。
    """
    lines = {ln["id"]: ln for ln in task["lines"]}
    raw = appraisal.get("lines") if isinstance(appraisal, dict) else None
    if not isinstance(raw, list):
        raw = appraisal.get("shots") if isinstance(appraisal, dict) else None
    if not isinstance(raw, list):
        return {"scene_id": task["scene_id"], "shots": [], "_appraisal": {},
                "_clamps": ["判读输出缺少 lines 数组"]}

    shots, kept, clamps = [], {}, []
    for item in raw:
        if not isinstance(item, dict) or not isinstance(item.get("line_id"), str):
            continue
        ln = lines.get(item["line_id"])
        if ln is None:
            continue
        shot, notes = compose_shot(item, ln, task["characters"])
        shots.append(shot)
        kept[item["line_id"]] = item
        clamps.extend(f"{item['line_id']}: {n}" for n in notes)

    return {
        "scene_id": task["scene_id"],
        "shots": shots,
        "_appraisal": kept,
        "_clamps": clamps,
    }
