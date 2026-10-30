"""金标准参考实现（deterministic gold）。

从 manga-performance-director/director/ 原样移植，**不引入任何 LLM 调用**。
移植的意义：本项目与论文的表演标定共用同一套参数定义，
论文的实验数据与这里的评测数据可以直接互相比对（立项方案里的"数据互通"）。

已知的方法论风险（必须主动交代，不要等面试官问）
-----------------------------------------------
金标准 = 参考实现，而参考实现里的常数（0.60+0.40·表现欲、0.9+0.50·intensity …）
是导演常识而非拟合结果。所以它不是一个"客观正确答案"，而是
**一个可追溯、可复算、可被替换的基线**。本项目的判卷因此分三层：
规则层（不依赖金标准）、容差层（对金标准，容差是显式设计变量）、
以及后续的 VLM 层（独立证据源）。报告里必须报告容差敏感性。
"""

from __future__ import annotations

# --------------------------------------------------------------------------
# 台词线索表（移植自 director.py，未改动）
# --------------------------------------------------------------------------
CUE_TABLE = [
    (["滚", "闭嘴", "你敢", "最后一遍", "再说一次"], -0.85, 8.6, "威胁"),
    (["对不起", "是我的错", "我认"], -0.4, 5.2, "认错"),
    (["别走", "等等", "求你"], -0.6, 7.4, "挽留"),
    (["我喜欢你", "想你", "别怕", "我在"], 0.75, 6.2, "表白/安抚"),
    (["嗯", "知道了", "随便", "随你"], 0.0, 4.0, "隐忍"),
    (["哈哈", "笑死", "真的假的"], 0.6, 6.8, "玩笑"),
    (["哇", "好可爱", "太棒", "天哪", "好美", "好厉害"], 0.85, 7.2, "惊喜"),
    (["死", "血", "来不及", "快跑"], -0.9, 9.0, "危机"),
    (["为什么", "凭什么", "你说什么"], -0.5, 7.8, "质疑"),
]

SCENE_PROFILE = {
    "冲突对峙": {"ceiling": 8.0, "space": "公共", "allow_shout": True},
    "亲密低语": {"ceiling": 4.5, "space": "私密", "allow_shout": False},
    "独白": {"ceiling": 5.5, "space": "私密", "allow_shout": False},
    "危机": {"ceiling": 9.0, "space": "公共", "allow_shout": True},
    "日常": {"ceiling": 5.0, "space": "公共", "allow_shout": False},
    "玩笑打闹": {"ceiling": 7.5, "space": "私密", "allow_shout": False},
}

RELATION_MOD = {
    "长辈": -1.2, "上级": -1.0, "陌生人": -0.6,
    "挚友": +0.4, "恋人": +0.3, "恋人未满": -0.2, "对手": +0.2,
}

# 禁则的硬阈值：allow_shout=False 的场景里，强度与音高范围的上限
SHOUT_CAP_INTENSITY = 6.5
SHOUT_CAP_F0_RANGE = 5.0
NON_SHOUT_STYLES = ("压抑", "克制", "自然")


# --------------------------------------------------------------------------
# 第一层：内部情绪
# --------------------------------------------------------------------------
def personality_baseline(p: dict) -> dict:
    warmth = p.get("warmth", 0.5)
    dominance = p.get("dominance", 0.5)
    energy = p.get("energy", 0.5)
    neuro = p.get("neuroticism", 0.5)
    expressiveness = 0.3 * energy + 0.3 * dominance + 0.2 * neuro + 0.2 * warmth
    return {
        "expressiveness": round(expressiveness, 3),
        "intensity_cap": round(3.5 + 4.5 * expressiveness, 2),
        "rate_base": round(3.6 + 1.2 * energy, 2),
        "f0_base": round(-1.5 + 3.0 * (1 - dominance) * 0.5, 2),
    }


def appraise(text: str, subtext: str = "") -> dict:
    blob = text + " " + subtext
    for cues, val, aro, intent in CUE_TABLE:
        if any(c in blob for c in cues):
            if any(w in subtext for w in ("心跳", "其实", "忍住", "怕", "紧张")):
                aro = min(10.0, aro + 1.2)
            return {"valence": val, "arousal": round(aro, 2), "intent": intent}
    if any(w in subtext for w in ("心跳", "怕", "紧张", "犹豫")):
        return {"valence": -0.1, "arousal": 6.2, "intent": "忐忑"}
    return {"valence": 0.0, "arousal": 5.0, "intent": "陈述"}


# --------------------------------------------------------------------------
# 第二层：外部表演（抑制校准 —— 论文的方法贡献）
# --------------------------------------------------------------------------
def calibrate(internal: dict, baseline: dict, scene: dict, relation: str) -> dict:
    arousal = internal["arousal"]
    expressiveness = baseline["expressiveness"]

    expressive_gain = 0.60 + 0.40 * expressiveness
    cap_personality = baseline["intensity_cap"]
    cap_scene = scene["ceiling"] + RELATION_MOD.get(relation, 0.0)

    raw = arousal * expressive_gain
    intensity = min(raw, cap_personality, cap_scene)
    if not scene["allow_shout"]:
        intensity = min(intensity, SHOUT_CAP_INTENSITY)

    intensity = round(max(1.0, intensity), 2)
    suppression = round(intensity / max(arousal, 0.1), 3)
    withheld = round(max(0.0, arousal - intensity), 2)

    f0_range = round(min(6.0, 0.9 + 0.50 * intensity), 2)
    f0_offset = round(baseline["f0_base"] + 0.45 * (intensity - 4.0), 2)
    rate = round(baseline["rate_base"] * (0.85 + 0.035 * intensity), 2)
    pause_after = int(180 + (10 - intensity) * 35 + (260 if withheld >= 2.0 else 0))

    if withheld >= 2.0 and intensity < 5.0:
        style = "压抑"
        au = {"AU4": 0.75, "AU20": 0.10, "AU26": 0.08, "AU12": 0.05, "AU17": 0.55}
        gaze = "避开对视，短暂下移"
    elif withheld >= 2.0:
        style = "克制"
        au = {"AU4": 0.80, "AU20": 0.35, "AU26": 0.25, "AU12": 0.08, "AU17": 0.45}
        gaze = "直视但眼神收紧"
    elif intensity >= 7.0:
        style = "外放"
        au = {"AU4": 0.55, "AU20": 0.70, "AU26": 0.55, "AU12": 0.15, "AU17": 0.20}
        gaze = "直视对方"
    else:
        style = "自然"
        au = {"AU4": 0.35, "AU20": 0.20, "AU26": 0.15, "AU12": 0.30, "AU17": 0.10}
        gaze = "自然对视"

    return {
        "intensity": intensity,
        "suppression": suppression,
        "withheld": withheld,
        "f0_offset": f0_offset,
        "f0_range": f0_range,
        "rate": rate,
        "pause_after_ms": pause_after,
        "expression": {"style": style, "au": au, "gaze": gaze},
    }


# --------------------------------------------------------------------------
# 对外：一条台词 -> 金标准镜头
# --------------------------------------------------------------------------
def required_constraints(scene_type: str) -> list[str]:
    """场景规范推出的必守禁则。allow_shout=False 的场景一律不许喊。"""
    prof = SCENE_PROFILE.get(scene_type, SCENE_PROFILE["日常"])
    return [] if prof["allow_shout"] else ["no_shout"]


def director_note(intensity: float, suppression: float, withheld: float,
                  scene: dict, intent: str) -> str:
    if withheld >= 2.0:
        tail = "眼神和呼吸" if withheld >= 3 else "句尾的停顿"
        return f"只演 {intensity:.1f} 分（抑制系数 {suppression:.2f}）——不要喊，把力气留在{tail}上。"
    if scene["space"] == "私密" and intensity >= 4.0:
        return f"私密场景，音量压低、气声为主（抑制系数 {suppression:.2f}）；情绪靠停顿传递。"
    if intensity >= 7.0:
        return f"可以外放（抑制系数 {suppression:.2f}），但守住音高上限，不要破音。"
    return f"内心与外在基本一致（{suppression:.2f}），自然表达即可。"


def gold_shot(line: dict, characters: dict) -> dict:
    """确定性金标准：输入一条台词，输出契约格式的镜头。"""
    ch = characters[line["character"]]
    baseline = personality_baseline(ch.get("personality", {}))
    internal = appraise(line["text"], line.get("subtext", ""))
    scene_type = line.get("scene_type", "日常")
    scene = SCENE_PROFILE.get(scene_type, SCENE_PROFILE["日常"])
    relation = line.get("relation", "陌生人")

    cal = calibrate(internal, baseline, scene, relation)
    return {
        "line_id": line["id"],
        "character": line["character"],
        "intensity": cal["intensity"],
        "suppression": cal["suppression"],
        "f0_offset": cal["f0_offset"],
        "f0_range": cal["f0_range"],
        "rate": cal["rate"],
        "pause_after_ms": cal["pause_after_ms"],
        "expression": cal["expression"],
        "constraints": required_constraints(scene_type),
        "evidence": line["text"],
        "director_note": director_note(
            cal["intensity"], cal["suppression"], cal["withheld"], scene,
            internal["intent"],
        ),
        # 以下字段不参与判卷，仅供分析与 Mock 使用
        "_internal": internal,
        "_baseline": baseline,
        "_withheld": cal["withheld"],
        "_naive_intensity": internal["arousal"],
    }


JUDGED_NUMERIC = ("intensity", "f0_offset", "f0_range", "rate", "pause_after_ms")
JUDGED_FIELDS = JUDGED_NUMERIC + ("style", "gaze", "au")
