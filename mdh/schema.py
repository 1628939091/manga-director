"""表演参数契约（the contract）。

设计要点
--------
1. **每个决策必须带 evidence**：evidence 必须是台词原文的子串。
   这一条把"参数是否忠于原文"从主观判断变成可确定性校验的规则，
   是整套判卷里性价比最高的机制（幻觉 = evidence 不在原文里）。
2. **每一维参数都有自己的物理区间**：越界即判错，不进入容差比较。
   区间来自 director/schema.py 的 RANGES，与论文保持一致。
3. **禁则是显式字段**，不是藏在备注里的一句自然语言。
   导演备注写"不要喊"，模型照样输出 intensity=9，这是 AI 漫剧的原始病灶；
   把禁则提升为结构化字段后，"违反禁令"才可被计数。

契约（一条台词 = 一个镜头 = 一次表演决策）::

    {
      "line_id":   "S01-L02",
      "character": "沈砚",
      "intensity": 4.1,              # 外部表演强度 0-10
      "suppression": 0.79,           # 抑制系数 = 外部/内部
      "f0_offset": -0.9,             # 相对本人基线的半音偏移
      "f0_range": 3.0,               # 音高动态范围（半音）—— 喊叫的核心特征
      "rate": 4.1,                   # 语速 字/秒
      "pause_after_ms": 620,
      "expression": {
        "style": "压抑",             # 压抑 / 克制 / 自然 / 外放
        "au": {"AU4":0.75,"AU12":0.05,"AU17":0.55,"AU20":0.10,"AU26":0.08},
        "gaze": "避开对视，短暂下移"
      },
      "constraints": ["no_shout"],   # 本镜头生效的禁令
      "evidence": "……我知道了。",     # 必须是台词原文子串
      "director_note": "不要喊，把力气留在句尾的停顿上。"
    }
"""

from __future__ import annotations

import re

# --------------------------------------------------------------------------
# 枚举（封闭集合，便于严格判等；开放式文本无法判等，是评测设计的第一号坑）
# --------------------------------------------------------------------------
STYLES = ("压抑", "克制", "自然", "外放")
GAZES = (
    "直视对方",
    "直视但眼神收紧",
    "避开对视，短暂下移",
    "自然对视",
)
AU_KEYS = ("AU4", "AU12", "AU17", "AU20", "AU26")

# 数值区间：(下界, 上界)，与 director/schema.py 完全一致
RANGES = {
    "intensity": (0.0, 10.0),
    "suppression": (0.0, 1.0),
    "f0_offset": (-4.0, 6.0),
    "f0_range": (0.5, 8.0),
    "rate": (2.0, 9.0),
    "pause_after_ms": (0, 2000),
}

NUMERIC_FIELDS = tuple(RANGES)

REQUIRED_FIELDS = (
    "line_id",
    "character",
    "intensity",
    "suppression",
    "f0_offset",
    "f0_range",
    "rate",
    "pause_after_ms",
    "expression",
    "constraints",
    "evidence",
)

# 禁令 -> 人话（报告里要能直接读）
CONSTRAINT_LABEL = {
    "no_shout": "不许喊：音高与强度必须压住",
    "no_cry": "不许哭腔",
    "no_long_tone": "不许拉长音",
    "no_scream_laugh": "不许大笑出声",
}


def validate_shot(obj: object) -> list[str]:
    """L1a：结构校验。返回错误列表，空列表 = 结构合法。"""
    errs: list[str] = []
    if not isinstance(obj, dict):
        return [f"镜头不是对象，而是 {type(obj).__name__}"]

    for f in REQUIRED_FIELDS:
        if f not in obj:
            errs.append(f"缺字段 {f}")
    if errs:
        return errs

    for k in ("line_id", "character"):
        if not isinstance(obj[k], str) or not obj[k]:
            errs.append(f"{k} 必须是非空字符串")

    for k in NUMERIC_FIELDS:
        v = obj[k]
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            errs.append(f"{k} 必须是数值，得到 {type(v).__name__}")
            continue
        lo, hi = RANGES[k]
        if not (lo <= v <= hi):
            errs.append(f"{k}={v} 越界 [{lo},{hi}]")

    expr = obj["expression"]
    if not isinstance(expr, dict):
        errs.append("expression 必须是对象")
    else:
        if expr.get("style") not in STYLES:
            errs.append(f"expression.style={expr.get('style')!r} 不在枚举内")
        if expr.get("gaze") not in GAZES:
            errs.append(f"expression.gaze={expr.get('gaze')!r} 不在枚举内")
        au = expr.get("au")
        if not isinstance(au, dict):
            errs.append("expression.au 必须是对象")
        else:
            extra = set(au) - set(AU_KEYS)
            if extra:
                errs.append(f"expression.au 含未知 AU: {sorted(extra)}")
            for k, v in au.items():
                if not isinstance(v, (int, float)) or isinstance(v, bool):
                    errs.append(f"AU {k} 必须是数值")
                elif not (0.0 <= v <= 1.0):
                    errs.append(f"AU {k}={v} 越界 [0,1]")

    cons = obj["constraints"]
    if not isinstance(cons, list) or any(not isinstance(c, str) for c in cons):
        errs.append("constraints 必须是字符串数组")

    ev = obj["evidence"]
    if not isinstance(ev, str) or not ev.strip():
        errs.append("evidence 必须是非空字符串")

    return errs


def check_evidence(evidence: str, text: str) -> list[str]:
    """L1b：归因校验。evidence 必须是台词原文的子串。

    归一化：去空白 + 去引号/括注标点。模型引用台词时常补 「」『』""
    等外层引号或句尾标点，这不算幻觉；只有引入原文没有的字词才算。
    """
    if not isinstance(evidence, str):
        return ["evidence 不是字符串"]
    strip = "，。？！、；：""''「」『』…—·,.?!;:\"'"
    clean = lambda s: "".join(ch for ch in s if not ch.isspace() and ch not in strip)
    # 全角括号整块视为模型的标注（如「（潜台词：…）」），不参与引文比对；
    # 但只允许出现在开头/结尾，夹在引文中间的括号仍算违规（可能藏真幻觉）
    ev = re.sub(r"^（[^）]*）|（[^）]*）$", "", evidence)
    ev = clean(ev)
    if not ev:
        return ["evidence 为空"]
    if ev not in clean(text):
        return [f"evidence {evidence!r} 不是台词原文的子串（幻觉归因）"]
    return []


# ==========================================================================
# 判读层契约（appraisal）
# ==========================================================================
# 为什么要单独一个契约
# -------------------
# 8 个表演参数里，只有 arousal 是"读台词"读出来的，其余 7 个都是
# arousal × 性格 × 场景 的确定性函数（见 reference.calibrate）。
# 让 LLM 一次吐出 8 个字段，等于让它同时做判读和算术；
# 195 次真实运行证明它算不准（见 outputs/analysis_real_v32.md 第 4 节）。
# 所以第二层拆出来：模型只输出下面 4 个字段，标定交回确定性代码。
#
# INTENTS 必须与 reference.CUE_TABLE 的标签**逐字一致**，
# 否则"意图准确率"会因为标签写法不同而虚低。
INTENTS = (
    "威胁", "认错", "挽留", "表白/安抚", "隐忍", "玩笑",
    "惊喜", "危机", "质疑", "忐忑", "陈述",
)

APPRAISAL_RANGES = {"arousal": (0.0, 10.0), "valence": (-1.0, 1.0)}

APPR_REQUIRED = ("line_id", "arousal", "valence", "intent", "evidence")


def validate_appraisal(obj: object) -> list[str]:
    """判读层的 L1：结构、区间、枚举、归因。不依赖金标准。"""
    errs: list[str] = []
    if not isinstance(obj, dict):
        return [f"判读结果不是对象，而是 {type(obj).__name__}"]

    for f in APPR_REQUIRED:
        if f not in obj:
            errs.append(f"缺字段 {f}")
    if errs:
        return errs

    if not isinstance(obj["line_id"], str) or not obj["line_id"]:
        errs.append("line_id 必须是非空字符串")

    for k, (lo, hi) in APPRAISAL_RANGES.items():
        v = obj[k]
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            errs.append(f"{k} 必须是数值，得到 {type(v).__name__}")
        elif not (lo <= v <= hi):
            errs.append(f"{k}={v} 越界 [{lo},{hi}]")

    if obj["intent"] not in INTENTS:
        errs.append(f"intent={obj['intent']!r} 不在枚举内")

    ev = obj["evidence"]
    if not isinstance(ev, str) or not ev.strip():
        errs.append("evidence 必须是非空字符串")

    return errs
