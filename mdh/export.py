"""导出层：把镜头（shots）编译成下游工具可直接消费的三种格式。

为什么需要这一层
----------------
契约里的 10 维参数是「导演视角」的中间表示；要真的被做出来，得落到三处：
配音（SSML / TTS 参数）、生成视频（提示词）。这三个函数就是从契约到
生产工具的桥——简历里说的「编译成可执行参数」的出口。

容错策略（刻意与判卷相反）
------------------------
judge 对越界值判死，export 对越界值**截断**：导出是生产链路的末端，
此时抛异常只会让整批产物作废，把参数夹回物理区间（mdh.schema.RANGES）
是无信息损失最小代价的处理。缺字段同理，落到中性默认值。
「判卷从严、导出从宽」——两头的容错语义本来就不同。

三个函数
--------
- to_ssml(shots)         -> 一份 SSML 文档（W3C prosody：rate/pitch/volume + break）
- to_tts_params(shots)   -> list[dict]，字段对齐 CosyVoice / GPT-SoVITS 的速度/音高/音量
- to_video_prompt(shots) -> list[str]，每镜头一句中文提示词（给 Wan2.1 / 可灵）

约定：shots 里每项是镜头 dict；台词文本依次取 ``text``、``evidence``
（金标准的 evidence 就是台词原文），都没有则为空串。非 dict 项跳过。
"""

from __future__ import annotations

from xml.sax.saxutils import escape

from mdh.schema import RANGES

# 语速基准（字/秒）：rate 是绝对字/秒，SSML/TTS 的语速是相对量，
# 需要一个锚点。取 4.5 ≈ 中文普通话自然语速（新闻播音 4~5 字/秒）。
RATE_BASE = 4.5
# 强度→音量的锚点：I=4（中性基线，见 reference.calibrate）为 0 dB。
VOL_BASE_INTENSITY = 4.0
VOL_DB_PER_POINT = 1.5   # 每差 1 分强度 ≈ 1.5 dB
VOL_DB_MAX = 12.0        # SSML 引擎普遍接受 ±12dB 上限

# 中性默认值（缺字段时落点，与中性角色卡的输出一致）
_DEFAULTS = {
    "intensity": 4.0,
    "f0_offset": 0.0,
    "f0_range": 3.0,
    "rate": RATE_BASE,
    "pause_after_ms": 0,
}

# AU -> 导演能读懂的中文（提示词里只挑数值最高的两个写出来）
AU_LABEL = {
    "AU4": "眉头紧锁",
    "AU12": "嘴角上扬",
    "AU17": "下巴内收抿嘴",
    "AU20": "嘴角绷紧",
    "AU26": "张嘴",
}
STYLE_ADVERB = {"压抑": "压抑地", "克制": "克制地", "自然": "自然地", "外放": "外放地"}


def _clamp(v: float, key: str) -> float:
    lo, hi = RANGES[key]
    return round(min(hi, max(lo, v)), 2)


def _num(shot: dict, key: str) -> float:
    """取数值字段：缺失落默认，越界截断，非数值也当缺失。"""
    v = shot.get(key, _DEFAULTS[key])
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        v = _DEFAULTS[key]
    return _clamp(float(v), key)


def _pause(shot: dict) -> int:
    v = shot.get("pause_after_ms", 0)
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        v = 0
    return int(min(RANGES["pause_after_ms"][1], max(0, v)))


def _text(shot: dict) -> str:
    for k in ("text", "evidence"):
        v = shot.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _clean(shots) -> list[dict]:
    if not shots:
        return []
    return [s for s in shots if isinstance(s, dict)]


# --------------------------------------------------------------------------
# 1. SSML
# --------------------------------------------------------------------------
def to_ssml(shots) -> str:
    """镜头列表 -> 一份 SSML 文档。

    映射：rate(字/秒) -> 相对百分比；f0_offset(半音) -> pitch ±st；
    intensity -> volume ±dB（I=4 为 0 dB）；pause_after_ms -> <break>。
    """
    items = _clean(shots)
    body = []
    for s in items:
        intensity = _num(s, "intensity")
        rate = _num(s, "rate")
        f0 = _num(s, "f0_offset")
        pause = _pause(s)
        rate_pct = int(round(rate / RATE_BASE * 100))
        vol_db = (intensity - VOL_BASE_INTENSITY) * VOL_DB_PER_POINT
        vol_db = max(-VOL_DB_MAX, min(VOL_DB_MAX, vol_db))
        text = escape(_text(s))
        brk = f'<break time="{pause}ms"/>' if pause > 0 else ""
        who = escape(str(s.get("character", "")))
        body.append(
            f'  <!-- {who} ｜ I={intensity:.2f} -->\n'
            f'  <prosody rate="{rate_pct}%" pitch="{f0:+.1f}st" '
            f'volume="{vol_db:+.1f}dB">{text}{brk}</prosody>'
        )
    return (
        '<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" '
        'xml:lang="zh-CN">\n' + "\n".join(body) + "\n</speak>"
    )


# --------------------------------------------------------------------------
# 2. TTS 参数（对齐 CosyVoice / GPT-SoVITS）
# --------------------------------------------------------------------------
def to_tts_params(shots) -> list[dict]:
    """镜头列表 -> TTS 参数数组。

    字段对齐：
    - speed            CosyVoice `speed` / GPT-SoVITS `speed_factor`，0.5~2.0
    - pitch_semitones  相对说话人基线的半音偏移（渲染端换算成 Hz 或音高指令）
    - f0_range_semitones  音高动态范围（部分引擎不支持，留作参考字段）
    - volume_db        相对基线增益
    其余原样透传，渲染端不用再回查契约。
    """
    out = []
    for s in _clean(shots):
        rate = _num(s, "rate")
        speed = min(2.0, max(0.5, rate / RATE_BASE))
        out.append({
            "line_id": s.get("line_id", ""),
            "character": s.get("character", ""),
            "text": _text(s),
            "speed": round(speed, 2),
            "pitch_semitones": _num(s, "f0_offset"),
            "f0_range_semitones": _num(s, "f0_range"),
            "volume_db": round(max(-VOL_DB_MAX, min(
                VOL_DB_MAX,
                (_num(s, "intensity") - VOL_BASE_INTENSITY) * VOL_DB_PER_POINT)), 1),
            "pause_after_ms": _pause(s),
            "intensity": _num(s, "intensity"),
            "suppression": s.get("suppression") if isinstance(
                s.get("suppression"), (int, float)) else None,
            "constraints": s.get("constraints")
            if isinstance(s.get("constraints"), list) else [],
        })
    return out


# --------------------------------------------------------------------------
# 3. 视频提示词（Wan2.1 / 可灵）
# --------------------------------------------------------------------------
def to_video_prompt(shots) -> list[str]:
    """镜头列表 -> 每镜头一句中文提示词。

    从 expression.style / au / gaze 和强度参数翻译成人话——
    视频模型吃不动 AU 编号，只吃得了「皱着眉、避开对视」这种短语。
    """
    out = []
    for s in _clean(shots):
        intensity = _num(s, "intensity")
        rate = _num(s, "rate")
        pause = _pause(s)
        expr = s.get("expression") if isinstance(s.get("expression"), dict) else {}
        style = expr.get("style") if expr.get("style") in STYLE_ADVERB else "自然"
        gaze = expr.get("gaze") if isinstance(expr.get("gaze"), str) else ""
        au = expr.get("au") if isinstance(expr.get("au"), dict) else {}
        top = sorted(
            ((k, v) for k, v in au.items()
             if isinstance(v, (int, float)) and v >= 0.3),
            key=lambda kv: -kv[1])[:2]
        au_txt = "、".join(AU_LABEL.get(k, k) for k, _ in top)
        pace = "语速偏快" if rate > 5.2 else ("语速偏慢" if rate < 3.8 else "语速平稳")
        parts = [str(s.get("character", "")).strip() or "角色",
                 STYLE_ADVERB[style]]
        if au_txt:
            parts.append(au_txt)
        if gaze:
            parts.append(gaze)
        parts.append(pace)
        if pause > 0:
            parts.append(f"说完停顿 {pause / 1000:.1f} 秒")
        prompt = f"{parts[0]}{parts[1]}说话（{'，'.join(parts[2:])}）"
        out.append(prompt)
    return out
