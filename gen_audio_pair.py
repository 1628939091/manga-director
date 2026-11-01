"""B：用角色卡制造差异——同一句台词，极内敛 vs 极外放。

为什么这样做：
    上一版 demo 是同一张中性卡下 naive vs calibrated，ΔI 上限只有 4.2，
    落到声学上只有 1.9 个半音——人耳分不出来。
    抑制系数的真实作用对象是「性格」，所以这里把性格拉到两端：
        内敛卡（四维 0.05）→ expressiveness 0.05 → 强度上限 3.7
        外放卡（四维 0.95）→ expressiveness 0.95 → 强度上限 7.8
    同一句话、同一个内心唤醒，两张卡演出来的强度差就是抑制系数本身。

诚实标注：真实剧本没有角色卡，这两张卡是**演示用构造**，
          目的是让「抑制」这个机制变得可听；不是实验数据。
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

import edge_tts
import numpy as np
import soundfile as sf

ROOT = r"D:\develop\2026-10-21-12-21-27\projects\manga-director"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "..", "manga-performance-director", "director"))

import oai  # noqa: E402
from mdh import reference as ref  # noqa: E402

OUT = os.path.join(ROOT, "outputs", "audio_pair")
VOICE = "zh-CN-YunxiNeural"
SR = 16000

INTROVERT = {k: 0.05 for k in ("warmth", "dominance", "energy", "neuroticism")}
EXTROVERT = {k: 0.95 for k in ("warmth", "dominance", "energy", "neuroticism")}
NEUTRAL = {k: 0.5 for k in ("warmth", "dominance", "energy", "neuroticism")}

LINES = [
    ("gold-18", "沙僧", "前面有埋伏！"),
    ("gold-15", "悟空", "哗－－！闭嘴！"),
    ("gold-27", "紫霞", "你又明不明白我已经不再是神仙了！"),
]


def spec_for(intensity: float, card: dict, scene: str, relation: str) -> dict:
    base = ref.personality_baseline(card)
    sc = ref.SCENE_PROFILE.get(scene, ref.SCENE_PROFILE["日常"])
    appr = ref.appraise("")          # 只为拿 valence/intent 的默认值
    internal = {"arousal": intensity, "valence": appr["valence"], "intent": appr["intent"]}
    cal = ref.calibrate(dict(internal), base, sc, relation)
    return {k: cal[k] for k in ("intensity", "f0_offset", "f0_range", "rate", "pause_after_ms")}


async def synth(text: str, spec: dict, mp3: str) -> None:
    rate_pct = max(-60, min(60, round((spec["rate"] / 4.2 - 1) * 100)))
    vol_pct = max(-50, min(50, round((spec["intensity"] - 4) * 8)))
    hz = round(180.0 * (2 ** (spec["f0_offset"] / 12) - 1))
    await edge_tts.Communicate(text, VOICE, rate=f"{rate_pct:+d}%",
                               volume=f"{vol_pct:+d}%", pitch=f"{hz:+d}Hz").save(mp3)


def mp3_to_wav(mp3: str, wav: str, pause_ms: int) -> None:
    data, sr = sf.read(mp3)
    if getattr(data, "ndim", 1) > 1:
        data = data.mean(axis=1)
    s = [float(v) for v in np.asarray(data, dtype=float)]
    s.extend([0.0] * int(pause_ms / 1000 * sr))
    oai.write_wav(wav, s, sr)


async def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for lid, spk, text in LINES:
        # 内心唤醒：取一个高位值，模拟「心里已经很激动」
        arousal = 8.5
        ext = spec_for(arousal, EXTROVERT, "危机", "对手")
        intro = spec_for(arousal, INTROVERT, "危机", "对手")
        base = spec_for(4.0, NEUTRAL, "危机", "对手")

        wavs = {}
        for tag, sp in (("extrovert", ext), ("introvert", intro), ("baseline", base)):
            mp3 = os.path.join(OUT, f"{lid}_{tag}.mp3")
            wav = os.path.join(OUT, f"{lid}_{tag}.wav")
            await synth(text, sp, mp3)
            mp3_to_wav(mp3, wav, sp["pause_after_ms"])
            os.remove(mp3)
            wavs[tag] = wav

        f_ext = oai.extract_features(wavs["extrovert"])
        f_int = oai.extract_features(wavs["introvert"])
        o_ext = oai.compute_oai(wavs["extrovert"], wavs["baseline"], arousal)
        o_int = oai.compute_oai(wavs["introvert"], wavs["baseline"], arousal)

        d_sem = ext["f0_offset"] - intro["f0_offset"]
        rows.append({
            "line_id": lid, "character": spk, "text": text, "arousal": arousal,
            "ext": ext, "intro": intro, "baseline": base, "wavs": wavs,
            "oai_ext": o_ext["oai"], "oai_int": o_int["oai"],
            "f0_ext": f_ext["f0_mean_st"], "f0_int": f_int["f0_mean_st"],
            "d_semitone": d_sem,
            "d_ratio": 2 ** (d_sem / 12) - 1,
            "d_rate": ext["rate"] / intro["rate"] - 1,
            "d_vol": (ext["intensity"] - intro["intensity"]) * 8 / 100,
        })
        print(f"ok {lid}  {spk}：「{text[:14]}」 I {ext['intensity']:.1f} vs {intro['intensity']:.1f} "
              f"→ 音高差 {d_sem:.1f} 半音")

    json.dump(rows, open(os.path.join(OUT, "manifest.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    print(f"\n{'line':12s} {'I 外放/内敛':>12s} {'音高差':>10s} {'语速差':>7s} {'音量差':>7s} {'OAI 外放/内敛':>16s}")
    for r in rows:
        print(f"{r['line_id']:12s} {r['ext']['intensity']:5.1f}/{r['intro']['intensity']:<5.1f} "
              f"{r['d_semitone']:+6.1f}半音({r['d_ratio']*100:+.0f}%) "
              f"{r['d_rate']*100:+6.0f}% {r['d_vol']*100:+6.0f}% "
              f"{r['oai_ext']:+7.2f}/{r['oai_int']:+6.2f}")
    print(f"\n输出：{OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
