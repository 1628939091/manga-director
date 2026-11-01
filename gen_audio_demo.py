"""B2：音频闭环 demo —— 同一句台词，naive vs calibrated 两版配音 + OAI 对比。

数据链路
------
1. 挑句：手工指定三句（DEMO_LINES），各配一张有性格差异的角色卡——
   不再统一用中性卡。三句的抑制来源刻意不同，demo 才有梯度：
     real-15 悟空 A=8   日常+长辈（对师父）→ 场景上限 3.8 —— 场景/关系压制，ΔI≈4.2
     real-38 二当家 A=7 日常+上级（对帮主）→ 场景上限 4.0 —— 关系压制，ΔI≈3.0
     real-27 青霞 A=6   日常+恋人未满 → 冷傲低表现欲，轻度压制，ΔI≈1.6
   arousal 用 real_form.csv 人工金标（不用规则尺——规则尺对真实台词大多落 5.0）。
2. calibrated：mdh.reference.calibrate 的 7 参数
   （intensity 受人格上限 + 场景上限 + 禁吼约束压制）。
3. naive：intensity 直接 = 内部 arousal（没有抑制校准），其余派生参数
   按 calibrate 同一套公式但代入 naive intensity（withheld = 0）。
   这正是论文对照条件：「让模型把内部唤醒直接当表演强度」。
4. baseline：每角色一条中性朗读（intensity=4），供 compute_oai 做 z 分数基线。
5. 渲染：主 demo 用 edge-tts 真人声 → outputs/audio_edge/；
   合成信号版（oai.render_demo_wav，参数线性直控）→ outputs/audio/ 作示意（非人声）。
   edge-tts 参数映射（诚实记录，f0_range 无法直接控制）：
   pitch = f0_offset(st)、rate = (rate/4.2−1)×100%、volume = (intensity−4)×8%。

用法：python gen_audio_demo.py
输出：outputs/audio/{stem}_{naive|calibrated|baseline}.wav       （合成示意版）
      outputs/audio_edge/…                                       （真人声主版）
      outputs/audio/manifest.json
"""

from __future__ import annotations

import asyncio
import csv
import json
import os
import sys

import edge_tts

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mdh import reference as ref  # noqa: E402

sys.path.insert(0, os.path.join("..", "manga-performance-director"))
OAI_DIR = os.path.join("..", "manga-performance-director", "director")
sys.path.insert(0, OAI_DIR)
import oai  # noqa: E402

SR = 16000
VOICE_DEFAULT = "zh-CN-YunxiNeural"
VOICE_FEMALE = "zh-CN-XiaoxiaoNeural"
FEMALE = {"青霞", "香香", "妈妈", "紫霞"}

OUT_DIR = os.path.join("outputs", "audio")        # 合成信号示意版
OUT_EDGE = os.path.join("outputs", "audio_edge")  # edge-tts 真人声（主 demo）

# --------------------------------------------------------------------------
# 手工 demo 配置：三句 × 三张差异化角色卡（含抑制来源说明，进 manifest 给页面展示）
# --------------------------------------------------------------------------
DEMO_LINES = [
    {
        "line_id": "real-15",
        "personality": {"warmth": 0.30, "dominance": 0.85,
                        "energy": 0.90, "neuroticism": 0.50},
        "scene_type": "日常", "relation": "长辈",
        "note": "桀骜外放卡（表现欲 0.69），但日常场景对师父（长辈 −1.2）→ 场景上限 3.8 压制",
    },
    {
        "line_id": "real-38",
        "personality": {"warmth": 0.45, "dominance": 0.25,
                        "energy": 0.45, "neuroticism": 0.70},
        "scene_type": "日常", "relation": "上级",
        "note": "谄媚慌张卡（表现欲 0.44），对帮主（上级 −1.0）→ 场景上限 4.0 压制",
    },
    {
        "line_id": "real-27",
        "personality": {"warmth": 0.15, "dominance": 0.60,
                        "energy": 0.30, "neuroticism": 0.25},
        "scene_type": "日常", "relation": "恋人未满",
        "note": "冷傲克制卡（表现欲 0.32）→ 低表现欲轻压 + 恋人未满 −0.2",
    },
]


def derived_spec(intensity: float, baseline: dict, withheld: float) -> dict:
    """calibrate 的第二层派生公式（reference.py L96-108 同式，demo 渲染用）。"""
    return {
        "intensity": round(intensity, 2),
        "f0_range": round(min(6.0, 0.9 + 0.50 * intensity), 2),
        "f0_offset": round(baseline["f0_base"] + 0.45 * (intensity - 4.0), 2),
        "rate": round(baseline["rate_base"] * (0.85 + 0.035 * intensity), 2),
        "pause_after_ms": int(180 + (10 - intensity) * 35
                              + (260 if withheld >= 2.0 else 0)),
    }


async def synth(text: str, spec: dict, voice: str, mp3_path: str) -> None:
    rate_pct = max(-60, min(60, round((spec["rate"] / 4.2 - 1) * 100)))
    vol_pct = max(-40, min(40, round((spec["intensity"] - 4) * 8)))
    # edge-tts 7.x 只认 ±整数Hz；按 180Hz 参考音高把半音偏移换算成 Hz
    hz = round(180.0 * (2 ** (spec["f0_offset"] / 12) - 1))
    com = edge_tts.Communicate(text, voice, rate=f"{rate_pct:+d}%",
                               volume=f"{vol_pct:+d}%", pitch=f"{hz:+d}Hz")
    await com.save(mp3_path)


def mp3_to_wav(mp3_path: str, wav_path: str, pause_ms: int) -> None:
    import numpy as np
    import soundfile as sf
    data, sr = sf.read(mp3_path)
    if getattr(data, 'ndim', 1) > 1:
        data = data.mean(axis=1)
    samples = [float(v) for v in np.asarray(data, dtype=float)]
    samples.extend([0.0] * int(pause_ms / 1000 * sr))
    oai.write_wav(wav_path, samples, sr)


def pick_lines() -> list[dict]:
    """按 DEMO_LINES 手工配置从金标取句，组装 line/task 结构。"""
    gold = {r["line_id"]: r for r in csv.DictReader(
        open("annotations/real_form.csv", encoding="utf-8-sig", newline=""))
        if r["arousal(1-10)"].strip()}

    picks = []
    for cfg in DEMO_LINES:
        g = gold.get(cfg["line_id"])
        if not g:
            raise SystemExit(f"[缺少数据] real_form.csv 里没有 {cfg['line_id']} 的金标")
        speaker = g["说话人"]
        picks.append({
            "config": cfg,
            "task": {"characters": {speaker: {"personality": cfg["personality"]}}},
            "line": {"id": f"gold-{g['line_id']}", "character": speaker,
                     "scene": g["场次"], "scene_type": cfg["scene_type"],
                     "listener": "", "relation": cfg["relation"],
                     "text": g["台词"], "subtext": ""},
            "arousal": float(g["arousal(1-10)"]),
        })
    return picks


async def main_async() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(OUT_EDGE, exist_ok=True)
    picks = pick_lines()
    manifest, mp3s = [], []

    for p in picks:
        ln, task, cfg = p["line"], p["task"], p["config"]
        ch = task["characters"][ln["character"]]
        baseline = ref.personality_baseline(ch.get("personality", {}))
        scene = ref.SCENE_PROFILE.get(ln["scene_type"],
                                      ref.SCENE_PROFILE["日常"])
        appr = ref.appraise(ln["text"])
        internal = {"arousal": p["arousal"], "valence": appr["valence"],
                    "intent": appr["intent"]}

        cal = ref.calibrate(dict(internal), baseline, scene, ln["relation"])
        calibrated_spec = {k: cal[k] for k in
                           ("intensity", "f0_offset", "f0_range", "rate", "pause_after_ms")}
        naive_spec = derived_spec(p["arousal"], baseline, withheld=0.0)
        base_spec = derived_spec(4.0, baseline, withheld=0.0)

        voice = VOICE_FEMALE if ln["character"] in FEMALE else VOICE_DEFAULT
        stem = ln["id"].replace("real-", "")  # gold-15
        jobs = []
        for tag, spec in (("naive", naive_spec), ("calibrated", calibrated_spec),
                          ("baseline", base_spec)):
            wav = os.path.join(OUT_DIR, f"{stem}_{tag}.wav")
            jobs.append({"tag": tag, "wav": wav, "spec": spec, "voice": voice})

        # 合成信号版：参数线性直控（示意，非人声）
        for j in jobs:
            oai.render_demo_wav(j["spec"], ln["text"], j["wav"], SR)

        # edge-tts 真人声主版（同 spec 同 voice）
        for j in jobs:
            wav_e = os.path.join(OUT_EDGE, os.path.basename(j["wav"]))
            mp3 = wav_e.replace(".wav", ".mp3")
            await synth(ln["text"], j["spec"], j["voice"], mp3)
            mp3_to_wav(mp3, wav_e, j["spec"]["pause_after_ms"])
            mp3s.append(mp3)

        manifest.append({
            "line_id": ln["id"], "character": ln["character"], "text": ln["text"],
            "scene": ln.get("scene", ""), "arousal_human": p["arousal"],
            "voice": voice,
            "demo_card": {
                "personality": cfg["personality"],
                "scene_type": cfg["scene_type"], "relation": cfg["relation"],
                "expressiveness": baseline["expressiveness"],
                "intensity_cap": baseline["intensity_cap"],
                "note": cfg["note"],
            },
            "calibrated": calibrated_spec, "naive": naive_spec, "baseline": base_spec,
            # 相对路径统一用正斜杠（demo 页面直接拼 URL，反斜杠在 URL 里有歧义）
            "wavs": {j["tag"]: os.path.relpath(j["wav"]).replace(os.sep, "/")
                     for j in jobs},
            "wavs_edge": {j["tag"]: os.path.relpath(
                os.path.join(OUT_EDGE, os.path.basename(j["wav"]))
            ).replace(os.sep, "/") for j in jobs},
        })
        print(f"ok {stem}  {ln['character']}：「{ln['text'][:20]}」 arousal={p['arousal']}"
              f"  naive I={naive_spec['intensity']} vs cal I={calibrated_spec['intensity']}"
              f"  ΔI={naive_spec['intensity'] - calibrated_spec['intensity']:.1f}")

    json.dump(manifest, open(os.path.join(OUT_DIR, "manifest.json"), "w",
                             encoding="utf-8"), ensure_ascii=False, indent=1)
    for mp3 in mp3s:  # 中间 mp3 不进交付
        if os.path.exists(mp3):
            os.remove(mp3)
    print(f"\n输出 {len(manifest)} 组 × 3 wav → {OUT_EDGE}/ (真人声主版) 与 {OUT_DIR}/ (合成示意)")
    return 0


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    raise SystemExit(main())
