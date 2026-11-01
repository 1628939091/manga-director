"""B3：对 gen_audio_demo.py 产出的音频算 OAI，出 naive vs calibrated 对比表。

    OAI = 0.40·z_E + 0.30·z_R + 0.15·z_F + 0.15·z_S − E_expected

expected_arousal 用人工金标（同一句两版相同）——这样两版 OAI 之差
全部来自音频特征差，而不是「这句本来就该激烈」的上下文项。

两套音频各算一份：合成信号版（outputs/audio/，参数直控示意）
与 edge-tts 真人声版（outputs/audio_edge/，主 demo）。

用法：python calc_oai.py
输出：outputs/audio/oai_table.md / oai_table.json
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join("..", "manga-performance-director", "director"))
import oai  # noqa: E402

OUT = os.path.join("outputs", "audio")
OUT_EDGE = os.path.join("outputs", "audio_edge")


def pair_oai(wavs: dict, base_dir: str, expected_arousal: float) -> dict:
    """一组 naive/calibrated/baseline wav → OAI 两行 + diff。"""
    base = os.path.join(base_dir, os.path.basename(wavs["baseline"]))
    res = {}
    for tag in ("naive", "calibrated"):
        tgt = os.path.join(base_dir, os.path.basename(wavs[tag]))
        res[tag] = oai.compute_oai(tgt, base, expected_arousal)
    res["diff"] = round(res["naive"]["oai"] - res["calibrated"]["oai"], 2)
    return res


def main() -> int:
    manifest = json.load(open(os.path.join(OUT, "manifest.json"), encoding="utf-8"))
    rows = []
    for m in manifest:
        row = {"line_id": m["line_id"], "character": m["character"],
               "text": m["text"], "arousal": m["arousal_human"],
               "synth": pair_oai(m["wavs"], OUT, m["arousal_human"])}
        if m.get("wavs_edge"):
            row["edge"] = pair_oai(m["wavs_edge"], OUT_EDGE, m["arousal_human"])
        rows.append(row)

    json.dump(rows, open(os.path.join(OUT, "oai_table.json"), "w",
                         encoding="utf-8"), ensure_ascii=False, indent=1)

    L = ["# OAI 对比：naive（intensity 直接当表演强度） vs calibrated（抑制校准）", "",
         "| line_id | 角色 | 台词 | 人工arousal | naive OAI | calibrated OAI | 合成diff | 真人声diff |",
         "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        s, e = r["synth"], r.get("edge", {})
        L.append(f"| {r['line_id']} | {r['character']} | {r['text'][:18]}… "
                 f"| {r['arousal']:.0f} | {s['naive']['oai']:.2f} "
                 f"| {s['calibrated']['oai']:.2f} | {s['diff']:+.2f} "
                 f"| {(e.get('diff') or 0):+.2f} |")
    L += ["", "## 逐项 z 分数（合成信号版 / 真人声版）", "",
          "| line_id | 版本 | z_energy | z_f0_range | z_f0_mean | z_rate |",
          "|---|---|---|---|---|---|"]
    for r in rows:
        for set_name in ("synth", "edge"):
            if set_name not in r:
                continue
            for tag in ("naive", "calibrated"):
                z = r[set_name][tag]
                L.append(f"| {r['line_id']} | {set_name}/{tag} | {z['z_energy']:+.2f} "
                         f"| {z['z_f0_range']:+.2f} | {z['z_f0_mean']:+.2f} "
                         f"| {z['z_rate']:+.2f} |")
    L += ["", f"判读基准：OAI < 1.0 正常；1.0–2.0 偏夸张；> 2.0 表演过度。",
          "expected_arousal 两版相同（人工金标），diff 全部来自音频特征差。"]

    open(os.path.join(OUT, "oai_table.md"), "w", encoding="utf-8").write("\n".join(L))

    for set_name, label in (("synth", "合成信号"), ("edge", "真人声")):
        sub = [r for r in rows if set_name in r]
        if not sub:
            continue
        n_pos = sum(1 for r in sub if r[set_name]["diff"] > 0)
        print(f"[{label}]")
        print("| line_id | naive OAI | calibrated OAI | diff |")
        print("|---|---|---|---|")
        for r in sub:
            print(f"| {r['line_id']} | {r[set_name]['naive']['oai']:.2f} "
                  f"| {r[set_name]['calibrated']['oai']:.2f} "
                  f"| {r[set_name]['diff']:+.2f} |")
        print(f"naive > calibrated: {n_pos}/{len(sub)}\n")
    print("输出: outputs/audio/oai_table.md / oai_table.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
