"""workbench CLI：剧本片段 -> 分镜 JSON (+SSML / TTS 参数 / 视频提示词)。

把「编译成可执行参数」变成一条命令：

    python workbench.py script.txt --out storyboard.json --ssml --tts --prompt

输入格式：每行「角色：台词」（全角/半角冒号都认）。无法解析的非空行跳过并计数，
不猜、不补——剧本解析错了应该让人看见，而不是让模型脑补。

角色一律用中性卡（性格四维全 0.5）：本工具不负责角色设定，那是上游的事；
要按性格压制，把 storyboard.json 里的镜头喂给 mdh.reference.calibrate 前先换卡。

零第三方依赖：只 import 标准库 + mdh。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mdh import export  # noqa: E402
from mdh import reference as ref  # noqa: E402

# 中性角色卡：与 stage_adapter.NEUTRAL_CARD 同构，但不 import stage_adapter
#（那边会牵扯 data/ 目录依赖，workbench 应该对任意纯文本可用）。
DEFAULT_CARD = {
    "role": "配角",
    "voice": "未知",
    "personality": {"warmth": 0.5, "dominance": 0.5,
                    "energy": 0.5, "neuroticism": 0.5},
}

LINE_RE = re.compile(r"^\s*([^：:]{1,12})[：:]\s*(.+?)\s*$")


def parse_script(path: str) -> tuple[list[dict], int]:
    """解析剧本文本 -> (lines, 跳过行数)。lines 结构与 mdh 任务一致。"""
    raw = open(path, encoding="utf-8-sig").read()
    lines: list[dict] = []
    skipped = 0
    for i, ln in enumerate(raw.splitlines(), 1):
        if not ln.strip():
            continue
        m = LINE_RE.match(ln)
        if not m:
            skipped += 1
            continue
        speaker, text = m.group(1).strip(), m.group(2).strip()
        lines.append({
            "id": f"W01-L{i:02d}",
            "character": speaker,
            "text": text,
            "scene_type": "日常",
            "relation": "陌生人",
            "listener": "",
            "subtext": "",
        })
    return lines, skipped


def build_shots(lines: list[dict]) -> list[dict]:
    """台词 -> 金标准镜头（确定性，零 LLM）。"""
    cast = sorted({ln["character"] for ln in lines})
    characters = {c: DEFAULT_CARD for c in cast}
    return [ref.gold_shot(ln, characters) for ln in lines]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="剧本 -> 分镜/SSML/TTS/提示词")
    ap.add_argument("script", help="剧本文本，每行「角色：台词」")
    ap.add_argument("--out", default="storyboard.json", help="分镜 JSON 输出路径")
    ap.add_argument("--ssml", action="store_true", help="同时导出 SSML")
    ap.add_argument("--tts", action="store_true", help="同时导出 TTS 参数 JSON")
    ap.add_argument("--prompt", action="store_true", help="同时导出视频提示词")
    args = ap.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    if not os.path.isfile(args.script):
        print(f"[错误] 找不到 {args.script}")
        return 1
    lines, skipped = parse_script(args.script)
    if not lines:
        print("[错误] 没解析出任何「角色：台词」行；检查格式（全角/半角冒号）")
        return 1
    shots = build_shots(lines)

    out_path = args.out
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    json.dump(shots, open(out_path, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    stem = os.path.splitext(out_path)[0]
    written = [f"{out_path}（{len(shots)} 个镜头）"]
    if args.ssml:
        p = stem + "_ssml.xml"
        open(p, "w", encoding="utf-8").write(export.to_ssml(shots))
        written.append(p)
    if args.tts:
        p = stem + "_tts.json"
        json.dump(export.to_tts_params(shots), open(p, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        written.append(p)
    if args.prompt:
        p = stem + "_prompts.txt"
        open(p, "w", encoding="utf-8").write(
            "\n".join(export.to_video_prompt(shots)) + "\n")
        written.append(p)

    if skipped:
        print(f"[提示] 跳过 {skipped} 行不符合「角色：台词」格式的行")
    for w in written:
        print(f"  ✓ {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
