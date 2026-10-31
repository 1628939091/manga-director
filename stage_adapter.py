"""把 STAGE（真实电影剧本）转成 mdh 的 tasks 格式。

为什么需要这一步
----------------
项目此前所有场景都是模板生成的：台词按"线索桶"设计，潜台词成对写好。
它证明的是"能不能对齐关键词"，不是"能不能读真实台词"。
外部效度必须用真实剧本验一次。

数据来源
--------
zhuiwen-ai/STAGE（GitHub）：42 部中文电影剧本，逐场景切分。
每个 <hash>/script.json 是 [{_id, title, subtitle, content}]，
content 里是「角色：台词」格式的自由文本。

三个必须交代的转换假设
--------------------
1. **没有角色卡**。真实剧本只给对白，不给性格四维。这里统一用中性卡
   （0.5/0.5/0.5/0.5 → expressiveness 0.5、强度上限 5.75）。
   这本身就是真实数据与模板的差别之一。
2. **没有潜台词**。真实剧本的对白外面没有"内心"标注，所以 subtext 一律为空。
   于是真实组天然是"无潜台词线索"条件——正好和模板组形成对照。
3. **场景类型无法自动判定**。标题只有「第1场」，没有地点与内外景。
   统一置为「日常」（ceiling 5.0、禁止外放）。
   注意：appraise() 不看 scene_type，所以**不影响 arousal 金标**，
   只影响 intensity 等派生参数。
"""

from __future__ import annotations

import json
import os
import random
import re
from collections import Counter

LINE_RE = re.compile(r"^([\u4e00-\u9fa5A-Za-z0-9·]{1,10})[：:](.+)$")

NEUTRAL_CARD = {
    "role": "配角",
    "voice": "未知",
    "personality": {"warmth": 0.5, "dominance": 0.5,
                    "energy": 0.5, "neuroticism": 0.5},
}

# 过滤掉不像"表演决策"的行：太短（嗯、哦）、太长（旁白/动作描写）
MIN_LEN, MAX_LEN = 4, 28


def parse_script(path: str) -> list[dict]:
    scenes = json.load(open(path, encoding="utf-8"))
    out = []
    for si, s in enumerate(scenes, 1):
        lines = []
        for raw in (s.get("content") or "").split("\n"):
            t = raw.strip()
            if not t:
                continue
            m = LINE_RE.match(t)
            if not m:
                continue
            spk, txt = m.group(1).strip(), m.group(2).strip()
            if not (MIN_LEN <= len(txt) <= MAX_LEN):
                continue
            lines.append({"speaker": spk, "text": txt})
        if len(lines) >= 3:
            out.append({"index": si, "lines": lines})
    return out


def build_tasks(script_paths: list[str], n_scenes: int = 12,
                lines_per_scene: int = 3, seed: int = 20261031) -> list[dict]:
    rng = random.Random(seed)
    pool = []
    for p in script_paths:
        tag = os.path.basename(p).replace(".json", "")[:8]
        for sc in parse_script(p):
            pool.append((tag, sc))
    if not pool:
        return []

    rng.shuffle(pool)
    tasks = []
    seen_text: set[str] = set()
    for tag, sc in pool:
        if len(tasks) >= n_scenes:
            break
        picked, cast = [], []
        for ln in sc["lines"]:
            if ln["text"] in seen_text:
                continue
            seen_text.add(ln["text"])
            picked.append(ln)
            if ln["speaker"] not in cast:
                cast.append(ln["speaker"])
            if len(picked) >= lines_per_scene:
                break
        if len(picked) < lines_per_scene or len(cast) < 2:
            continue

        scene_id = f"real-{tag}-S{sc['index']:02d}"
        tasks.append({
            "scene_id": scene_id,
            "work_id": f"real-{tag}",
            "title": f"真实剧本场景 {sc['index']}",
            "characters": {name: NEUTRAL_CARD for name in cast},
            "lines": [{
                "id": f"{scene_id}-L{i + 1:02d}",
                "character": ln["speaker"],
                "scene": f"真实剧本第 {sc['index']} 场",
                "scene_type": "日常",
                "listener": picked[(i + 1) % len(picked)]["speaker"],
                "relation": "陌生人",
                "text": ln["text"],
                "subtext": "",
            } for i, ln in enumerate(picked)],
        })
    return tasks


def main() -> int:
    root = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(root, "data", "stage")
    out_dir = os.path.join(root, "data", "real_stage")
    os.makedirs(out_dir, exist_ok=True)

    paths = sorted(os.path.join(src_dir, f) for f in os.listdir(src_dir)
                   if f.endswith(".json"))
    if not paths:
        print(f"[缺少数据] {src_dir} 里没有 script.json，先下载 STAGE")
        return 1

    tasks = build_tasks(paths)
    out = os.path.join(out_dir, "tasks.json")
    json.dump(tasks, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    n_lines = sum(len(t["lines"]) for t in tasks)
    spk = Counter(ln["character"] for t in tasks for ln in t["lines"])
    print(f"源 {len(paths)} 部剧本 → {len(tasks)} 场 / {n_lines} 行")
    print(f"角色 {len(spk)} 个，top5: {spk.most_common(5)}")
    print(f"输出：{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
