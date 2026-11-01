"""从 data/stage/*.json 生成 annotations/real_form_v2.csv（100 行真实剧本标注表）。

与 real_form.csv（40 行大话西游）的关系：
- 列结构逐字一致，v2 只是扩大样本；line_id 前缀 real2-，避免与 real-NN 冲突。
- 合并走 merge_real_annotations.py，eval_real.py 无需改动。

清洗规则（复用 stage_adapter 的 LINE_RE / MIN_LEN / MAX_LEN）：
1. 只收「角色：台词」行；剥掉全/半角括号里的动作提示后重验长度 4–28。
2. 台词文本全局去重（含与 real_form.csv 现有 40 行去重）。
3. 前后句取 content 里相邻的非空原文行（不筛长度，保上下文完整），
   与 real_form.csv 的既有口径一致。
4. 分散：影片轮转，每场最多取 1 句，场次按序推进。
"""

from __future__ import annotations

import csv
import glob
import json
import os
import re
import sys

from stage_adapter import LINE_RE, MAX_LEN, MIN_LEN

OUT_PATH = os.path.join("annotations", "real_form_v2.csv")
N_TARGET = 100
PER_MOVIE = 34    # 目标 100 / 有效片数向上取整
MIN_POOL = 40     # 候选不足此数的片跳过（纯叙述式或台词被硬换行截断的片）

FIELDS = ["序号", "line_id", "出处", "场次", "说话人", "前一句", "台词",
          "后一句", "arousal(1-10)", "备注"]

PAREN = ("(", "（", ")", "）")

# 不是真人在说话的「说话人」：字幕、旁白、画外音没有可标的内心状态
NON_SPEAKER = ("字幕", "旁白", "画外音", "标题", "歌词")


def is_real_speaker(spk: str) -> bool:
    return not any(w in spk for w in NON_SPEAKER)


def is_cast_list(text: str) -> bool:
    """出场人物清单行：如「家宁、小芝芝、经理、服务生若干」「老朱、家宁」。
    特征：顿号分隔的纯汉字人名串（不含任何其他标点/句读）；
    正常台词几乎必有标点或虚词，不会长这样。"""
    return "、" in text and bool(re.fullmatch(r"[\u4e00-\u9fa5、]{2,28}", text))


def strip_paren(text: str) -> str:
    """剥掉任意位置的 (...) / （...） 动作提示，保留其余文本。"""
    out, depth = [], 0
    for ch in text:
        if ch in PAREN[:2]:
            depth += 1
        elif ch in PAREN[2:]:
            depth = max(0, depth - 1)
        elif depth == 0:
            out.append(ch)
    return "".join(out).strip()


def extract_movie(path: str) -> list[dict]:
    """返回 [{scene_index, speaker, text, prev_raw, next_raw}]，已清洗过滤。"""
    scenes = json.load(open(path, encoding="utf-8"))
    out = []
    for s in scenes:
        raw_lines = [t.strip() for t in (s.get("content") or "").split("\n") if t.strip()]
        hits = []  # (raw_idx, speaker, cleaned_text)
        for ri, raw in enumerate(raw_lines):
            m = LINE_RE.match(raw)
            if not m or not is_real_speaker(m.group(1).strip()):
                continue
            text = strip_paren(m.group(2))
            if not (MIN_LEN <= len(text) <= MAX_LEN) or is_cast_list(text):
                continue
            hits.append((ri, m.group(1).strip(), text))
        if len(hits) < 3:
            continue
        for ri, spk, text in hits:
            prev_raw = raw_lines[ri - 1] if ri > 0 else ""
            next_raw = raw_lines[ri + 1] if ri + 1 < len(raw_lines) else ""
            out.append({"scene": s.get("_id", s.get("index")),
                        "speaker": spk, "text": text,
                        "prev": prev_raw, "next": next_raw})
    return out


def main() -> int:
    info = {r["movie_id"]: r["title"] for r in csv.DictReader(
        open(os.path.join("data", "stage", "chinese_movie_info.csv"),
             encoding="utf-8-sig"))}

    # 全局去重集：v2 自身 + 现有 real_form.csv 的台词
    seen = {r["台词"] for r in
            csv.DictReader(open("annotations/real_form.csv", encoding="utf-8-sig", newline=""))}

    # 每部片：场次升序、每场取第 1 条合格台词 → 天然按场分散
    movies = []  # [title, [(scene, cand)]]
    for path in sorted(glob.glob(os.path.join("data", "stage", "*.json"))):
        mid = os.path.basename(path).replace(".json", "")
        title = info.get(mid, mid[:10])
        per_scene: dict[int, list[dict]] = {}
        for ln in extract_movie(path):
            per_scene.setdefault(ln["scene"], []).append(ln)
        cands = []
        for sc in sorted(per_scene):
            cands.append(per_scene[sc][0])  # 每场只留 1 条
        movies.append([title, cands])

    movies.sort(key=lambda m: -len(m[1]))  # 台词多的片先出，保证凑满
    movies = [m for m in movies if len(m[1]) >= MIN_POOL]

    rows, i_movie = [], 0
    while len(rows) < N_TARGET and any(m[1] for m in movies):
        title, cands = movies[i_movie % len(movies)]
        taken = sum(1 for r in rows if r["出处"] == title)
        if cands and taken < PER_MOVIE:
            ln = cands.pop(0)
            if ln["text"] not in seen:
                seen.add(ln["text"])
                rows.append({"出处": title, "场次": f"第{ln['scene']}场",
                             "说话人": ln["speaker"], "前一句": ln["prev"],
                             "台词": ln["text"], "后一句": ln["next"]})
        else:
            movies[i_movie % len(movies)][1] = []  # 该片配额用尽
        i_movie += 1

    if len(rows) < N_TARGET:
        print(f"[警告] 只凑到 {len(rows)} 行（目标 {N_TARGET}），可调大 PER_MOVIE")

    for i, r in enumerate(rows, 1):
        r["序号"] = str(i)
        r["line_id"] = f"real2-{i:03d}"
        r["arousal(1-10)"] = ""
        r["备注"] = ""

    os.makedirs("annotations", exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    # 验收输出
    n_movie = len({r["出处"] for r in rows})
    n_scene = len({(r["出处"], r["场次"]) for r in rows})
    lens = [len(r["台词"]) for r in rows]
    print(f"写出 {OUT_PATH}: {len(rows)} 行（含表头 {len(rows) + 1}）")
    print(f"覆盖 {n_movie} 部影片 / {n_scene} 个场次")
    print(f"台词长度 {min(lens)}–{max(lens)} 字")
    print(f"line_id 首尾: {rows[0]['line_id']} .. {rows[-1]['line_id']}")
    print("出处分布:", {t: sum(1 for r in rows if r["出处"] == t)
                      for t in dict.fromkeys(r["出处"] for r in rows)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
