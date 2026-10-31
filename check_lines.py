"""台词库体检：唯一性、桶覆盖、潜台词合法性、arousal 离散度。

改完 mdh/lines.py 必须跑一遍。四项检查对应四个已知失效模式：
1. 文本重复     → 独立刺激数被高估，MAE 的有效自由度虚高
2. 桶偏斜       → 按 arousal 分层的分析失真
3. 潜台词污染   → neutral 里混入线索词会替代台词决定金标（2026-10-31 的真实 bug）
4. 取值太少     → 判读任务退化成"十选一"，MAE 没有分辨力
"""

from __future__ import annotations

import collections
import sys

sys.path.insert(0, ".")

from mdh import lines  # noqa: E402
from mdh import reference as ref  # noqa: E402

ORDER = ["威胁", "认错", "挽留", "表白/安抚", "隐忍", "玩笑", "惊喜", "危机", "质疑", "陈述", "忐忑"]


def bucket_of(text: str, subtext: str = "") -> tuple[str, float, float]:
    r = ref.appraise(text, subtext)
    return r["intent"], r["arousal"], r["valence"]


def _cue_words() -> list[str]:
    """CUE_TABLE 里所有关键词（含触发桶的那些单字词）。"""
    return [w for cues, *_ in ref.CUE_TABLE for w in cues]


def main() -> int:
    ok = True
    all_texts = lines.all_lines()
    pairs = lines.all_pairs()

    # ---- 1. 唯一性 ----
    dup = [t for t, c in collections.Counter(all_texts).items() if c > 1]
    print(f"台词总数: {len(all_texts)}　唯一: {len(set(all_texts))}")
    if dup:
        ok = False
        print("❌ 重复文本:", dup)

    norm = collections.Counter("".join("".join(t.split()).split("…")) for t in all_texts)
    near = [t for t, c in norm.items() if c > 1]
    if near:
        print("⚠️ 归一化后重复（仅标点/省略号不同）:", near)

    # ---- 2. 潜台词合法性（这一项是 2026-10-31 加的）----
    # neutral 必须"干净"：不含触发词，也不含任何 CUE_TABLE 关键词，
    # 否则它自己就能决定金标，台词读不读得懂就无所谓了。
    cue_words = _cue_words()
    bad_neutral, bad_tense, hijack = [], [], []
    for text, neu, ten in pairs:
        if any(w in neu for w in lines.NEUTRAL_FORBIDDEN):
            bad_neutral.append((text, neu, "含触发词"))
        elif any(w in neu for w in cue_words):
            bad_neutral.append((text, neu, "含线索词，会替代台词定金标"))
        if not any(w in ten for w in lines.TENSE_CUES):
            bad_tense.append((text, ten, "不含触发词，+1.2 不生效"))
        # tense 不能把台词拖进更靠前的桶（顺序取第一个命中）。
        # 唯一允许的过渡是「陈述 → 忐忑」：台词本身无线索、潜台词说"心里发慌"
        # 时，appraise 的兜底分支本来就该给 忐忑——这不是劫持，是设计。
        b_text = bucket_of(text, "")[0]
        b_pair = bucket_of(text, ten)[0]
        if b_text != b_pair and not (b_text == "陈述" and b_pair == "忐忑"):
            hijack.append((text, ten, b_text, b_pair))

    print("\n潜台词检查")
    print(f"  neutral 干净: {len(pairs) - len(bad_neutral)} / {len(pairs)}")
    for t, s, why in bad_neutral[:6]:
        print(f"    ❌ {t} | {s} —— {why}")
    print(f"  tense 触发词有效: {len(pairs) - len(bad_tense)} / {len(pairs)}")
    for t, s, why in bad_tense[:6]:
        print(f"    ❌ {t} | {s} —— {why}")
    print(f"  tense 未劫持桶归属: {len(pairs) - len(hijack)} / {len(pairs)}")
    for t, s, b1, b2 in hijack[:6]:
        print(f"    ❌ {t} | {s} —— {b1} 被改成 {b2}")
    if bad_neutral or bad_tense or hijack:
        ok = False

    # ---- 3. 桶覆盖（按 neutral 潜台词，即"只凭台词"时的金标）----
    print("\n各场景类型的桶分布（neutral 潜台词）")
    print(f"{'场景':10s}" + "".join(f"{b:>7s}" for b in ORDER[:-1]))
    total = collections.Counter()
    for st, group in lines.SCENE_LINES.items():
        c = collections.Counter(bucket_of(t, neu)[0] for t, neu, _ in group)
        total.update(c)
        print(f"{st:10s}" + "".join(f"{c.get(o, 0):>8d}" for o in ORDER[:-1]))
    print(f"{'合计':10s}" + "".join(f"{total.get(o, 0):>8d}" for o in ORDER[:-1]))

    missing = [o for o in ORDER[:-1] if total.get(o, 0) == 0]
    if missing:
        ok = False
        print("❌ 完全没有覆盖的桶:", missing)

    # ---- 4. arousal 离散度（neutral 与 tense 两档都算）----
    print("\narousal 取值（neutral + tense）")
    vals = set()
    for text, neu, ten in pairs:
        vals.add(bucket_of(text, neu)[1])
        vals.add(bucket_of(text, ten)[1])
    print(f"  唯一取值 {len(vals)} 个：{sorted(vals)}")
    if len(vals) < 12:
        ok = False
        print("❌ 离散度不足（<12），判读任务过于粗糙")

    # ---- 5. 抽查：一眼看语义是否成立（人工复核用）----
    print("\n抽查 8 条（台词 / 中性 / 紧张 / 两档 arousal）")
    checked = 0
    for st, group in lines.SCENE_LINES.items():
        for text, neu, ten in group[:2]:
            a1 = bucket_of(text, neu)[1]
            a2 = bucket_of(text, ten)[1]
            print(f"  {st:8s} {text:14s} {neu:12s} → {ten:14s} aro {a1} / {a2}")
            checked += 1
            if checked >= 8:
                break
        if checked >= 8:
            break

    print("\n结论:", "通过" if ok else "需要修")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
