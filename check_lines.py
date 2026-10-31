"""台词库体检：唯一性、桶覆盖、arousal 离散度。

改完 mdh/lines.py 必须跑一遍。三项检查对应三个已知失效模式：
1. 文本重复 → 独立刺激数被高估，MAE 的有效自由度虚高
2. 桶偏斜   → 按 arousal 分层的分析失真
3. 取值太少 → 判读任务退化成"十选一"，MAE 没有分辨力
"""

from __future__ import annotations

import collections
import sys

sys.path.insert(0, ".")

from mdh import lines  # noqa: E402
from mdh import reference as ref  # noqa: E402


def bucket_of(text: str, subtext: str = "") -> tuple[str, float, float]:
    r = ref.appraise(text, subtext)
    return r["intent"], r["arousal"], r["valence"]


def main() -> int:
    ok = True
    all_texts = lines.all_lines()

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

    # ---- 2. 桶覆盖 ----
    print("\n各场景类型的桶分布（neutral 潜台词）")
    print(f"{'场景':10s}" + "".join(f"{b:>7s}" for b in
          ["威胁", "认错", "挽留", "表白", "隐忍", "玩笑", "惊喜", "危机", "质疑", "陈述"]))
    order = ["威胁", "认错", "挽留", "表白/安抚", "隐忍", "玩笑", "惊喜", "危机", "质疑", "陈述", "忐忑"]
    short = ["威胁", "认错", "挽留", "表白", "隐忍", "玩笑", "惊喜", "危机", "质疑", "陈述"]
    total = collections.Counter()
    for st, texts in lines.SCENE_LINES.items():
        c = collections.Counter(bucket_of(t)[0] for t in texts)
        total.update(c)
        print(f"{st:10s}" + "".join(f"{c.get(o, 0):>8d}" for o in order[:-1]))
    print(f"{'合计':10s}" + "".join(f"{total.get(o, 0):>8d}" for o in order[:-1]))

    missing = [o for o in order[:-1] if total.get(o, 0) == 0]
    if missing:
        ok = False
        print("❌ 完全没有覆盖的桶:", missing)

    # ---- 3. arousal 离散度 ----
    print("\narousal 取值（含 tense 潜台词）")
    vals = set()
    for t in all_texts:
        vals.add(bucket_of(t, "")[1])
        vals.add(bucket_of(t, "其实心里发慌")[1])
    print(f"  唯一取值 {len(vals)} 个：{sorted(vals)}")
    if len(vals) < 12:
        ok = False
        print("❌ 离散度不足（<12），判读任务过于粗糙")

    # ---- 4. 抽查：哪些文本的桶与直觉不符（人工复核用） ----
    print("\n抽查 6 条（场景 / 台词 / 桶 / arousal）")
    for st, texts in lines.SCENE_LINES.items():
        t = texts[0]
        b, a, v = bucket_of(t)
        print(f"  {st:8s} {t:20s} {b:8s} aro={a:<5} val={v}")

    print("\n结论:", "通过" if ok else "需要修")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
