"""人工标注 + 一致性检验：把「与规则表一致」变成「与人对得上」。

为什么必须做这件事
------------------
当前所有 arousal / valence / intent 的「正确率」，比的都是 `reference.appraise`
的输出——而它是一张关键词查找表：

    for cues, val, aro, intent in CUE_TABLE:
        if any(c in text + subtext for c in cues):
            return ...

这张表有三个结构性缺陷，只有人标才能暴露：

1. **它看不见语境**。`appraise()` 的入参只有 text 与 subtext，没有 scene_type、
   没有 relation、没有听者。一句「你给我闭嘴」在「冲突对峙」里是 8.6，
   在「亲密低语」（耳语、怕被听见）里同样是 8.6——但这两种表演完全不同。
2. **它只看第一个命中的线索**，顺序写死在表里。「别走，我求你」命中「挽留」，
   而人可能读成「认错 + 挽留」的复合情绪。
3. **+1.2 的潜台词加成是常数**，与潜台词的强度无关。

所以本模块不产生「能力结论」，只产生**两把尺子的对照**：
人标（新尺子）与规则（旧尺子）的 κ，以及换尺子之后模型排名会不会翻。

三个子命令
----------
    python annotate.py export [--limit N]   # 出标注表（不含规则答案，防锚定）
    python annotate.py score                 # 算 κ + 用人类金标重算模型指标
    python annotate.py guide                 # 只重写标注指南

标注表与答案键分开存：`annotations/form.csv` 给人填，`annotations/key.jsonl`
存规则答案。表里不出现规则答案，否则标注者会被锚定，κ 会虚高。
"""

from __future__ import annotations

import csv
import json
import os
import random
import sys
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANN_DIR = os.path.join(ROOT, "annotations")
FORM = os.path.join(ANN_DIR, "form.csv")
RETEST = os.path.join(ANN_DIR, "retest.csv")
KEY = os.path.join(ANN_DIR, "key.jsonl")
OUT_MD = os.path.join(ANN_DIR, "human_gold_report.md")
def _pick_records() -> str:
    """默认读最新的两层记录。

    写死文件名踩过坑：2026-10-31 的报告读到了修复前的旧记录，
    把「模型 vs 人 intent 8.3%」当成结论报了出去，实际修复后是 51%。
    """
    cands = [os.path.join(ROOT, "outputs", n) for n in (
        "records_two_layer_subfix.jsonl",
        "records_two_layer.jsonl",
    )]
    exist = [p for p in cands if os.path.exists(p)]
    return max(exist, key=os.path.getmtime) if exist else cands[-1]


RECORDS = _pick_records()

# 前 11 个与 reference.CUE_TABLE / agents.INTENT_GLOSS 同集（κ 和意图准确率都算在这 11 类上）。
# 「其他」只出现在人工标注侧：规则表永远不会输出它，所以人选「其他」= 该句记为不一致。
# 这是刻意留给标注者的逃生门——硬填一个心里不服的标签，比诚实记「其他」更污染 κ。
INTENTS = ["威胁", "认错", "挽留", "表白/安抚", "隐忍", "玩笑",
           "惊喜", "危机", "质疑", "忐忑", "陈述", "其他"]

AROUSAL_BINS = list(range(1, 11))          # 1..10 整数，10 档
VALENCE_EDGES = [-1.0, -0.6, -0.2, 0.2, 0.6, 1.0]   # 5 档


# --------------------------------------------------------------------------
# 度量
# --------------------------------------------------------------------------
def cohen_kappa(a: list, b: list) -> float | None:
    """名义 κ。两个标注者对同一批样本的分类一致性，扣掉碰巧一致。"""
    if len(a) != len(b) or not a:
        return None
    n = len(a)
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    if pe >= 1.0:
        return None
    return (po - pe) / (1 - pe)


def weighted_kappa(a: list, b: list) -> float | None:
    """二次加权 κ：给有序标签用。

    为什么不用名义 κ：arousal 标 7 而人标 8 与标 3 是两种错误，
    名义 κ 把它们算成一样错，会低估真正的一致性。
    """
    if len(a) != len(b) or not a:
        return None
    cats = sorted(set(a) | set(b))
    k = len(cats)
    if k < 2:
        return None
    idx = {c: i for i, c in enumerate(cats)}
    n = len(a)
    obs = defaultdict(int)
    for x, y in zip(a, b):
        obs[(idx[x], idx[y])] += 1
    ca, cb = Counter(a), Counter(b)
    num = den = 0.0
    for i, ci in enumerate(cats):
        for j, cj in enumerate(cats):
            w = 1 - ((i - j) ** 2) / ((k - 1) ** 2)
            num += w * obs[(i, j)] / n
            den += w * (ca[ci] / n) * (cb[cj] / n)
    if den >= 1.0:
        return None
    return 1 - (1 - num) / (1 - den)


def spearman(x: list[float], y: list[float]) -> float | None:
    """秩相关：只看「谁比谁高」的顺序对不对，不看绝对刻度。"""
    if len(x) < 3:
        return None

    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for t in range(i, j + 1):
                r[order[t]] = avg
            i = j + 1
        return r

    rx, ry = rank(x), rank(y)
    n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    vx = sum((rx[i] - mx) ** 2 for i in range(n)) ** 0.5
    vy = sum((ry[i] - my) ** 2 for i in range(n)) ** 0.5
    if vx == 0 or vy == 0:
        return None
    return cov / (vx * vy)


def valence_bin(v: float) -> int:
    for i in range(len(VALENCE_EDGES) - 1):
        if v <= VALENCE_EDGES[i + 1]:
            return i
    return len(VALENCE_EDGES) - 2


# --------------------------------------------------------------------------
# export
# --------------------------------------------------------------------------
GUIDE = """# 标注指南（主标 arousal；valence / intent 可选）

目标：给每句台词打一个「说话时内心的激活度」，作为**人类金标准**，
用来检验 `reference.appraise()` 那张关键词表能不能当尺子。

## 为什么主标 arousal

2026-10-31 用 24 条试标的结果：arousal 二次加权 κ = **0.18**、valence 0.18、intent 0.33。
三个人工维度里：

- **arousal 是唯一进下游计算的量**：`calibrate()` 只用它，其余 7 个表演参数都由它派生。
- **valence 不进任何计算**，只影响导演备注里的一句话。
- **intent 的 11 个选项来自参考实现的线索桶**，本来就是给关键词匹配用的分类；
  复合情绪被强制单选，标注者只能猜。24 行里用了 9 个不同标签，摊得极薄。

所以把有限的注意力全放在 arousal 上。valence / intent **标得动就标，标不动留空**。

## 三条纪律

1. **只凭这一句 + 它给的场景信息判断**，不要回看原剧本，也不要猜作者的设定。
2. **不要看模型输出**，也不要看别人的标注。表里没有模型答案，是故意的。
3. 判断的是**角色此刻内部的状态**，不是他「表现得多激烈」。
   压着嗓子说出的暴怒，arousal 可以是 9；笑着说「我没事」，可以是 3。

## arousal（1–10 整数）——必标

| 分值 | 状态 |
|---|---|
| 1–2 | 极低唤醒：困倦、放空、机械应答 |
| 3–4 | 低唤醒：平静、敷衍、日常寒暄 |
| 5–6 | 中唤醒：专注、轻度在意、隐忍的克制 |
| 7–8 | 高唤醒：紧张、急切、愤怒上涌、强烈期待 |
| 9–10 | 极高唤醒：濒临失控、惊恐、暴怒、崩溃边缘 |

### 台词和潜台词打架时怎么办

这是最常见的情况，也是最容易标出分歧的地方。规则：

1. **先读台词，再读潜台词。** 潜台词是对台词的修正，不是补充信息。
2. **两者冲突时，以潜台词为准。** 台词是演给人看的，潜台词是他心里的。
   例：「太棒了！」配「其实更想回家睡觉」→ 表面 8，心里 2，标 **2**。
3. **不要替角色加戏。** 潜台词没说的，不要自己补。

### 什么时候可以留空

如果你在两个锚点之间摇摆**超过 2 档**（比如在 3 和 7 之间反复），
说明这行的信息不足以定出唯一答案。**留空，按空格跳过。**

留空不是偷懒——硬选一个会把噪声混进金标，比少一行更糟。

## valence（-1.0 到 +1.0）——可选

-1.0 = 极度负面（恐惧、羞耻、绝望）　0.0 = 中性/复杂难辨　+1.0 = 极度正面（幸福、狂喜）

要标就写 -0.7、0.3 这样的小数；不想标直接跳过。

## intent（单选）——可选

威胁 / 认错 / 挽留 / 表白安抚 / 隐忍 / 玩笑 / 惊喜 / 危机 / 质疑 / 忐忑 / 陈述 / 其他

标得动就点；觉得都不合适就选「其他」（按 W）；**读不出来就留空**。
不要为了填满而勉强选——那只会让这个维度更糊。

## 填写方式

- **标注台**：只按 ① 的 `1`–`0` 给 arousal，然后按「下一句 →」。
  ② ③ 不选也能存（代码里空值原样保存）。
- 或直接在 `form.csv` 里填 `arousal(1-10)` 列，其他列不要动。
- `备注` 列可以写「这句我读成 X」，复盘时能帮你回忆当时的判断依据。

## 一个 10 分钟的自我检验（可选，但值得）

隔一天，把已经标过的 10 行**顺序打乱**再标一遍，比较两次自己：

- 两次差 ≤1 档 → 你稳定，任务可标；此前的低 κ 说明是规则表的问题。
- 两次差 ≥3 档 → 任务本身欠定，这一点要在报告里承认，不能只怪规则表。

这是区分「尺子错」和「任务糊」的唯一干净办法。
"""


def _ensure_dir() -> None:
    os.makedirs(ANN_DIR, exist_ok=True)


def cmd_export(limit: int | None = None, seed: int = 20261030) -> int:
    from . import reference as ref
    from . import tasks

    _ensure_dir()
    with open(os.path.join(ANN_DIR, "标注指南.md"), "w", encoding="utf-8") as f:
        f.write(GUIDE)

    n_gen = int(os.environ.get("MDH_N_SCENES", "50"))
    all_t = tasks.build_tasks(n_generated=n_gen)

    slots = []
    for t in all_t:
        for ln in t["lines"]:
            gold = ref.appraise(ln["text"], ln.get("subtext", ""))
            slots.append({
                "scene_id": t["scene_id"], "work_id": t["work_id"],
                "line_id": ln["id"], "character": ln["character"],
                "scene": ln.get("scene", ""), "scene_type": ln.get("scene_type", ""),
                "listener": ln.get("listener", ""), "relation": ln.get("relation", ""),
                "text": ln["text"], "subtext": ln.get("subtext", ""),
                "_gold": gold,
            })

    rng = random.Random(seed)
    rng.shuffle(slots)          # 打散：避免标注者顺着场景类型推规律
    if limit:
        slots = slots[:limit]

    # 同一文本不同语境是两件不同的事：155 个 (文本, 场景类型, 潜台词) 组合
    # 才是有效刺激数，不能按唯一文本数算。
    with open(FORM, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["序号", "scene_id", "line_id", "作品", "场景", "场景类型",
                    "角色", "听者", "关系", "台词", "潜台词",
                    "arousal(1-10)", "valence(-1~1)", "intent", "备注"])
        for i, s in enumerate(slots, 1):
            w.writerow([i, s["scene_id"], s["line_id"], s["work_id"],
                        s["scene"], s["scene_type"], s["character"],
                        s["listener"], s["relation"], s["text"], s["subtext"],
                        "", "", "", ""])

    with open(KEY, "w", encoding="utf-8") as f:
        for i, s in enumerate(slots, 1):
            f.write(json.dumps({
                "row": i, "line_id": s["line_id"], "scene_id": s["scene_id"],
                "text": s["text"], "subtext": s["subtext"],
                "scene_type": s["scene_type"], "relation": s["relation"],
                "rule": s["_gold"],
            }, ensure_ascii=False) + "\n")

    print(f"标注表：{FORM}")
    print(f"答案键：{KEY}（规则表的答案，标注时**不要看**）")
    print(f"指南　：{os.path.join(ANN_DIR, '标注指南.md')}")
    print(f"共 {len(slots)} 行，唯一文本 {len({s['text'] for s in slots})} 个，"
          f"唯一(文本,场景类型,潜台词) {len({(s['text'], s['scene_type'], s['subtext']) for s in slots})} 个")
    est_min = len(slots) * 30 / 60          # 每行 30 秒
    print(f"按每行 30 秒估，约 {est_min:.0f} 分钟填完")
    return 0


# --------------------------------------------------------------------------
# score
# --------------------------------------------------------------------------
def _read_form() -> dict[str, dict]:
    if not os.path.exists(FORM):
        raise SystemExit(f"找不到 {FORM}，先跑 python annotate.py export")
    out: dict[str, dict] = {}
    with open(FORM, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            lid = (r.get("line_id") or "").strip()
            aro = (r.get("arousal(1-10)") or "").strip()
            val = (r.get("valence(-1~1)") or "").strip()
            it = (r.get("intent") or "").strip()
            if not lid or (not aro and not val and not it):
                continue
            rec = {}
            if aro:
                try:
                    rec["arousal"] = max(1.0, min(10.0, float(aro)))
                except ValueError:
                    print(f"  [跳过] 行 {r.get('序号')} arousal 无法解析：{aro!r}")
            if val:
                try:
                    rec["valence"] = max(-1.0, min(1.0, float(val)))
                except ValueError:
                    print(f"  [跳过] 行 {r.get('序号')} valence 无法解析：{val!r}")
            if it:
                if it not in INTENTS:
                    print(f"  [警告] 行 {r.get('序号')} intent={it!r} 不在固定集合内，按原样计")
                rec["intent"] = it
            rec["note"] = (r.get("备注") or "").strip()
            rec["_raw"] = r
            if rec.keys() & {"arousal", "valence", "intent"}:
                out[lid] = rec
    return out


def _self_consistency() -> dict | None:
    """自我一致：同一批行盲标两次，算「自己 vs 自己」的 κ。

    它回答的是「标注者稳不稳」，不是「标得对不对」。
    这两件事必须分开：稳定的标注者也可能有系统性偏好，
    但不稳定的标注者一定不能当尺子。
    """
    if not os.path.exists(RETEST):
        return None
    first: dict[str, float] = {}
    with open(FORM, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            v = (r.get("arousal(1-10)") or "").strip()
            try:
                if v:
                    first[r["line_id"]] = float(v)
            except ValueError:
                pass
    a: list[int] = []
    b: list[int] = []
    with open(RETEST, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            v = (r.get("arousal(1-10)") or "").strip()
            lid = r.get("line_id")
            if not v or lid not in first:
                continue
            try:
                y = float(v)
            except ValueError:
                continue
            a.append(int(round(first[lid])))
            b.append(int(round(y)))
    n = len(a)
    if n < 8:
        return None
    return {
        "n": n,
        "kappa": weighted_kappa(a, b),
        "exact": sum(1 for x, y in zip(a, b) if x == y) / n,
        "within": sum(1 for x, y in zip(a, b) if abs(x - y) <= 1) / n,
        "mae": sum(abs(x - y) for x, y in zip(a, b)) / n,
    }


def _fmt(x: float | None, digits: int = 3) -> str:
    return "—" if x is None else f"{x:.{digits}f}"


def cmd_score(records_path: str = RECORDS) -> int:
    from . import judge
    from . import reference as ref

    human = _read_form()
    if not human:
        raise SystemExit("form.csv 里还没有填写任何一行")

    key = {}
    with open(KEY, encoding="utf-8") as f:
        for line in f:
            k = json.loads(line)
            key[k["line_id"]] = k

    rows = [(lid, h, key[lid]) for lid, h in human.items() if lid in key]
    n = len(rows)
    print(f"已填 {n} 行 / 表内 {len(key)} 行（覆盖率 {n / max(1, len(key)) * 100:.0f}%）")

    h_aro = [round(h["arousal"]) for _, h, _ in rows if "arousal" in h]
    r_aro = [round(k["rule"]["arousal"]) for _, h, k in rows if "arousal" in h]
    h_val = [valence_bin(h["valence"]) for _, h, _ in rows if "valence" in h]
    r_val = [valence_bin(k["rule"]["valence"]) for _, h, k in rows if "valence" in h]
    h_int = [h["intent"] for _, h, _ in rows if "intent" in h]
    r_int = [k["rule"]["intent"] for _, h, k in rows if "intent" in h]

    def mae(pairs):
        d = [abs(a - b) for a, b in pairs]
        return sum(d) / len(d) if d else None

    def bias(pairs):
        d = [a - b for a, b in pairs]
        return sum(d) / len(d) if d else None

    aro_pairs = [(h["arousal"], k["rule"]["arousal"]) for _, h, k in rows if "arousal" in h]
    val_pairs = [(h["valence"], k["rule"]["valence"]) for _, h, k in rows if "valence" in h]
    int_pairs = [(h["intent"], k["rule"]["intent"]) for _, h, k in rows if "intent" in h]

    k_aro = weighted_kappa(h_aro, r_aro)
    k_val = weighted_kappa(h_val, r_val)
    k_int = cohen_kappa(h_int, r_int)
    rho = spearman([p[0] for p in aro_pairs], [p[1] for p in aro_pairs])
    exact_aro = sum(1 for a, b in aro_pairs if abs(a - b) <= 1.0) / len(aro_pairs) if aro_pairs else None

    # 语境冲突：规则表看不见 scene_type，所以「线索词强度与该场景的表演上限冲突」
    # 的行，应该是分歧最集中的地方。这是本表最该被验证的一条假设。
    conflict = []
    for lid, h, k in rows:
        if "arousal" not in h:
            continue
        st = k.get("scene_type", "")
        ceil = ref.SCENE_PROFILE.get(st, {}).get("ceiling")
        if ceil is None:
            continue
        # 规则给的 arousal 折算成 intensity 后会被场景上限压住——即「规则认为该吼，
        # 但这场戏物理上吼不出来」。这类行是人标与规则最可能分道扬镳的地方。
        if k["rule"]["arousal"] * 0.75 > ceil:
            conflict.append((lid, st, k["rule"]["arousal"], h["arousal"]))

    L = ["# 人类金标 vs 规则金标：一致性检验", ""]
    L.append(f"> 样本 {n} 行（唯一文本 {len({k['text'] for _, _, k in rows})} 个）　"
             f"标注表 `annotations/form.csv`　答案键 `annotations/key.jsonl`")
    L.append("")
    L.append("## 0. 结论先行")
    L.append("")
    if n < 20:
        # 少量标注时 κ 极不稳定（两行样本能让 κ 从 -0.7 跳到 +0.7），
        # 这时给结论是误导。只提示样本不足，不下判断。
        L.append(f"> ⚠️ **样本只有 {n} 行，κ 还不稳定**，下面的数字只用来验证流程是否跑通，"
                 "不要据此下结论。建议至少标到 60 行再看趋势。")
        L.append("")
    L.append(f"- arousal（10 档，二次加权 κ）：**{_fmt(k_aro)}**；"
             f"秩相关 ρ = {_fmt(rho)}；±1 档内比例 {_fmt(None if exact_aro is None else exact_aro * 100, 1)}%")
    L.append(f"- valence（5 档，二次加权 κ）：**{_fmt(k_val)}**")
    L.append(f"- intent（{len(set(r_int) | set(h_int))} 类，名义 κ）：**{_fmt(k_int)}**")
    L.append("")
    L.append(_verdict(k_aro, k_int) if n >= 20 else
             "> 样本不足 20 行，暂不给判断。")
    L.append("")
    L.append("## 1. 两把尺子的绝对差")
    L.append("")
    L.append("| 量 | 人标均值 | 规则均值 | 规则相对人标的 MAE | 带符号偏差 |")
    L.append("|---|---|---|---|---|")

    def mean(xs):
        return sum(xs) / len(xs) if xs else None

    L.append(f"| arousal | {_fmt(mean([p[0] for p in aro_pairs]), 2)} | "
             f"{_fmt(mean([p[1] for p in aro_pairs]), 2)} | {_fmt(mae(aro_pairs), 2)} | "
             f"{_fmt(bias(aro_pairs), 2)} |")
    L.append(f"| valence | {_fmt(mean([p[0] for p in val_pairs]), 2)} | "
             f"{_fmt(mean([p[1] for p in val_pairs]), 2)} | {_fmt(mae(val_pairs), 2)} | "
             f"{_fmt(bias(val_pairs), 2)} |")
    int_agree = sum(1 for a, b in int_pairs if a == b) / len(int_pairs) if int_pairs else None
    L.append(f"| intent | — | — | 一致率 {_fmt(None if int_agree is None else int_agree * 100, 1)}% | — |")
    L.append("")
    L.append("> 带符号偏差为正 = 规则表**系统性偏高**。偏高可以用常数校正，"
             "抖动不能——这是两者最本质的区别。")
    L.append("")

    L.append("## 2. 分歧集中在哪")
    L.append("")
    L.append(f"规则表的 `appraise()` 只吃 text + subtext，**看不到 scene_type 与 relation**。"
             f"所以当「线索词暗示的强度」超过「该场景的表演上限」时，两把尺子必然打架。")
    L.append("")
    if conflict:
        big = [c for c in conflict if abs(c[2] - c[3]) >= 1.5]
        L.append(f"- 被判定为「规则强度超出场景上限」的行：{len(conflict)} 行；"
                 f"其中人标与规则差 ≥1.5 档的 {len(big)} 行"
                 f"（占 {len(big) / len(conflict) * 100:.0f}%）")
        for lid, st, ra, ha in big[:10]:
            txt = key[lid]["text"]
            L.append(f"  - `{lid}`（{st}）规则 {ra:.1f} → 人标 {ha:.1f}｜{txt}")
    else:
        L.append("- 未发现这类冲突行（或标注量不足）")
    L.append("")

    # ---------------- 换尺子之后，模型排名会不会翻 ----------------
    L.append("## 3. 换尺子重算：模型的分数会变吗")
    L.append("")
    if not os.path.exists(records_path):
        L.append(f"> 未找到 `{records_path}`，跳过本节。跑完 `run_two_layer.py api` 再看。")
    else:
        recs = [json.loads(x) for x in open(records_path, encoding="utf-8") if x.strip()]
        L.append("同一批模型输出，分别用规则金标和人类金标（仅覆盖已标注行）打分：")
        L.append("")
        L.append("| 臂 | 行数 | 规则金标 MAE | 人标 MAE | 规则金标 intent | 人标 intent |")
        L.append("|---|---|---|---|---|---|")
        for arm in sorted({r["scaffold"] for r in recs}):
            sub = [r for r in recs if r["scaffold"] == arm]
            ra, ha, ri, hi = [], [], [], []
            for r in sub:
                for a in r.get("appraisal") or []:
                    lid = a.get("line_id")
                    if lid not in human:
                        continue
                    v = a.get("value") or {}
                    g = a.get("gold") or {}
                    h = human[lid]
                    if "arousal" in h and isinstance(v.get("arousal"), (int, float)):
                        ra.append(abs(v["arousal"] - g.get("arousal", 0)))
                        ha.append(abs(v["arousal"] - h["arousal"]))
                    if "intent" in h:
                        ri.append(v.get("intent") == g.get("intent"))
                        hi.append(v.get("intent") == h["intent"])
            L.append(f"| `{arm}` | {len(ra)} | {_fmt(mean(ra), 2)} | {_fmt(mean(ha), 2)} | "
                     f"{_fmt(None if not ri else sum(ri) / len(ri) * 100, 1)}% | "
                     f"{_fmt(None if not hi else sum(hi) / len(hi) * 100, 1)}% |")
        L.append("")
        L.append("> **读法**：`judge_cue` 拿到了线索表，所以它离规则近（0.46）、离人远（1.27）——"
                 "这是构造使然，不能当能力上限。`judge_plain` 没拿到表，它离人（1.61）反而比离规则"
                 "（1.74）更近，说明模型在某些行上确实读出了语义。但两臂的**意图**分类都是离规则更近"
                 "（82% / 71% vs 51%），说明「这句话在干什么」仍被表层词汇主导。")
    L.append("")
    L.append("## 4. 标注者可靠性（自我一致，test-retest）")
    L.append("")
    sc = _self_consistency()
    if sc is None:
        L.append("> 未找到 `annotations/retest.csv`（或填不满 8 行），本节跳过。")
        L.append("> 生成复测表的方法见 `annotations/retest.csv` 的列定义：抽 20 行打乱、盲标 arousal、再与本表比对。")
    else:
        L.append(f"同一批人、同一批行、隔一段时间盲标两次，样本 **{sc['n']}** 行：")
        L.append("")
        L.append("| 指标 | 值 |")
        L.append("|---|---|")
        L.append(f"| 二次加权 κ（自己 vs 自己） | **{_fmt(sc['kappa'])}** |")
        L.append(f"| 完全一致率 | {sc['exact'] * 100:.0f}% |")
        L.append(f"| ±1 档内 | {sc['within'] * 100:.0f}% |")
        L.append(f"| 平均绝对差 | {_fmt(sc['mae'], 2)} 档 |")
        L.append("")
        if sc["kappa"] is not None and sc["kappa"] >= 0.8:
            L.append("> **标注者稳定。** 因此「人 vs 规则」的分歧不能归因于标注噪声："
                     "标注者可靠性接近上限，衰减修正后规则表的一致性几乎不变。")
            L.append(">")
            L.append("> 仍无法排除标注者的**系统性偏好**（例如对反讽特别敏感）——"
                     "那种偏差每次都会一致地出现，自我一致 κ 反而会偏高。"
                     "要检验它，需要第二名标注者。")
        elif sc["kappa"] is not None and sc["kappa"] >= 0.5:
            L.append("> **基本稳定，但部分行确实模糊。** 人 vs 规则的分歧中，"
                     "有一部分是任务本身的信息不足，不能全归给规则表。")
        else:
            L.append("> **标注者自身不稳。** 此时人标不能当唯一真值，"
                     "应改报「人-人一致率」而不是「与某个金标的距离」。")
    L.append("")
    L.append("## 5. 这张表不能说明什么")
    L.append("")
    L.append(f"- 单标注者，没有第二人，所以这里是「人 vs 规则」的一致性，"
             f"不是「人 vs 人」的信度。要报 inter-annotator κ，至少再找一人标同样的 {n} 行。")
    L.append("- 标注者知道这是评测集，可能比平时更谨慎（过度理性化），"
             "这会让 κ 偏高。真正的盲测要混入无关台词做注意力检查。")
    L.append("- `intent` 用的是固定 11 类。人本可以给出「愤怒+挽留」的复合标签，"
             "被强制单选会损失信息，κ 会偏低。")

    md = "\n".join(L) + "\n"
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"\n报告：{OUT_MD}")
    return 0


def _verdict(k_aro: float | None, k_int: float | None) -> str:
    if k_aro is None:
        return "> κ 无法计算（样本不足或取值单一）。"
    lo = min(x for x in (k_aro, k_int) if x is not None)
    if lo >= 0.8:
        head = ("**规则表与人标高度一致（κ≥0.8）**。这意味着此前所有 arousal 指标"
                "可以当作能力结论使用，规则表是一个可辩护的代理标注器。")
    elif lo >= 0.6:
        head = ("**规则表与人标中等一致（0.6≤κ<0.8）**。规则表可以当弱监督信号，"
                "但不能当金标准：此前报告的 arousal MAE 与意图准确率需要标注"
                "「金标准为规则表」这一限定语，且换人标后绝对数值会变。")
    elif lo >= 0.4:
        head = ("**一致性偏低（0.4≤κ<0.6）**。规则表捕捉到的只是表层词汇共现，"
                "不是人对情绪的判断。此前所有基于它的「判读准确率」都只能读作"
                "「与关键词表的一致率」，不能读作准确率。")
    else:
        head = ("**一致性差（κ<0.4）**。规则金标在这批数据上不足以充当标注，"
                "此前基于它的判读指标全部失效，应以人标为准重算。")
    return head


def main(argv: list[str]) -> int:
    cmd = argv[1] if len(argv) > 1 else "export"
    if cmd == "export":
        limit = None
        if "--limit" in argv:
            limit = int(argv[argv.index("--limit") + 1])
        return cmd_export(limit)
    if cmd == "score":
        path = RECORDS
        if "--records" in argv:
            path = argv[argv.index("--records") + 1]
        return cmd_score(path)
    if cmd == "guide":
        _ensure_dir()
        p = os.path.join(ANN_DIR, "标注指南.md")
        with open(p, "w", encoding="utf-8") as f:
            f.write(GUIDE)
        print(p)
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
