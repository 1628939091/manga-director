"""任务定义与模板化生成。

一个任务 = 一场戏（scene）。判卷以"场"为单位、以"行"为粒度：
行级命中率是部分分，场级 pass 才是 pass^k 的原子事件。

为什么要模板化生成（而不是攒 10 个手写场景）
------------------------------------------
手写场景无法做消融：你没法控制"性格 × 场景 × 关系"的分布，
也就不可能回答"增益来自角色卡，还是来自台词本身更好懂"。
模板化把任务变成**受控因子组合**，这是能出实验结论的前提。

为什么要按作品（work_id）切分
-----------------------------
同一部作品里角色卡相同、说话风格相同，随机按场景切分必然泄漏。
按作品切分是评测基准建设的基本纪律（项目二踩过的坑在任务层面复现一次）。
"""

from __future__ import annotations

import json
import os
import random

from . import lines

# --------------------------------------------------------------------------
# 角色原型（性格四维 → 决定表演强度上限，是金标准的核心输入）
# --------------------------------------------------------------------------
ARCHETYPES = {
    "内敛男主": {"role": "男主", "voice": "低沉青年男声，气声偏多",
             "personality": {"warmth": 0.35, "dominance": 0.65, "energy": 0.25, "neuroticism": 0.55}},
    "外放女主": {"role": "女主", "voice": "清亮少女声",
             "personality": {"warmth": 0.80, "dominance": 0.35, "energy": 0.85, "neuroticism": 0.45}},
    "威严师父": {"role": "师父", "voice": "中年沙哑男声",
             "personality": {"warmth": 0.50, "dominance": 0.90, "energy": 0.75, "neuroticism": 0.20}},
    "怯懦后辈": {"role": "后辈", "voice": "偏细的少女声",
             "personality": {"warmth": 0.70, "dominance": 0.15, "energy": 0.30, "neuroticism": 0.80}},
    "阴鸷对手": {"role": "对手", "voice": "偏低的男声，气息冷",
             "personality": {"warmth": 0.15, "dominance": 0.85, "energy": 0.40, "neuroticism": 0.35}},
}

# 场景类型 -> (现场描述, 候选台词, 关系候选)
SCENE_TEMPLATES = {
    "冲突对峙": {
        "where": ["演武场，众人围观", "宗门大殿，长老列席", "雨夜巷口，两人对峙"],
        "lines": ["你说什么？再说一次！", "闭嘴。", "最后一遍，让开。"],
        "relations": ["长辈", "对手", "上级"],
    },
    "亲密低语": {
        "where": ["雨夜屋檐下", "深夜的病房走廊", "关灯后的宿舍"],
        "lines": ["……我知道了。", "别走。", "我刚才说的那些，你听到了吗？"],
        "relations": ["恋人未满", "恋人", "挚友"],
    },
    "日常": {
        "where": ["街头，看到流浪猫", "便利店门口", "放学后的操场"],
        "lines": ["哇——好可爱！", "嗯，随便。", "真的假的？"],
        "relations": ["挚友", "陌生人", "恋人"],
    },
    "危机": {
        "where": ["塌方的矿道", "燃烧的仓库", "失控的列车车厢"],
        "lines": ["快跑！来不及了。", "血……好多血。", "别回头，跑！"],
        "relations": ["挚友", "陌生人", "上级"],
    },
    "独白": {
        "where": ["空荡的旧居", "天台，风很大", "停摆的钟表店"],
        "lines": ["为什么。到底为什么。", "我认。", "……随你。"],
        "relations": ["陌生人", "挚友"],
    },
    "玩笑打闹": {
        "where": ["社团活动室", "厨房，面粉满天飞", "教室后排"],
        "lines": ["哈哈，笑死我了。", "你再这样我真生气了。", "好厉害！"],
        "relations": ["挚友", "恋人", "对手"],
    },
}


DEMO_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                         "data", "demo_scene.json")


# --------------------------------------------------------------------------
def make_scene(scene_id: str, work_id: str, title: str, cast: list[str],
               lines: list[dict]) -> dict:
    return {
        "scene_id": scene_id,
        "work_id": work_id,
        "title": title,
        "characters": {name: ARCHETYPES[name] for name in cast},
        "lines": lines,
    }


def generate_scenes(n: int = 50, seed: int = 20261030, scenes_per_work: int = 2) -> list[dict]:
    """受控因子组合生成场景。确定性（固定 seed），可复现。

    与 v0 的三个区别：
    1. 台词从 `lines.SCENE_LINES` 取，**同一作品内不重复**（v0 同类型场景台词完全相同）
    2. 场景类型按轮次配平，且每部作品的"类型组合"错开，避免类型与 work_id 混淆
    3. 潜台词分中性/紧张两档，紧张档会触发金标准的 +1.2 加成，
       使 arousal 取值从 10 个扩到 15 个——判读任务不再是"十选一"
    """
    rng = random.Random(seed)
    scene_types = list(lines.SCENE_LINES)
    cast_pool = list(ARCHETYPES)
    scenes: list[dict] = []
    used_by_work: dict[str, set[str]] = {}

    for i in range(n):
        # 轮次配平：50 场 / 6 类 ≈ 每类 8–9 场；偏移量让不同作品的类型组合不完全相同
        st = scene_types[(i + i // len(scene_types)) % len(scene_types)]
        tpl = SCENE_TEMPLATES[st]
        where = tpl["where"][(i // len(scene_types)) % len(tpl["where"])]
        work_id = f"gen{i // scenes_per_work:02d}"
        scene_id = f"{work_id}-S{i % scenes_per_work + 1:02d}"

        cast = rng.sample(cast_pool, 2)
        while len(cast) < 3:
            c = rng.choice(cast_pool)
            if c not in cast:
                cast.append(c)

        used = used_by_work.setdefault(work_id, set())
        pool = [pr for pr in lines.SCENE_LINES[st] if pr[0] not in used]
        n_lines = rng.choice([3, 3, 4])
        if len(pool) < n_lines:            # 库不够时退回全库（正常配置下不会触发）
            used.clear()
            pool = list(lines.SCENE_LINES[st])
        picks = rng.sample(pool, n_lines)

        line_rows = []
        for j, (text, sub_neutral, sub_tense) in enumerate(picks):
            used.add(text)
            speaker = cast[j % len(cast)]
            listener = cast[(j + 1) % len(cast)]
            relation = rng.choice(tpl["relations"])
            # 让关系修正只在合理的方向上生效（长辈/上级不会出现在恋人关系里）
            if relation in ("恋人", "恋人未满") and speaker == "威严师父":
                relation = "长辈"
            # 潜台词与台词逐句配对（lines.py 里成对写好），不再随机抽短语：
            # 随机抽会把「表面平静，心里没底」配给「这猫好可爱」，而潜台词
            # 会进 reference.appraise() 的加成分支——等于用随机数改写金标
            tense = (i + j) % 4 == 0
            subtext = sub_tense if tense else sub_neutral
            line_rows.append({
                "id": f"{scene_id}-L{j + 1:02d}",
                "character": speaker,
                "scene": where,
                "scene_type": st,
                "listener": listener,
                "relation": relation,
                "text": text,
                "subtext": subtext,
            })

        scenes.append(make_scene(scene_id, work_id, f"{where}·{st}", cast, line_rows))

    return scenes


def load_demo_scene(path: str = DEMO_PATH) -> list[dict]:
    """真实手写场景（来自 manga-performance-director/scripts/demo_script.json）。"""
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return [{
        "scene_id": "demo-S01",
        "work_id": "demo_yuye",
        "title": raw.get("title", "示例漫剧"),
        "characters": raw["characters"],
        "lines": raw["lines"],
    }]


def build_tasks(n_generated: int = 12, seed: int = 20261030) -> list[dict]:
    return load_demo_scene() + generate_scenes(n_generated, seed)


def split_by_work(tasks: list[dict], holdout_ratio: float = 0.25) -> dict[str, list[dict]]:
    """按 work_id 切分，保证同一作品不会跨 train/test。"""
    works = sorted({t["work_id"] for t in tasks})
    n_hold = max(1, round(len(works) * holdout_ratio)) if len(works) > 1 else 0
    hold = set(works[-n_hold:]) if n_hold else set()
    train = [t for t in tasks if t["work_id"] not in hold]
    test = [t for t in tasks if t["work_id"] in hold]
    return {"train": train, "test": test}
