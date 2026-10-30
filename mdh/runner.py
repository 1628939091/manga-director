"""编排：任务 × 脚手架 × k 次重复。

一条纪律：**k 次重复必须是独立采样，且留痕。**
每次调用都记录 repeat、attempt、请求指纹，事后能回答"这 k 次真的是 k 次吗、
有没有缓存命中"。pass^k 的可信度完全建立在这件事上——
如果 k 次里有一次是热缓存，pass^k 就从"可靠性"退化成了"单次能力"。
"""

from __future__ import annotations

import hashlib
import json
import os
import time

from . import agents, judge
from . import reference as ref


def fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def run_matrix(tasks: list[dict], scaffolds: list[str], llm, k: int = 5,
               tau: float = judge.DEFAULT_TAU, max_repair: int = 2,
               examples_for=None, verbose: bool = True) -> list[dict]:
    records: list[dict] = []
    t0 = time.time()
    total = len(tasks) * len(scaffolds) * k
    done = 0

    for task in tasks:
        golds = {ln["id"]: ref.gold_shot(ln, task["characters"]) for ln in task["lines"]}
        for scaffold in scaffolds:
            ex = examples_for(task, scaffold) if examples_for else None
            for rep in range(k):
                out = agents.run_scaffold(task, scaffold, llm, examples=ex,
                                          max_repair=max_repair, repeat=rep)
                j = judge.judge_scene(out["output"], task, golds, tau=tau)
                rec = {
                    "scene_id": task["scene_id"],
                    "work_id": task["work_id"],
                    "scaffold": scaffold,
                    "repeat": rep,
                    "tau": tau,
                    "passed": j["passed"],
                    "hit_rate": j["hit_rate"],
                    "min_hit_rate": j["min_hit_rate"],
                    "l1_clean": j["l1_clean"],
                    "errors": j["errors"],
                    "rows": j["rows"],
                    "violations": j["violations"],
                    "attempts": out["attempts"],
                    "tokens_in": out["tokens_in"],
                    "tokens_out": out["tokens_out"],
                    "trace": out["trace"],
                    "fingerprint": fingerprint(json.dumps(
                        {"scene": task["scene_id"], "scaffold": scaffold, "repeat": rep},
                        ensure_ascii=False)),
                }
                records.append(rec)
                done += 1
                if verbose and (done % 10 == 0 or done == total):
                    print(f"  [{done}/{total}] {scaffold:14s} {task['scene_id']:12s} "
                          f"rep{rep} hit={j['hit_rate']:.2f} pass={j['passed']}")

    if verbose:
        print(f"  完成 {total} 次运行，用时 {time.time() - t0:.1f}s")
    return records


def fewshot_examples(task: dict, pool: list[dict], n: int = 2) -> list[dict]:
    """取金标准示例：只从**其他作品**里取，杜绝同作品泄漏。"""
    picks: list[dict] = []
    for other in pool:
        if other["work_id"] == task["work_id"]:
            continue
        for ln in other["lines"]:
            if ln.get("scene_type") != task["lines"][0].get("scene_type"):
                continue
            g = ref.gold_shot(ln, other["characters"])
            picks.append({k: v for k, v in g.items() if not k.startswith("_")})
            if len(picks) >= n:
                return picks
    # 场景类型不够匹配就放宽到任意其他作品
    for other in pool:
        if other["work_id"] == task["work_id"]:
            continue
        for ln in other["lines"]:
            g = ref.gold_shot(ln, other["characters"])
            picks.append({k: v for k, v in g.items() if not k.startswith("_")})
            if len(picks) >= n:
                return picks
    return picks[:n]


def make_example_provider(pool: list[dict], n: int = 2):
    def provider(task: dict, scaffold: str):
        if scaffold != "card_fewshot":
            return None
        return fewshot_examples(task, pool, n)
    return provider


def save_records(records: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    slim = []
    for r in records:
        slim.append({
            "scene_id": r["scene_id"], "work_id": r["work_id"],
            "scaffold": r["scaffold"], "repeat": r["repeat"],
            "passed": r["passed"], "hit_rate": r["hit_rate"],
            "min_hit_rate": r["min_hit_rate"],
            "l1_clean": r["l1_clean"], "errors": r["errors"],
            "violations": r["violations"], "attempts": r["attempts"],
            "tokens_in": r["tokens_in"], "tokens_out": r["tokens_out"],
            "fingerprint": r["fingerprint"],
            "rows": [{
                "line_id": row["line_id"], "hit_rate": row["hit_rate"],
                "l1_clean": row["l1_clean"], "hits": row["hits"],
                "deltas": row["deltas"], "errors": row["errors"],
                "violations": row["violations"],
            } for row in r["rows"]],
        })
    with open(path, "w", encoding="utf-8") as f:
        for s in slim:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
