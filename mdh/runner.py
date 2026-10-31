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
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import agents, judge
from . import reference as ref


def fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _run_one(task: dict, scaffold: str, rep: int, golds: dict, ex,
             llm, max_repair: int, tau: float) -> dict:
    """跑一次并判卷。单条失败不掀桌——记成 passed=False 入档，事后能查到是哪条。"""
    try:
        out = agents.run_scaffold(task, scaffold, llm, examples=ex,
                                  max_repair=max_repair, repeat=rep)
    except Exception as e:  # noqa: BLE001 - 网络/接口错误都要留痕，不能中断整批
        return {
            "scene_id": task["scene_id"], "work_id": task["work_id"],
            "scaffold": scaffold, "repeat": rep, "tau": tau,
            "passed": False, "hit_rate": 0.0, "min_hit_rate": 0.0,
            "l1_clean": False, "errors": [f"调用失败: {e}"],
            "rows": [], "violations": [], "attempts": 0,
            "appraisal": None, "clamps": None,
            "tokens_in": 0, "tokens_out": 0, "trace": [],
            "fingerprint": fingerprint(json.dumps(
                {"scene": task["scene_id"], "scaffold": scaffold, "repeat": rep},
                ensure_ascii=False)),
        }

    j = judge.judge_scene(out["output"], task, golds, tau=tau)

    # 判读层（仅两层架构的臂有）：单独判，避免和派生参数的命中率混在一起
    appr = out.get("appraisal")
    appr_rows: list[dict] | None = None
    if appr is not None:
        raw = appr.get("lines") if isinstance(appr.get("lines"), list) else appr.get("shots")
        by_id = {x.get("line_id"): x for x in (raw or []) if isinstance(x, dict)}
        appr_rows = []
        for ln in task["lines"]:
            item = by_id.get(ln["id"])
            gold_i = golds[ln["id"]].get("_internal") or {}
            if item is None:
                appr_rows.append({"line_id": ln["id"], "hits": {}, "deltas": {},
                                  "signed": {}, "value": {}, "gold": gold_i})
                continue
            ja = judge.judge_appraisal(item, ln, golds[ln["id"]])
            appr_rows.append({
                "line_id": ln["id"], "hits": ja["hits"], "deltas": ja["deltas"],
                "signed": ja.get("signed", {}),
                "value": {"arousal": item.get("arousal"),
                          "valence": item.get("valence"),
                          "intent": item.get("intent")},
                "gold": gold_i,
            })

    return {
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
        "appraisal": appr_rows,
        "clamps": (out["output"] or {}).get("_clamps") if isinstance(out["output"], dict) else None,
        "attempts": out["attempts"],
        "tokens_in": out["tokens_in"],
        "tokens_out": out["tokens_out"],
        "trace": out["trace"],
        "fingerprint": fingerprint(json.dumps(
            {"scene": task["scene_id"], "scaffold": scaffold, "repeat": rep},
            ensure_ascii=False)),
    }


def run_matrix(tasks: list[dict], scaffolds: list[str], llm, k: int = 5,
               tau: float = judge.DEFAULT_TAU, max_repair: int = 2,
               examples_for=None, verbose: bool = True,
               workers: int = 1) -> list[dict]:
    """跑「任务 × 脚手架 × k 次」。

    workers>1 时并发发送请求。后端 OpenAICompat 每次调用都是独立的
    urllib 请求、没有共享可变状态，所以线程安全；限流由 llm 层退避兜底。
    结果按 jobs 的原始顺序返回，不受完成先后影响，保证可复现。
    """
    t0 = time.time()
    total = len(tasks) * len(scaffolds) * k

    jobs = []
    for task in tasks:
        golds = {ln["id"]: ref.gold_shot(ln, task["characters"]) for ln in task["lines"]}
        for scaffold in scaffolds:
            ex = examples_for(task, scaffold) if examples_for else None
            for rep in range(k):
                jobs.append((task, scaffold, rep, golds, ex))

    records: list[dict] = []
    lock = threading.Lock()

    def note(rec: dict) -> None:
        records.append(rec)
        if verbose and (len(records) % 10 == 0 or len(records) == total):
            print(f"  [{len(records)}/{total}] {rec['scaffold']:14s} {rec['scene_id']:12s} "
                  f"rep{rec['repeat']} hit={rec['hit_rate']:.2f} pass={rec['passed']}")

    if workers <= 1:
        for job in jobs:
            note(_run_one(*job, llm=llm, max_repair=max_repair, tau=tau))
    else:
        slots: list[dict | None] = [None] * len(jobs)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_run_one, *job, llm=llm,
                            max_repair=max_repair, tau=tau): i
                for i, job in enumerate(jobs)
            }
            for fut in as_completed(futures):
                slots[futures[fut]] = fut.result()
                with lock:
                    note(slots[futures[fut]])
        records = [r for r in slots if r is not None]

    if verbose:
        failed = sum(1 for r in records if any("调用失败" in str(e) for e in r["errors"]))
        print(f"  完成 {total} 次运行，用时 {time.time() - t0:.1f}s"
              + (f"，调用失败 {failed} 次（已记入 records）" if failed else ""))
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
            "appraisal": r.get("appraisal"), "clamps": r.get("clamps"),
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
