"""指标层：pass@1 / pass^k / 命中率 / 违反率 / 聚类 Bootstrap。

三个必须讲清的统计口径
--------------------
1. **pass@1 与 pass^k 是回答两个不同的问题。**
   pass@1 问"一次能不能过"，pass^k 问"每次都能过吗"。
   生产环境要的是后者：一场戏错了，观众看到的就是错的。
2. **CI 必须按场景聚类。** 同一场戏的多条台词不独立（共用角色卡与场景规范），
   把行当独立样本会把 CI 算窄——这是项目一 deff≈3.3 那个坑的同构版本。
3. **k 次重复不是样本量。** n 个场景 × k 次重复的有效样本量接近 n，
   不是 n·k。报告里所有 CI 一律按场景重采样。
"""

from __future__ import annotations

import random

from . import reference as ref
from .judge import DEFAULT_TOL as TOL_BASE


# --------------------------------------------------------------------------
# 聚合
# --------------------------------------------------------------------------
def scene_pass_table(records: list[dict]) -> dict[tuple[str, str], list[bool]]:
    """(scaffold, scene_id) -> 按 repeat 排序的 pass 列表。"""
    table: dict[tuple[str, str], list[tuple[int, bool]]] = {}
    for r in records:
        table.setdefault((r["scaffold"], r["scene_id"]), []).append((r["repeat"], r["passed"]))
    return {k: [p for _, p in sorted(v)] for k, v in table.items()}


def pass_at_1(records: list[dict], scaffold: str) -> float:
    ps = [r["passed"] for r in records if r["scaffold"] == scaffold]
    return sum(ps) / len(ps) if ps else 0.0


def pass_pow_k(records: list[dict], scaffold: str) -> float:
    """pass^k：每场戏的 k 次重复**全部**通过的比例。"""
    table = scene_pass_table([r for r in records if r["scaffold"] == scaffold])
    if not table:
        return 0.0
    ok = [all(v) for v in table.values() if v]
    return sum(ok) / len(ok) if ok else 0.0


def violation_rate(records: list[dict], scaffold: str) -> float:
    """禁令违反率：违反的**行**数 / 总行数（不是场数，场会把问题稀释）。"""
    bad = total = 0
    for r in records:
        if r["scaffold"] != scaffold:
            continue
        for row in r["rows"]:
            total += 1
            if row["violations"]:
                bad += 1
    return bad / total if total else 0.0


def l1_clean_rate(records: list[dict], scaffold: str) -> float:
    rs = [r for r in records if r["scaffold"] == scaffold]
    return sum(1 for r in rs if r["l1_clean"]) / len(rs) if rs else 0.0


def mean_hit_rate(records: list[dict], scaffold: str) -> float:
    rs = [r for r in records if r["scaffold"] == scaffold]
    return sum(r["hit_rate"] for r in rs) / len(rs) if rs else 0.0


def line_level_pass(records: list[dict], scaffold: str, tau: float = 0.8) -> float:
    """行级 pass（不计场景连乘）——用来暴露"场景级可靠性被连乘吃掉"。"""
    ok = total = 0
    for r in records:
        if r["scaffold"] != scaffold:
            continue
        for row in r["rows"]:
            total += 1
            if row["l1_clean"] and row["hit_rate"] >= tau:
                ok += 1
    return ok / total if total else 0.0


def mae_by_field(records: list[dict], scaffold: str) -> dict[str, float]:
    acc: dict[str, list[float]] = {k: [] for k in ref.JUDGED_NUMERIC}
    for r in records:
        if r["scaffold"] != scaffold:
            continue
        for row in r["rows"]:
            for k, d in row.get("deltas", {}).items():
                if k in acc:
                    acc[k].append(d)
    return {k: (sum(v) / len(v) if v else 0.0) for k, v in acc.items()}


def mean_tokens(records: list[dict], scaffold: str) -> float:
    rs = [r for r in records if r["scaffold"] == scaffold]
    return sum(r["tokens_in"] + r["tokens_out"] for r in rs) / len(rs) if rs else 0.0


def mean_attempts(records: list[dict], scaffold: str) -> float:
    rs = [r for r in records if r["scaffold"] == scaffold]
    return sum(r["attempts"] for r in rs) / len(rs) if rs else 0.0


# --------------------------------------------------------------------------
# 聚类 Bootstrap
# --------------------------------------------------------------------------
def cluster_bootstrap_ci(values_by_cluster: dict[str, list[float]],
                         stat: str = "mean", B: int = 2000,
                         alpha: float = 0.05, seed: int = 20261030):
    """按簇（场景）重采样。返回 (lo, hi, point)。"""
    keys = list(values_by_cluster)
    if not keys:
        return (0.0, 0.0, 0.0)
    flat = [x for k in keys for x in values_by_cluster[k]]
    point = (sum(flat) / len(flat)) if stat == "mean" else (
        sum(1 for x in flat if x) / len(flat))

    rng = random.Random(seed)
    ests = []
    for _ in range(B):
        sample = [rng.choice(keys) for _ in keys]
        vals = [x for k in sample for x in values_by_cluster[k]]
        if not vals:
            continue
        ests.append(sum(vals) / len(vals))
    ests.sort()
    lo = ests[int(alpha / 2 * len(ests))]
    hi = ests[int((1 - alpha / 2) * len(ests)) - 1]
    return (round(lo, 4), round(hi, 4), round(point, 4))


def pass_ci(records: list[dict], scaffold: str) -> tuple[float, float, float]:
    """场景级 pass@1 的聚类 CI（簇 = 场景）。"""
    by_scene: dict[str, list[float]] = {}
    for r in records:
        if r["scaffold"] == scaffold:
            by_scene.setdefault(r["scene_id"], []).append(1.0 if r["passed"] else 0.0)
    return cluster_bootstrap_ci(by_scene)


# --------------------------------------------------------------------------
# 敏感性：结论扛不扛得住换阈值
# --------------------------------------------------------------------------
def tau_sensitivity(records: list[dict], scaffold: str, taus=(0.5, 0.6, 0.7, 0.8, 0.9, 1.0)) -> dict:
    """重算 pass。口径与 judge_scene 一致：**每一行**都要过，不是平均过。"""
    out = {}
    for tau in taus:
        by_scene: dict[str, list[bool]] = {}
        for r in records:
            if r["scaffold"] != scaffold:
                continue
            p = bool(r["l1_clean"] and r["min_hit_rate"] >= tau)
            by_scene.setdefault(r["scene_id"], []).append(p)
        pass1 = sum(sum(v) for v in by_scene.values()) / sum(len(v) for v in by_scene.values())
        passk = sum(1 for v in by_scene.values() if all(v)) / len(by_scene)
        out[tau] = {"pass@1": round(pass1, 3), "pass^k": round(passk, 3)}
    return out


def tol_scale_sensitivity(records: list[dict], scaffold: str, scale: float,
                          tau: float = 0.8) -> dict:
    """把所有数值容差乘以 scale，重算 pass。检验"结论是不是我调容差调出来的"。"""
    n_pass = n_tot = 0
    for r in records:
        if r["scaffold"] != scaffold:
            continue
        n_tot += 1
        ok = r["l1_clean"]
        if ok:
            # 每一行都要过（与 judge_scene 同口径），不用平均
            for row in r["rows"]:
                if not row.get("deltas"):
                    ok = False
                    break
                hits = [d <= TOL_BASE[k] * scale for k, d in row["deltas"].items()]
                hits += [bool(row["hits"].get("style")), bool(row["hits"].get("gaze"))]
                if sum(hits) / len(hits) < tau:
                    ok = False
                    break
        n_pass += 1 if ok else 0
    return {"scale": scale, "pass@1": round(n_pass / n_tot, 3) if n_tot else 0.0}
