"""读回分析产物：把 `outputs/analysis_two_layer.md` 里的数字解析成结构化数据。

为什么要有这一层
----------------
页面和 README 必须引用**同一份**数字。页面如果自己从 records 现算，就会出现第二套
口径：现算是 0.40 / 0.55 / 0.77 / 0.93，产物 md 是 0.42 / 0.53 / 0.78 / 0.96。
差 0.01~0.03 看着小，但两处并存，评审者一对账就会怀疑整份评测的可靠性。

所以 web 层只做「读」，不做「算」：唯一真值是 `mdh.analysis.render_two_layer`
落盘的 md。这个模块只负责把那份 markdown 解析回来，零副作用、零依赖。

解析失败时**不兜底估算**——直接报错并指出缺哪块。悄悄退回现算，等于把两套口径
的坑重新挖回来。
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

# 因子表里的两条线（md 里的行首标签）
_LAYER_SINGLE = "单层"
_LAYER_TWO = "两层"

# `> 模型：-　k = 3　臂：...　记录 306 次运行`
_K_RE = re.compile(r"k\s*=\s*(\d+)")
_RUNS_RE = re.compile(r"记录\s*(\d+)\s*次运行")


def _cells(line: str) -> list[str]:
    """`| a | b |` -> ['a', 'b']。"""
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _f(s: str) -> float | None:
    """取单元格里的第一个数字；`+0.36`、`0.42`、`41.1%`、`4.71` 都吃。"""
    m = re.search(r"[-+]?\d+(?:\.\d+)?", s)
    return float(m.group()) if m else None


def _nums(line: str) -> list[float]:
    return [float(x) for x in re.findall(r"[-+]?\d+\.\d+", line)]


def _ci(cell: str) -> list[float] | None:
    """`41.1% [38.6, 43.7]` -> [41.1, 38.6, 43.7]。"""
    nums = re.findall(r"\d+(?:\.\d+)?", cell)
    return [float(n) for n in nums[:3]] if len(nums) >= 3 else None


def _section(lines: list[str], keyword: str) -> list[str]:
    """截取包含 keyword 的 ### 小节（到下一个 ### 或 ## 为止）。"""
    start = None
    for i, ln in enumerate(lines):
        if ln.startswith("###") and keyword in ln:
            start = i + 1
            break
    if start is None:
        return []
    out = []
    for ln in lines[start:]:
        if ln.startswith("##"):
            break
        out.append(ln)
    return out


def parse_two_layer_md(text: str) -> dict[str, Any]:
    """解析 md 文本；缺块时返回 `{"error": ...}`，绝不猜数字。"""
    lines = text.splitlines()

    # ---- 2×2 因子表 ----
    factor_rows: dict[str, list[float]] = {}
    for ln in _section(lines, "因子分解"):
        if not ln.lstrip().startswith("|"):
            continue
        cs = _cells(ln)
        if len(cs) < 3:
            continue
        head = cs[0]
        layer = (_LAYER_SINGLE if head.startswith(_LAYER_SINGLE)
                 else _LAYER_TWO if head.startswith(_LAYER_TWO) else "")
        if not layer:
            continue
        vals = [_f(cs[1]), _f(cs[2])]
        if None not in vals:
            factor_rows[layer] = [v for v in vals if v is not None]  # type: ignore[misc]
    missing = [k for k in (_LAYER_SINGLE, _LAYER_TWO) if k not in factor_rows]
    if missing:
        return {"error": f"analysis_two_layer.md 里没解析到因子表的行：{'、'.join(missing)}"}

    single_plain, single_cue = factor_rows[_LAYER_SINGLE]
    two_plain, two_cue = factor_rows[_LAYER_TWO]

    # ---- 效应量（直接抄 md 的结论行，不从四格反推：四格是四舍五入后的数，
    #      反推会和 md 差 0.01，那就又造出第二套口径了）----
    def effect_line(keyword: str, n: int) -> list[float] | None:
        for ln in lines:
            if keyword in ln and "**" in ln:
                nums = _nums(ln)
                if len(nums) >= n:
                    return nums[:n]
        return None

    split = effect_line("拆层效应", 2)
    cue = effect_line("给表效应", 2)
    inter = effect_line("交互", 1)
    if split is None or cue is None or inter is None:
        return {"error": "analysis_two_layer.md 里没解析到「拆层效应 / 给表效应 / 交互」三行"}

    # ---- 明细表（四行同口径）----
    arms: list[dict[str, Any]] = []
    for ln in _section(lines, "明细"):
        if not ln.lstrip().startswith("|"):
            continue
        cs = _cells(ln)
        if len(cs) < 6 or "·" not in cs[0]:
            continue
        layer, arm = (p.strip() for p in cs[0].split("·", 1))
        arms.append({
            "layer": layer,
            "arm": arm,
            "label": cs[0],
            "scene_median": _f(cs[1]),
            "ci": _ci(cs[2]),
            "line_pass": _f(cs[3]),
            "wrong_fields": _f(cs[4]),
            "raw_pass1": _f(cs[5]),
        })
    if len(arms) != 4:
        return {"error": f"analysis_two_layer.md 的明细表应有 4 行，解析到 {len(arms)} 行"}

    # ---- 自检：明细表的命中率必须等于因子表对应格 ----
    by_arm = {a["arm"]: a for a in arms}
    warnings: list[str] = []
    expect = {"card_injected": single_plain, "card_formula": single_cue,
              "judge_plain": two_plain, "judge_cue": two_cue}
    for arm, want in expect.items():
        got = (by_arm.get(arm) or {}).get("scene_median")
        if got is None or abs(got - want) > 0.005:
            warnings.append(f"明细表 {arm} 的命中率 {got} 与因子表 {want} 不一致")

    header = next((ln for ln in lines if "次运行" in ln), "")
    k_m = _K_RE.search(header)
    runs_m = _RUNS_RE.search(header)

    return {
        "cells": {
            "single_plain": single_plain, "single_cue": single_cue,
            "two_plain": two_plain, "two_cue": two_cue,
        },
        "effects": {
            "split": split, "cue": cue, "interaction": inter[0],
        },
        "arms": arms,
        "k": int(k_m.group(1)) if k_m else None,
        "n_runs": int(runs_m.group(1)) if runs_m else None,
        "warnings": warnings,
    }


def read_two_layer_factors(path: str) -> dict[str, Any]:
    """读 md 文件并解析；文件不存在时返回 error（不抛异常，交给调用方决定怎么显示）。"""
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        return {"error": f"{os.path.basename(path)} 不存在，先跑 python run_two_layer.py"}
    data = parse_two_layer_md(text)
    if "error" not in data:
        data["source"] = path
        data["source_mtime"] = int(os.path.getmtime(path))
    return data


def read_real_summary(path: str) -> dict[str, Any]:
    """读 `outputs/real_eval_summary.json`（`python analyze_real.py` 的产物）。

    页面的 MAE / CI / ρ 一律从这里取，不再从明细行现算——现算与产物会各有一个 ρ
    （0.538 vs 0.54，因为浮点与四舍五入路径不同），这种 1% 级别的漂移最难解释。
    """
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
    except FileNotFoundError:
        return {"error": "outputs/real_eval_summary.json 不存在，先跑 python analyze_real.py"}

    by_arm = {a["arm"]: a for a in raw.get("arms", [])}
    model = by_arm.get("judge_plain")
    rule = by_arm.get("规则表") or by_arm.get("rule")
    if not model or not rule:
        return {"error": "real_eval_summary.json 缺少 judge_plain / 规则表 两行"}
    return {
        "n": raw.get("n"),
        "boot": raw.get("boot"), "seed": raw.get("seed"),
        "summary": {
            "mae_model": model["mae"], "ci_model": model["ci"], "rho_model": model["rho"],
            "mae_rule": rule["mae"], "ci_rule": rule["ci"], "rho_rule": rule["rho"],
        },
        "by_work": raw.get("by_work", []),
        "source": path,
        "source_mtime": int(os.path.getmtime(path)),
    }
