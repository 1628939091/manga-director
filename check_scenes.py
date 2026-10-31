"""场景体检：人物、关系、场景类型是否自洽。

为什么需要这个脚本
----------------
2026-10-31 用户标注时发现：有些行的「谁对谁说」根本不成立
（后辈称师父为上级、师徒互称恋人、非对手角色互称对手）。

根因在 tasks.py：`relation` 是 `rng.choice(...)` 独立抽的，和角色身份无关，
只有一条不完整的补丁（师父+恋人 → 改成长辈）。

这是同一类 bug 的第三次（前两次：随机潜台词污染金标、neutral 写成动作）。
前两次都靠人肉读出来，所以这一次把它写成会报错的检查。

用法：
    python check_scenes.py            # 体检当前场景集
    python check_scenes.py --strict   # 有不自洽就返回 1（给 CI / 重建后用）
"""

from __future__ import annotations

import collections
import sys

sys.path.insert(0, ".")

from mdh import tasks  # noqa: E402

# 角色身份的辈分：师父 > 男主/女主/对手 > 后辈
SENIORITY = {"师父": 3, "男主": 1, "女主": 1, "对手": 1, "后辈": 0}

# 「长辈 / 上级」只允许从高辈分指向低辈分
HIER_RELATIONS = ("长辈", "上级")
INTIMATE_RELATIONS = ("恋人", "恋人未满")

# 这些角色之间出现「恋人」关系属于设定冲突（师徒恋）
FORBIDDEN_INTIMATE = ({"师父", "后辈"}, {"师父", "男主"}, {"师父", "女主"})


def _seniority(role: str) -> int:
    return SENIORITY.get(role, 1)


def check_line(line: dict, cast: dict[str, dict]) -> list[str]:
    """返回这一行的不自洽原因；空列表 = 通过。"""
    why: list[str] = []
    spk = line.get("character")
    lis = line.get("listener")
    rel = line.get("relation")
    sr = cast.get(spk, {}).get("role", "?")
    lr = cast.get(lis, {}).get("role", "?")

    if spk not in cast:
        why.append(f"说话人 {spk} 不在角色表里")
    if lis not in cast:
        why.append(f"听者 {lis} 不在角色表里")
    if spk == lis:
        why.append("自问自答")

    if rel in HIER_RELATIONS and _seniority(sr) <= _seniority(lr):
        why.append(f"{sr} 称 {lr} 为「{rel}」，辈分方向反了")

    if rel in INTIMATE_RELATIONS:
        pair = frozenset((sr, lr))
        if pair in FORBIDDEN_INTIMATE:
            why.append(f"{sr} 与 {lr} 互称「{rel}」，与设定冲突")

    if rel == "对手" and "对手" not in (sr, lr):
        why.append(f"{sr} 与 {lr} 互称「对手」，但双方都不是对手角色")

    return why


def main(argv: list[str]) -> int:
    strict = "--strict" in argv
    n_gen = 50
    for a in argv:
        if a.startswith("--n="):
            n_gen = int(a.split("=", 1)[1])

    scenes = tasks.build_tasks(n_generated=n_gen)
    total = 0
    bad: list[tuple[str, str, str, str, str]] = []

    for t in scenes:
        for ln in t["lines"]:
            total += 1
            for why in check_line(ln, t["characters"]):
                bad.append((t["scene_id"], ln.get("scene_type", "?"),
                            ln["character"], ln.get("listener", "?"), why))

    by_type = collections.Counter(w.split("，")[0] for _, _, _, _, w in bad)
    affected = len({sid for sid, *_ in bad})

    print(f"场景 {len(scenes)} 个 / 台词 {total} 行")
    print(f"不自洽 {len(bad)} 处（{len(bad) / max(total, 1) * 100:.0f}%），"
          f"涉及 {affected} 个场景")

    if by_type:
        print("\n按原因:")
        for k, v in by_type.most_common():
            print(f"  {k} ... {v} 处")

    if bad:
        print("\n前 12 条:")
        for sid, st, spk, lis, why in bad[:12]:
            print(f"  [{st}] {sid}  {spk} → {lis}  ← {why}")

    ok = not bad
    print("\n结论:", "通过" if ok else "不自洽——重建场景时必须先修 relation 的生成方式")
    if strict and not ok:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
