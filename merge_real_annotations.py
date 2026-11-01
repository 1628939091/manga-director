"""把 real_form_v2.csv 里已标注的行合并进 annotations/real_form.csv。

目的：eval_real.py 只认 real_form.csv，不改动它就能吃更大的真实样本。

合并规则：
- real_form.csv 为基底（含全部旧列值）；
- v2 中 arousal(1-10) 非空的行参与合并；line_id 已存在 → v2 优先覆盖，
  不存在 → 整行追加（其余列原样带过来）；
- 表头、列序、utf-8-sig 编码保持不变。

安全：首行运行前自动备份 real_form.csv → real_form_backup_<日期>.csv
（已存在备份则不再覆盖，保留最早的那份）。
幂等：重复运行结果一致，不会产生重复行。

用法：python merge_real_annotations.py
"""

from __future__ import annotations

import csv
import datetime
import os

BASE = os.path.join("annotations", "real_form.csv")
V2 = os.path.join("annotations", "real_form_v2.csv")
BACKUP = os.path.join("annotations", "real_form_backup_%s.csv")


def load(path: str) -> tuple[list[str], list[dict]]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
        return list(rows[0].keys()), rows


def main() -> int:
    for p in (BASE, V2):
        if not os.path.exists(p):
            print(f"[缺少文件] {p}")
            return 1

    fields, base = load(BASE)
    fields2, v2 = load(V2)
    if fields != fields2:
        print(f"[表头不一致] base={fields}\n             v2  ={fields2}")
        return 1

    if not os.path.exists(BACKUP % datetime.date.today().isoformat()):
        bak = BACKUP % datetime.date.today().isoformat()
        raw = open(BASE, "rb").read()
        open(bak, "wb").write(raw)
        print(f"已备份 → {bak}")

    filled = [r for r in v2 if r["arousal(1-10)"].strip()]
    by_id = {r["line_id"]: r for r in base}

    appended, overwritten = 0, 0
    for r in filled:
        if r["line_id"] in by_id:                       # v2 优先覆盖
            base[base.index(by_id[r["line_id"]])] = r
            overwritten += 1
        else:
            base.append(r)
            by_id[r["line_id"]] = r
            appended += 1

    with open(BASE, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(base)

    print(f"v2 已填 {len(filled)}/{len(v2)} 行 → 追加 {appended}，覆盖 {overwritten}")
    print(f"real_form.csv 现为 {len(base)} 行（表头除外）")
    n_movie = len({r["出处"] for r in base})
    print(f"覆盖影片 {n_movie} 部:", sorted({r["出处"] for r in base}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
