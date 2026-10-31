"""接新模型的第一道关：2 场 × 2 个 scaffold × 1 次，验证 JSON 能解析、判卷能跑。

为什么要单独一个冒烟脚本：全量动辄几百次调用，如果模型返回的不是合法 JSON，
或者 schema 对不上，跑到一半才发现，钱和时间都白花。这个脚本 4 次调用、十几秒，
专门用来回答两个问题——模型听话吗？判卷器接得住吗？

用法::

    python smoke_api.py                                  # 默认 DeepSeek-V3.2
    python smoke_api.py deepseek-ai/DeepSeek-V4-Flash    # 换模型
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mdh import agents, judge, llm, tasks  # noqa: E402
from mdh import reference as ref  # noqa: E402

DEFAULT_MODEL = "deepseek-ai/DeepSeek-V3.2"


def main(argv: list[str]) -> int:
    model = argv[1] if len(argv) > 1 else DEFAULT_MODEL
    temperature = float(os.environ.get("MDH_TEMPERATURE", "0.3"))

    try:
        backend = llm.OpenAICompat(model=model, temperature=temperature, max_tokens=2048)
    except RuntimeError as e:
        print(f"[配置错误] {e}")
        return 2

    print(f"后端 {backend.base_url}")
    print(f"模型 {backend.model} / T={temperature} / max_tokens={backend.max_tokens}\n")

    all_tasks = tasks.build_tasks(n_generated=12)
    subset = all_tasks[:2]
    total_in = total_out = 0
    ok = fail = 0
    t0 = time.time()

    for task in subset:
        golds = {ln["id"]: ref.gold_shot(ln, task["characters"]) for ln in task["lines"]}
        for scaffold in ("naive_shot", "card_formula"):
            try:
                out = agents.run_scaffold(task, scaffold, backend, max_repair=0, repeat=0)
            except RuntimeError as e:
                print(f"[{task['scene_id']}] {scaffold}  调用失败：{e}")
                fail += 1
                continue

            total_in += out["tokens_in"]
            total_out += out["tokens_out"]
            parsed = out["output"] is not None

            if not parsed:
                fail += 1
                print(f"[{task['scene_id']}] {scaffold}  ✗ JSON 解析失败")
                raw = out["trace"][0]["raw_head"] if out["trace"] else "(无输出)"
                print("    原始输出前 180 字：" + raw.replace("\n", " ⏎ "))
                continue

            j = judge.judge_scene(out["output"], task, golds)
            ok += 1
            print(f"[{task['scene_id']}] {scaffold}  pass={j['passed']}  "
                  f"命中率={j['hit_rate']:.2f}  行数={len(j['rows'])}/{len(task['lines'])}")
            if j["errors"]:
                print("    L1 错误：" + " | ".join(str(e) for e in j["errors"][:3]))
            if j["violations"]:
                count = sum(len(v) for v in j["violations"].values()) if isinstance(
                    j["violations"], dict) else len(j["violations"])
                print(f"    禁令违反 {count} 处")

    elapsed = time.time() - t0
    print(f"\n{'=' * 56}")
    print(f"可解析 {ok} 条 / 失败 {fail} 条 | 耗时 {elapsed:.1f}s")
    print(f"token：输入 {total_in}，输出 {total_out}，合计 {total_in + total_out}")
    if ok and fail == 0:
        print("→ 链路正常，可以跑全量：python run_demo.py api 3 " + model)
    elif ok:
        print("→ 部分失败，先看上面的原始输出，可能是模型不听话或 schema 太严")
    else:
        print("→ 全部失败，先别跑全量")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
