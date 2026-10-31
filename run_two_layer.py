"""两层架构实验：只跑判读层，标定由代码完成。

用法::

    python run_two_layer.py mock 1              # 离线，零成本，验证链路
    python run_two_layer.py api 3               # 真实模型，两个臂 × 13 场 × k=3
    python run_two_layer.py api 3 deepseek-ai/DeepSeek-V3.2

与 run_demo.py 的分工
--------------------
run_demo.py 跑的是**单层**五臂（模型一次输出 8 个参数），产出
outputs/records.jsonl —— 那是"让模型当计算器"的基线。
本脚本跑的是**两层**两臂（模型只判读），产出
outputs/records_two_layer.jsonl，并在报告里把两者放同一张表比。

为什么两个臂
-----------
`judge_plain` 不给线索表：测「模型自己能不能读出台词的内心强度」。
`judge_cue`   给出线索表：测「给了规则之后能对齐到几分」。
两者的差就是"规则可教性"，而它们与单层 `card_formula` 的差
是"架构可替换性"——后者才是本实验真正想回答的问题。
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mdh import analysis, judge, llm, runner, tasks  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, "outputs")
ARMS = ("judge_plain", "judge_cue")


def main(argv: list[str]) -> int:
    mode = argv[1] if len(argv) > 1 else "mock"
    k = int(argv[2]) if len(argv) > 2 else 3
    model = argv[3] if len(argv) > 3 else "deepseek-ai/DeepSeek-V3.2"

    n_gen = int(os.environ.get("MDH_N_SCENES", "50"))
    all_tasks = tasks.build_tasks(n_generated=n_gen)
    # 按作品切分：同一部作品的角色卡与说话风格相同，按场景随机切会泄漏
    split = tasks.split_by_work(all_tasks, holdout_ratio=0.25)
    for t in split["train"]:
        t["split"] = "train"
    for t in split["test"]:
        t["split"] = "test"
    all_tasks = split["train"] + split["test"]
    n_lines = sum(len(t["lines"]) for t in all_tasks)

    print("=" * 62)
    print(" 两层架构实验：判读 → 模型，标定 → 代码")
    print("=" * 62)
    print(f"场景 {len(all_tasks)} 场 / 台词 {n_lines} 条 | k={k} | 臂 {len(ARMS)} 个 | 模式 {mode}")

    if mode == "api":
        temperature = float(os.environ.get("MDH_TEMPERATURE", "0.7"))
        try:
            backend = llm.OpenAICompat(model=model, temperature=temperature, seed=20261030)
        except RuntimeError as e:
            print(f"[配置错误] {e}")
            return 2
        print(f"后端：{backend.base_url} / {backend.model} / T={temperature}")
    else:
        backend = llm.MockDirector(name="mock-director")
        print("后端：MockDirector（注入 arousal 偏移 / 意图误判 / 归因改写）")
    print(f"预计调用：{len(all_tasks) * len(ARMS) * k} 次\n")

    records = runner.run_matrix(
        all_tasks, list(ARMS), backend, k=k,
        tau=judge.DEFAULT_TAU,
        examples_for=None,
        workers=int(os.environ.get("MDH_WORKERS", "1")),
    )

    os.makedirs(OUT_DIR, exist_ok=True)
    # Mock 与真实运行分文件存放：mock 只用来验证链路，绝不能覆盖已跑完的真实记录与报告
    suffix = "" if mode == "api" else "_mock"
    rec_path = os.path.join(OUT_DIR, f"records_two_layer{suffix}.jsonl")
    runner.save_records(records, rec_path)

    base_path = os.path.join(OUT_DIR, "records.jsonl")
    baseline = analysis.load(base_path) if os.path.exists(base_path) else None

    md = analysis.render_two_layer(records, {
        "scaffolds": list(ARMS), "k": k, "tau": judge.DEFAULT_TAU,
        "model": getattr(backend, "model", getattr(backend, "name", "-")),
    }, baseline=baseline)
    out = os.path.join(OUT_DIR, f"analysis_two_layer{suffix}.md")
    analysis.save(md, out)

    print("\n--- 摘要 ---")
    for s in ARMS:
        st = analysis.appraisal_stats(records, s)
        e2e = analysis.end_to_end_by_arousal(records, s)
        print(f"  {s:13s} arousal MAE={st['arousal_mae']:.2f}  "
              f"意图={_r(st['hits']['intent'])}  "
              f"端到端(判对)={e2e['arousal_ok'][0]:.3f}  (判错)={e2e['arousal_bad'][0]:.3f}")
    print(f"\n报告：{out}")
    print(f"记录：{rec_path}")
    return 0


def _r(pair) -> str:
    h, n = pair
    return f"{h / n * 100:.0f}%" if n else "—"


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
