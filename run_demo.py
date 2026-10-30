"""一键入口：跑「任务 × 脚手架 × k 次重复」，出报告。

用法::

    python run_demo.py                 # Mock 校准（零成本、离线、可复现）
    python run_demo.py api 5           # 真实模型，k=5
    python run_demo.py api 5 deepseek-chat

环境变量（接真实模型时）::

    DEEPSEEK_API_KEY=sk-xxx            # 或 MDH_API_KEY
    MDH_BASE_URL=https://api.deepseek.com/v1
    MDH_TEMPERATURE=0.7
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mdh import judge, llm, report, runner, tasks  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
SCAFFOLDS = ("naive_shot", "card_injected", "card_fewshot", "card_repair", "card_formula")

BANNER = """
==============================================================
 漫剧导演 Agent · 可靠性评测（MangaDirector Harness）
 判卷分三层：规则（不依赖金标准）→ 参考实现容差 → (后续) VLM
==============================================================
"""


def main(argv: list[str]) -> int:
    mode = argv[1] if len(argv) > 1 else "mock"
    k = int(argv[2]) if len(argv) > 2 else 5
    model = argv[3] if len(argv) > 3 else "deepseek-chat"

    print(BANNER)

    all_tasks = tasks.build_tasks(n_generated=12)
    # Mock 演示用全量评估（4 场的 holdout 太小，1 场 = 25 个百分点，读不出东西）。
    # 示例检索仍然按 work 隔离：fewshot_examples 会跳过同 work 的场景。
    split = tasks.split_by_work(all_tasks, holdout_ratio=0.25)
    eval_tasks = all_tasks
    pool = all_tasks

    n_lines = sum(len(t["lines"]) for t in eval_tasks)
    print(f"场景 {len(eval_tasks)} 场 / 台词 {n_lines} 条 | "
          f"k={k} | 脚手架 {len(SCAFFOLDS)} 个 | 模式 {mode}")
    print(f"作品 {len({t['work_id'] for t in all_tasks})} 部（示例检索按 work 隔离，不跨作品泄漏）\n")

    if mode == "api":
        temperature = float(os.environ.get("MDH_TEMPERATURE", "0.7"))
        try:
            backend = llm.OpenAICompat(model=model, temperature=temperature, seed=20261030)
        except RuntimeError as e:
            print(f"[配置错误] {e}")
            print("  → 先跑 python run_demo.py 用 Mock 验证链路；接真模型时再设 key。")
            return 2
        print(f"后端：{backend.base_url} / {backend.model} / T={temperature}")
    else:
        backend = llm.MockDirector(name="mock-director")
        print("后端：MockDirector（注入四类已知缺陷，用于校准判卷链路）")
    print("")

    records = runner.run_matrix(
        eval_tasks, list(SCAFFOLDS), backend, k=k,
        tau=judge.DEFAULT_TAU,
        examples_for=runner.make_example_provider(pool, n=2),
    )

    os.makedirs(OUT_DIR, exist_ok=True)
    records_path = os.path.join(OUT_DIR, "records.jsonl")
    runner.save_records(records, records_path)

    md = report.render(records, {
        "scaffolds": list(SCAFFOLDS),
        "k": k,
        "tau": judge.DEFAULT_TAU,
        "n_tasks": len(eval_tasks),
        "n_lines": n_lines,
        "mode": mode,
        "model": getattr(backend, "model", getattr(backend, "name", "-")),
        "split": f"评估 {len(eval_tasks)} 场；金标准示例仅取自其他作品（按 work_id 隔离）",
    })
    report_path = os.path.join(OUT_DIR, "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md)

    print("\n--- 摘要 ---")
    for s in SCAFFOLDS:
        from mdh import metrics
        print(f"  {s:14s} pass@1={metrics.pass_at_1(records, s) * 100:5.1f}%  "
              f"pass^{k}={metrics.pass_pow_k(records, s) * 100:5.1f}%  "
              f"违反率={metrics.violation_rate(records, s) * 100:5.1f}%")

    print(f"\n报告：{report_path}")
    print(f"逐次记录：{records_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
