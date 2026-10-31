"""重分析已有的 records.jsonl —— 零 API 调用、零成本。

用法::

    python reanalyze.py                       # 读 outputs/records.jsonl
    python reanalyze.py outputs/records.jsonl deepseek-ai/DeepSeek-V3.2 3

为什么要有这个脚本
------------------
上一轮的 pass@1 全为 0，不是因为模型差，是因为判据（τ=0.8、每行都过、每场都过）
在真实模型上不可达。测量的东西没有分辨率，任何改动都会被压成 0。
这个脚本不重新调用模型，只把同一批记录换成有分辨率的连续口径重算一遍，
再补三张归因表：反事实、AU 级联、style 错因分解。
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mdh import analysis  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))


def main(argv: list[str]) -> int:
    src = argv[1] if len(argv) > 1 else os.path.join(ROOT, "outputs", "records.jsonl")
    model = argv[2] if len(argv) > 2 else "deepseek-ai/DeepSeek-V3.2"
    k = int(argv[3]) if len(argv) > 3 else 3

    if not os.path.exists(src):
        print(f"[错误] 找不到记录文件：{src}")
        print("  → 先跑 python run_demo.py api 3 <model> 产出 records.jsonl")
        return 2

    records = analysis.load(src)
    md = analysis.render_real(records, {"model": model, "k": k, "tau": 0.8})
    out = os.path.join(os.path.dirname(src), "analysis_real_v32.md")
    analysis.save(md, out)

    print(f"记录 {len(records)} 次运行，臂 {analysis.scaffolds_of(records)}")
    print(f"报告：{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
