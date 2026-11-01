"""report_io 的解析与「三处一致」守卫。

两类断言：
1. 解析器本身：缺块要报错、不猜数字、四格与明细表不一致要报警告。
2. 仓库里真实的三份产物（README / analysis md / summary json）必须逐字一致——
   这正是「页面不要再现算」的目的：数字只有一个生成点，其余都是读者。
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from mdh import report_io  # noqa: E402

MD = os.path.join(ROOT, "outputs", "analysis_two_layer.md")
REAL_MD = os.path.join(ROOT, "outputs", "analysis_real_scripts.md")
REAL_JSON = os.path.join(ROOT, "outputs", "real_eval_summary.json")
README = os.path.join(ROOT, "README.md")

def _read(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


# 最小可用样本：结构与真实产物同构，但数字可控
SAMPLE = """# 两层架构实验

> 模型：`-`　k = 3　臂：`judge_plain` / `judge_cue`　记录 306 次运行

### 因子分解：拆层 × 给不给线索表（场景平均命中率中位数）

| | 不给线索表 | 给线索表 |
|---|---|---|
| 单层（模型直接出 8 参数） | 0.42 | 0.53 |
| 两层（模型判读 + 代码标定） | 0.78 | 0.96 |

- **拆层效应**：+0.36（不给表）/ +0.43（给表）
- **给表效应**：+0.11（单层）/ +0.18（两层）
- **交互**：+0.06（= 给表效应在两层里减去在单层里）

### 明细（四行同口径）

| 配置 | 场景平均命中率 | 聚类 95% CI | 行级 pass@τ | 每行错字段 | 原始 pass@1 |
|---|---|---|---|---|---|
| 单层 · card_injected | 0.42 | 41.1% [38.6, 43.7] | 0.8% | 4.71 | 0.0% |
| 单层 · card_formula | 0.53 | 54.9% [52.0, 58.1] | 21.9% | 3.56 | 1.3% |
| 两层 · judge_plain | 0.78 | 74.4% [70.9, 77.8] | 56.6% | 2.00 | 13.1% |
| 两层 · judge_cue | 0.96 | 90.8% [88.0, 93.4] | 83.9% | 0.71 | 60.1% |
"""


class ParseTwoLayerTest(unittest.TestCase):
    def test_parses_sample(self):
        d = report_io.parse_two_layer_md(SAMPLE)
        self.assertEqual(d["cells"], {"single_plain": 0.42, "single_cue": 0.53,
                                      "two_plain": 0.78, "two_cue": 0.96})
        self.assertEqual(d["effects"]["split"], [0.36, 0.43])
        self.assertEqual(d["effects"]["cue"], [0.11, 0.18])
        self.assertEqual(d["effects"]["interaction"], 0.06)
        self.assertEqual(d["k"], 3)
        self.assertEqual(d["n_runs"], 306)
        self.assertEqual(len(d["arms"]), 4)
        self.assertEqual(d["arms"][0]["ci"], [41.1, 38.6, 43.7])
        self.assertEqual(d["warnings"], [])

    def test_missing_factor_table_is_error(self):
        d = report_io.parse_two_layer_md("# 只有标题\n\n没有表格。\n")
        self.assertIn("error", d)
        self.assertIn("因子表", d["error"])

    def test_missing_detail_rows_is_error(self):
        broken = SAMPLE.split("### 明细")[0]
        d = report_io.parse_two_layer_md(broken)
        self.assertIn("error", d)
        self.assertIn("明细表", d["error"])

    def test_inconsistent_cell_warns(self):
        bad = SAMPLE.replace("| 两层 · judge_cue | 0.96 |", "| 两层 · judge_cue | 0.90 |")
        d = report_io.parse_two_layer_md(bad)
        self.assertTrue(any("judge_cue" in w for w in d["warnings"]))

    def test_missing_file_is_error(self):
        d = report_io.read_two_layer_factors(os.path.join(ROOT, "outputs", "nope.md"))
        self.assertIn("error", d)


class ReadRealSummaryTest(unittest.TestCase):
    def test_parses_arms(self):
        payload = {"n": 140, "boot": 2000, "seed": 7,
                   "arms": [{"arm": "judge_plain", "mae": 1.25, "ci": [1.08, 1.43], "rho": 0.54},
                            {"arm": "规则表", "mae": 1.14, "ci": [0.99, 1.28], "rho": 0.09}]}
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False,
                                         encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
            path = f.name
        try:
            d = report_io.read_real_summary(path)
            self.assertEqual(d["summary"]["mae_model"], 1.25)
            self.assertEqual(d["summary"]["rho_rule"], 0.09)
            self.assertEqual(d["n"], 140)
        finally:
            os.unlink(path)

    def test_missing_arm_is_error(self):
        payload = {"n": 1, "arms": [{"arm": "judge_plain", "mae": 1.0, "ci": [1, 1], "rho": 0.5}]}
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False,
                                         encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
            path = f.name
        try:
            self.assertIn("error", report_io.read_real_summary(path))
        finally:
            os.unlink(path)


class ArtifactConsistencyTest(unittest.TestCase):
    """仓库里的三份产物必须同数——页面、md、README 打架是这套评测最贵的 bug。"""

    def test_readme_2x2_matches_analysis_md(self):
        art = report_io.read_two_layer_factors(MD)
        self.assertNotIn("error", art)
        text = _read(README)
        single = re.search(r"\| 单层：[^|]*\| *([\d.]+) *\| *([\d.]+) *\|", text)
        two = re.search(r"\| 两层：[^|]*\| *\**([\d.]+)\** *\| *\**([\d.]+)\** *\|", text)
        self.assertIsNotNone(single, "README 顶部 2×2 没找到「单层」行")
        self.assertIsNotNone(two, "README 顶部 2×2 没找到「两层」行")
        self.assertEqual(float(single.group(1)), art["cells"]["single_plain"])
        self.assertEqual(float(single.group(2)), art["cells"]["single_cue"])
        self.assertEqual(float(two.group(1)), art["cells"]["two_plain"])
        self.assertEqual(float(two.group(2)), art["cells"]["two_cue"])

    def test_readme_real_metrics_match_summary(self):
        summ = report_io.read_real_summary(REAL_JSON)
        self.assertNotIn("error", summ)
        text = _read(README)
        row = re.search(r"\| `judge_plain` \| ([\d.]+) \| \[([\d.]+), ([\d.]+)\] \| \**\+([\d.]+)\** \|", text)
        self.assertIsNotNone(row, "README 的三方对比表没找到 judge_plain 行")
        s = summ["summary"]
        self.assertEqual(float(row.group(1)), s["mae_model"])
        self.assertEqual(float(row.group(2)), s["ci_model"][0])
        self.assertEqual(float(row.group(3)), s["ci_model"][1])
        self.assertEqual(float(row.group(4)), s["rho_model"])

    def test_real_md_matches_summary(self):
        summ = report_io.read_real_summary(REAL_JSON)
        text = _read(REAL_MD)
        row = re.search(r"\| `judge_plain` \| ([\d.]+) \| \[([\d.]+), ([\d.]+)\] \| \**\+([\d.]+)\** \|", text)
        self.assertIsNotNone(row, "analysis_real_scripts.md 没找到 judge_plain 行")
        s = summ["summary"]
        self.assertEqual(float(row.group(1)), s["mae_model"])
        self.assertEqual(float(row.group(4)), s["rho_model"])


if __name__ == "__main__":
    unittest.main()
