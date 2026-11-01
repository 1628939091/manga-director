"""导出层单测：空输入 / 缺字段 / 越界值 / 端到端 CLI。

导出层的容错语义与判卷相反（judge 判死越界，export 截断越界），
所以这里的断言是「截断后落在区间内」，不是「报错」。
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mdh import export  # noqa: E402
from mdh.schema import RANGES  # noqa: E402

FULL_SHOT = {
    "line_id": "W01-L01",
    "character": "悟空",
    "text": "哗－－！闭嘴！",
    "intensity": 8.0,
    "suppression": 0.48,
    "f0_offset": 1.7,
    "f0_range": 4.9,
    "rate": 6.3,
    "pause_after_ms": 260,
    "expression": {"style": "外放",
                   "au": {"AU4": 0.55, "AU20": 0.70, "AU26": 0.55,
                          "AU12": 0.15, "AU17": 0.20},
                   "gaze": "直视对方"},
    "constraints": [],
    "evidence": "哗－－！闭嘴！",
}


class TestExportEmpty(unittest.TestCase):
    # ---------------- 空输入 ----------------
    def test_ssml_empty_is_valid_document(self):
        ssml = export.to_ssml([])
        self.assertIn("<speak", ssml)
        self.assertIn("</speak>", ssml)
        self.assertNotIn("prosody", ssml)

    def test_tts_empty_is_empty_list(self):
        self.assertEqual(export.to_tts_params([]), [])

    def test_prompt_empty_is_empty_list(self):
        self.assertEqual(export.to_video_prompt([]), [])

    def test_non_dict_items_are_skipped(self):
        ssml = export.to_ssml([None, "垃圾", FULL_SHOT])
        self.assertEqual(ssml.count("<prosody"), 1)
        params = export.to_tts_params([None, 42, FULL_SHOT])
        self.assertEqual(params[0]["line_id"], "W01-L01")


class TestExportMissing(unittest.TestCase):
    # ---------------- 缺字段 ----------------
    def test_missing_fields_fall_back_to_neutral(self):
        bare = {"line_id": "x", "character": "张三", "evidence": "嗯。"}
        p = export.to_tts_params([bare])[0]
        self.assertEqual(p["speed"], 1.0)          # rate 缺 -> 4.5 字/秒 -> 1.0
        self.assertEqual(p["pitch_semitones"], 0.0)
        self.assertEqual(p["volume_db"], 0.0)      # intensity 缺 -> 4 -> 0 dB
        self.assertEqual(p["pause_after_ms"], 0)
        self.assertEqual(p["text"], "嗯。")        # text 缺 -> evidence 兜底

    def test_missing_text_and_evidence_gives_empty_text(self):
        p = export.to_tts_params([{"line_id": "x", "character": "李四"}])[0]
        self.assertEqual(p["text"], "")


class TestExportOutOfRange(unittest.TestCase):
    # ---------------- 越界值 ----------------
    def test_out_of_range_values_are_clamped(self):
        crazy = {**FULL_SHOT, "intensity": 99, "rate": -5, "f0_offset": 50,
                 "pause_after_ms": 99999, "f0_range": 0.0}
        p = export.to_tts_params([crazy])[0]
        # rate -5 -> 区间下界 2.0 -> speed 0.44 -> 再被 speed 下限夹到 0.5
        self.assertEqual(p["speed"], 0.5)
        self.assertGreaterEqual(p["intensity"], RANGES["intensity"][0])
        self.assertLessEqual(p["intensity"], RANGES["intensity"][1])
        lo, hi = RANGES["f0_offset"]
        self.assertGreaterEqual(p["pitch_semitones"], lo)
        self.assertLessEqual(p["pitch_semitones"], hi)
        self.assertEqual(p["pause_after_ms"], 2000)
        self.assertEqual(p["f0_range_semitones"], 0.5)   # 0.0 -> 截回下界

    def test_ssml_volume_is_bounded(self):
        ssml = export.to_ssml([{**FULL_SHOT, "intensity": 10.0}])
        self.assertIn("+9.0dB", ssml)              # (10-4)*1.5=9，未超上限
        ssml2 = export.to_ssml([{**FULL_SHOT, "intensity": 99}])
        # intensity 先被契约区间截到 10.0，音量封顶 (10-4)*1.5=+9.0dB
        self.assertIn("+9.0dB", ssml2)
        self.assertNotIn("99", ssml2.split("prosody")[1][:60])


class TestExportContent(unittest.TestCase):
    # ---------------- 内容正确性 ----------------
    def test_ssml_carries_text_and_break(self):
        ssml = export.to_ssml([FULL_SHOT])
        self.assertIn("哗－－！闭嘴！", ssml)
        self.assertIn('<break time="260ms"/>', ssml)
        self.assertIn('pitch="+1.7st"', ssml)
        self.assertIn('rate="140%"', ssml)         # 6.3 / 4.5 = 1.4

    def test_ssml_escapes_xml_specials(self):
        ssml = export.to_ssml([{"character": "A<B>", "evidence": "1<2 & 3"}])
        self.assertIn("1&lt;2 &amp; 3", ssml)
        self.assertIn("A&lt;B&gt;", ssml)

    def test_prompt_contains_character_expression_gaze(self):
        (p,) = export.to_video_prompt([FULL_SHOT])
        self.assertIn("悟空", p)
        self.assertIn("外放", p)
        self.assertIn("直视对方", p)
        self.assertIn("眉头紧锁", p)               # AU 按数值取前两（AU20 在前）
        self.assertIn("嘴角绷紧", p)
        self.assertIn("说完停顿 0.3 秒", p)


class TestWorkbenchCLI(unittest.TestCase):
    # ---------------- CLI 端到端 ----------------
    def test_workbench_cli_end_to_end(self):
        d = tempfile.mkdtemp()
        script = os.path.join(d, "script.txt")
        with open(script, "w", encoding="utf-8") as f:
            f.write("【第二场 夜 城墙】\n\n悟空：哗－－！闭嘴！\n"
                    "这一行没有冒号格式\n紫霞：你又明不明白我已经不再是神仙了！\n")
        out = os.path.join(d, "sb.json")
        import workbench
        rc = workbench.main([script, "--out", out, "--ssml", "--tts", "--prompt"])
        self.assertEqual(rc, 0)
        shots = json.load(open(out, encoding="utf-8"))
        self.assertEqual(len(shots), 2)
        self.assertEqual(shots[0]["character"], "悟空")
        stem = os.path.splitext(out)[0]
        self.assertIn("哗－－！闭嘴！",
                      open(stem + "_ssml.xml", encoding="utf-8").read())
        tts = json.load(open(stem + "_tts.json", encoding="utf-8"))
        self.assertEqual(len(tts), 2)
        self.assertEqual(tts[1]["character"], "紫霞")
        prompts = open(stem + "_prompts.txt", encoding="utf-8").read().splitlines()
        self.assertEqual(len(prompts), 2)
        self.assertIn("紫霞", prompts[1])

    def test_workbench_rejects_unparseable_script(self):
        import workbench
        d = tempfile.mkdtemp()
        script = os.path.join(d, "bad.txt")
        with open(script, "w", encoding="utf-8") as f:
            f.write("没有任何冒号格式的行\n")
        rc = workbench.main([script, "--out", os.path.join(d, "o.json")])
        self.assertEqual(rc, 1)


if __name__ == "__main__":
    unittest.main()
