"""核心逻辑单测（只用标准库 unittest，无需 pytest）。

覆盖的是"判卷会不会判错"这件事——比覆盖率高更要紧：
判卷错了，后面所有 pass^k 都是假的。
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mdh import agents, annotate, judge, metrics, reference as ref, runner, schema, tasks  # noqa: E402


def demo_task():
    ts = tasks.build_tasks(n_generated=0)
    return ts[0]


def gold_of(task):
    return {ln["id"]: ref.gold_shot(ln, task["characters"]) for ln in task["lines"]}


def clean(shot):
    return {k: v for k, v in shot.items() if not k.startswith("_")}


class TestSchema(unittest.TestCase):
    def test_evidence_must_be_substring(self):
        task = demo_task()
        ln = task["lines"][0]
        self.assertEqual(schema.check_evidence(ln["text"], ln["text"]), [])
        # 省略号/空格差异应被容忍（归一化后比较）
        self.assertEqual(schema.check_evidence("刚才说的那些，你听到了吗", ln["text"]), [])
        # 外层引号 / 句尾括注 = 模型的引用格式，不算幻觉
        self.assertEqual(schema.check_evidence(f"「{ln['text']}」", ln["text"]), [])
        self.assertEqual(schema.check_evidence(ln["text"] + "（压低）", ln["text"]), [])
        # 引文中间夹括号 = 可疑（可能藏真幻觉）
        self.assertTrue(schema.check_evidence(
            "你听到了吗（压低）那些", ln["text"]))
        # 改写 = 幻觉
        self.assertTrue(schema.check_evidence(ln["text"] + "其实心跳很快", ln["text"]))

    def test_range_and_enum(self):
        task = demo_task()
        shot = clean(ref.gold_shot(task["lines"][0], task["characters"]))
        self.assertEqual(schema.validate_shot(shot), [])
        bad = dict(shot, intensity=11.0)
        self.assertTrue(any("越界" in e for e in schema.validate_shot(bad)))
        bad2 = dict(shot, expression=dict(shot["expression"], style="狂笑"))
        self.assertTrue(any("枚举" in e for e in schema.validate_shot(bad2)))


class TestReference(unittest.TestCase):
    def test_internal_character_is_suppressed(self):
        """内敛角色：外部强度必须低于内心唤醒（抑制系数 < 1）。"""
        task = demo_task()
        line = next(ln for ln in task["lines"] if ln["character"] == "沈砚")
        g = ref.gold_shot(line, task["characters"])
        self.assertLess(g["intensity"], g["_internal"]["arousal"])
        self.assertLess(g["suppression"], 1.0)

    def test_no_shout_scene_declares_constraint(self):
        self.assertEqual(ref.required_constraints("亲密低语"), ["no_shout"])
        self.assertEqual(ref.required_constraints("冲突对峙"), [])


class TestJudge(unittest.TestCase):
    def setUp(self):
        self.task = demo_task()
        self.golds = gold_of(self.task)

    def test_gold_passes_itself(self):
        out = {"shots": [clean(self.golds[ln["id"]]) for ln in self.task["lines"]]}
        r = judge.judge_scene(out, self.task, self.golds)
        self.assertTrue(r["passed"], r["errors"])
        self.assertEqual(r["hit_rate"], 1.0)

    def test_shouting_in_no_shout_scene_is_violation(self):
        shots = [clean(self.golds[ln["id"]]) for ln in self.task["lines"]]
        target = next(ln for ln in self.task["lines"] if ln.get("scene_type") == "亲密低语")
        idx = [ln["id"] for ln in self.task["lines"]].index(target["id"])
        shots[idx]["intensity"] = 9.5
        shots[idx]["expression"]["style"] = "外放"
        r = judge.judge_scene({"shots": shots}, self.task, self.golds)
        self.assertFalse(r["passed"])
        self.assertTrue(any(v["constraint"] == "no_shout" for v in r["violations"]))

    def test_scene_pass_is_not_diluted_by_average(self):
        """一行全对 + 一行 2 个字段错 → 平均命中率 0.875 ≥ τ，但场景必须判失败。"""
        shots = [clean(self.golds[ln["id"]]) for ln in self.task["lines"]]
        ln2 = self.task["lines"][1]
        shots[1]["intensity"] = round(shots[1]["intensity"] + 2.0, 2)
        shots[1]["f0_range"] = round(shots[1]["f0_range"] + 2.0, 2)
        r = judge.judge_scene({"shots": shots}, self.task, self.golds)
        self.assertGreater(r["hit_rate"], judge.DEFAULT_TAU)   # 平均是过的
        self.assertFalse(r["passed"])                          # 但场景必须失败
        self.assertLess(r["min_hit_rate"], judge.DEFAULT_TAU)

    def test_missing_shot_fails(self):
        shots = [clean(self.golds[ln["id"]]) for ln in self.task["lines"]][:1]
        r = judge.judge_scene({"shots": shots}, self.task, self.golds)
        self.assertFalse(r["passed"])
        self.assertTrue(any("缺镜头" in e for e in r["errors"]))

    def test_non_json_output_fails_gracefully(self):
        r = judge.judge_scene({"oops": 1}, self.task, self.golds)
        self.assertFalse(r["passed"])
        self.assertIn("shots", r["errors"][0])


class TestAgents(unittest.TestCase):
    def test_extract_json_variants(self):
        self.assertEqual(agents.extract_json('{"a": 1}'), {"a": 1})
        self.assertEqual(agents.extract_json('```json\n{"a": 1}\n```'), {"a": 1})
        self.assertEqual(agents.extract_json('好的：\n{"a": 1}\n希望有帮助'), {"a": 1})
        # 字符串里的花括号不能被当成结构
        self.assertEqual(agents.extract_json('{"a": "}{"}'), {"a": "}{"})
        self.assertIsNone(agents.extract_json("完全没有 JSON"))

    def test_repair_feedback_is_l1_only(self):
        """纯 L2 偏差（数值偏但结构合法）不应产生任何可回传的反馈。

        否则修复轮就变成了对着答案改错，pass^k 会被系统性虚高。
        """
        task = demo_task()
        shots = [clean(ref.gold_shot(ln, task["characters"])) for ln in task["lines"]]
        shots[0]["rate"] = round(shots[0]["rate"] + 0.8, 2)   # 只偏离容差，不触犯任何 L1 规则
        self.assertEqual(agents.l1_feedback({"shots": shots}, task), [])

    def test_prompts_differ_by_scaffold(self):
        task = demo_task()
        naive = agents.build_prompt(task, "naive_shot")[1]["content"]
        card = agents.build_prompt(task, "card_injected")[1]["content"]
        formula = agents.build_prompt(task, "card_formula")[1]["content"]
        self.assertNotIn("角色卡", naive)
        self.assertIn("角色卡", card)
        self.assertNotIn("0.60+0.40", card)
        self.assertIn("0.60+0.40", formula)


class TestMetrics(unittest.TestCase):
    def test_pass_pow_k_requires_all_repeats(self):
        recs = [
            {"scaffold": "x", "scene_id": "a", "repeat": 0, "passed": True},
            {"scaffold": "x", "scene_id": "a", "repeat": 1, "passed": False},
            {"scaffold": "x", "scene_id": "b", "repeat": 0, "passed": True},
            {"scaffold": "x", "scene_id": "b", "repeat": 1, "passed": True},
        ]
        self.assertAlmostEqual(metrics.pass_at_1(recs, "x"), 0.75)
        self.assertAlmostEqual(metrics.pass_pow_k(recs, "x"), 0.5)

    def test_bootstrap_single_cluster_is_degenerate(self):
        lo, hi, pt = metrics.cluster_bootstrap_ci({"a": [1.0, 1.0]}, B=200)
        self.assertEqual((lo, hi, pt), (1.0, 1.0, 1.0))


class TestTasks(unittest.TestCase):
    def test_fewshot_excludes_same_work(self):
        all_tasks = tasks.build_tasks(n_generated=4)
        t = all_tasks[1]
        ex = runner.fewshot_examples(t, all_tasks, n=2)
        self.assertTrue(ex)
        self.assertFalse(any(e.get("line_id", "").startswith(t["scene_id"]) for e in ex))

    def test_generation_is_deterministic(self):
        a = tasks.generate_scenes(6, seed=42)
        b = tasks.generate_scenes(6, seed=42)
        self.assertEqual(a, b)

    def test_split_keeps_works_together(self):
        ts = tasks.build_tasks(n_generated=8)
        sp = tasks.split_by_work(ts, holdout_ratio=0.25)
        train_works = {t["work_id"] for t in sp["train"]}
        test_works = {t["work_id"] for t in sp["test"]}
        self.assertFalse(train_works & test_works)


class TestAnnotate(unittest.TestCase):
    """一致性度量本身也要被测——算错 κ 会把「规则表可用」错判成「规则表不可用」。"""

    def test_kappa_perfect_and_chance(self):
        a = [1, 2, 3, 4, 1, 2, 3, 4]
        self.assertAlmostEqual(annotate.cohen_kappa(a, a), 1.0)
        # 两个标注者的边际分布完全不同 → 一致性不高于碰巧，κ ≤ 0
        self.assertLessEqual(annotate.cohen_kappa([1, 1, 1, 1], [2, 2, 2, 2]), 0.0)

    def test_weighted_kappa_near_miss_beats_far_miss(self):
        gold = [1, 2, 3, 4, 5, 6, 7, 8]
        near = [2, 3, 4, 5, 6, 7, 8, 8]      # 每行偏 1 档
        far = [4, 4, 4, 4, 4, 4, 4, 4]       # 一律压中间，与金标几乎无关
        self.assertGreater(annotate.weighted_kappa(near, gold),
                           annotate.weighted_kappa(far, gold))

    def test_weighted_kappa_degenerates_without_variance(self):
        # 取值全同时 κ 无定义（只有一个类别），必须返回 None 而不是 1.0——
        # 真实标注里会出现「某个场景 20 句全标 5」，把它当满分会把整体 κ 抬上去。
        self.assertIsNone(annotate.weighted_kappa([5, 5, 5, 5], [5, 5, 5, 5]))

    def test_spearman_monotone(self):
        self.assertAlmostEqual(annotate.spearman([1, 2, 3], [10, 20, 30]), 1.0)
        self.assertAlmostEqual(annotate.spearman([1, 2, 3], [30, 20, 10]), -1.0)

    def test_valence_bin_covers_edges(self):
        self.assertEqual(annotate.valence_bin(-1.0), 0)
        self.assertEqual(annotate.valence_bin(0.0), 2)
        self.assertEqual(annotate.valence_bin(1.0), 4)
        # 越界值不能抛异常（标注表里可能手滑写 1.5）
        self.assertEqual(annotate.valence_bin(1.5), 4)

    def test_intents_match_reference_table(self):
        used = {row[3] for row in ref.CUE_TABLE} | {"忐忑", "陈述"}
        # 「其他」只存在于人工标注侧，是逃生门：规则表永不输出它，
        # 人选「其他」即该句记为不一致——这正是它该有的语义。
        self.assertEqual(set(annotate.INTENTS) - {"其他"}, used)
        self.assertIn("其他", annotate.INTENTS)


if __name__ == "__main__":
    unittest.main(verbosity=2)
