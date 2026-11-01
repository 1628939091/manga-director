"""Agent 循环的单测：用脚本化的假模型驱动，不花 API 调用。

测的是"循环本身"是否正确：工具调用、观测回灌、步数上限、交卷。
"""

from __future__ import annotations

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mdh import agents, reference as ref, tasks, tools  # noqa: E402


class ScriptedLLM:
    """按预设脚本依次返回，每次记录收到的消息，方便断言观测回灌。"""

    def __init__(self, replies: list[dict]):
        self.replies = replies
        self.calls: list[list[dict]] = []

    def complete(self, messages, meta=None):
        self.calls.append(list(messages))
        i = len(self.calls) - 1
        obj = self.replies[min(i, len(self.replies) - 1)]
        text = json.dumps(obj, ensure_ascii=False)

        class R:
            pass
        r = R()
        r.text = text
        r.tokens_in = 10
        r.tokens_out = 20
        return r


def demo_task():
    return tasks.build_tasks(n_generated=0)[0]


def gold_shots(task):
    return [{k: v for k, v in ref.gold_shot(ln, task["characters"]).items()
             if not k.startswith("_")} for ln in task["lines"]]


class TestAgentLoop(unittest.TestCase):
    def setUp(self):
        self.task = demo_task()
        self.shots = gold_shots(self.task)

    def test_full_loop_reaches_finish(self):
        llm = ScriptedLLM([
            {"think": "先查角色卡", "tool": "get_character_card",
             "args": {"name": self.task["lines"][0]["character"]}},
            {"think": "再检索相似镜头", "tool": "search_similar_shots",
             "args": {"text": self.task["lines"][0]["text"], "k": 2}},
            {"think": "交卷", "tool": "finish", "args": {"shots": self.shots}},
        ])
        out = agents.run_scaffold(self.task, "agent_loop", llm, examples=[])
        self.assertIsNotNone(out["output"])
        self.assertEqual(len(out["output"]["shots"]), len(self.task["lines"]))
        self.assertEqual(out["attempts"], 3)
        self.assertEqual(out["n_tools"]["finish"], 1)
        self.assertEqual(out["tokens_in"], 30)

    def test_observation_is_fed_back(self):
        llm = ScriptedLLM([
            {"think": "查卡", "tool": "get_character_card", "args": {"name": "沈砚"}},
            {"think": "交卷", "tool": "finish", "args": {"shots": self.shots}},
        ])
        agents.run_scaffold(self.task, "agent_loop", llm, examples=[])
        # 第二次调用时，上下文里必须出现"工具返回"
        joined = json.dumps(llm.calls[1], ensure_ascii=False)
        self.assertIn("工具返回", joined)

    def test_unknown_tool_returns_error_and_continues(self):
        llm = ScriptedLLM([
            {"think": "乱调", "tool": "delete_everything", "args": {}},
            {"think": "交卷", "tool": "finish", "args": {"shots": self.shots}},
        ])
        out = agents.run_scaffold(self.task, "agent_loop", llm, examples=[])
        self.assertIsNotNone(out["output"])
        self.assertIn("未知工具", json.dumps(out["trace"], ensure_ascii=False))

    def test_max_steps_exhausted_returns_none(self):
        llm = ScriptedLLM([{"think": "一直查", "tool": "get_character_card",
                            "args": {"name": "沈砚"}}])
        out = tools.run_agent(self.task, llm, max_steps=3)
        self.assertIsNone(out["output"])
        self.assertEqual(out["attempts"], 3)

    def test_character_card_tool(self):
        obs = json.loads(tools.get_character_card(self.task, {"name": "沈砚"}))
        self.assertEqual(obs["name"], "沈砚")
        self.assertIn("personality", obs)
        miss = json.loads(tools.get_character_card(self.task, {"name": "不存在"}))
        self.assertIn("error", miss)

    def test_validate_tool_catches_violation(self):
        bad = gold_shots(self.task)
        target = next(i for i, ln in enumerate(self.task["lines"])
                      if ln.get("scene_type") == "亲密低语")
        bad[target]["intensity"] = 9.5
        errs = json.loads(tools.validate_params(self.task, {"shots": bad}))["errors"]
        self.assertTrue(any("intensity" in e for e in errs), errs)

    def test_search_excludes_nothing_but_empty_library_is_handled(self):
        obs = json.loads(tools.search_similar_shots([], {"text": "随便", "k": 3}))
        self.assertEqual(obs["results"], [])
        self.assertIn("note", obs)

    def test_search_ranks_by_overlap(self):
        lib = [{"text": "你给我闭嘴。", "shot": {"line_id": "x"}},
               {"text": "今天天气不错。", "shot": {"line_id": "y"}}]
        obs = json.loads(tools.search_similar_shots(lib, {"text": "闭嘴！", "k": 1}))
        self.assertEqual(obs["results"][0]["text"], "你给我闭嘴。")


if __name__ == "__main__":
    unittest.main(verbosity=2)
