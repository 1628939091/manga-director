"""mdh — MangaDirector Harness.

一个"答案不唯一"的生成任务评测框架：规则层 + 参考实现容差层 + pass^k。
"""

__version__ = "0.1.0"

from . import agents, judge, metrics, reference, report, runner, schema, tasks  # noqa: F401

__all__ = ["agents", "judge", "metrics", "reference", "report", "runner", "schema", "tasks"]
