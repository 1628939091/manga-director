"""人工标注入口 —— 给「金标准本身是关键词表」这个漏洞补尺子。

用法::

    python annotate.py export            # 生成 annotations/form.csv（不含规则答案）
    python annotate.py export --limit 60 # 只想先标一小批
    python annotate.py score             # 算 κ，并用人类金标重算模型指标

为什么要有这个脚本
------------------
此前的 arousal MAE、意图准确率，比的都是 reference.appraise 那张关键词表的输出。
它只看 text + subtext，看不到场景与关系，所以「与金标准一致」只等于
「与我的规则表一致」。这不是能力结论，是自证。

本脚本不产生新的模型分数，只回答一个问题：**那张规则表配不配当金标准。**
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mdh import annotate  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(annotate.main(sys.argv))
