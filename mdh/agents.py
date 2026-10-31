"""Agent 脚手架（scaffold）：同一任务、同一模型，换脚手架看可靠性怎么变。

四个（+1）脚手架，构成本项目的核心消融
------------------------------------
1. `naive_shot`      —— 极简提示，不给角色卡，不给场景规范（模拟"AI 直觉版"）
2. `card_injected`   —— 给角色卡 + 场景规范 + 契约示例，但**不给公式**
3. `card_fewshot`    —— 在 2 的基础上加 2 条金标准示例（只从训练作品里取）
4. `card_repair`     —— 在 2 的基础上加"校验失败自动重试"
5. `card_formula`    —— 在 2 的基础上把参考实现的公式与线索表**原样交出**

第 5 个是这套设计里最有信息量的一个臂：如果给了公式它就过了，说明
"表演指导"这层能力在模型侧是**计算**而不是**判断**——瓶颈不在生成，
在"从台词读出内心情绪"那一步。这个结论比"我的 Agent 准确率 85%"值钱得多。

一条不可让步的纪律：修复反馈只回传 L1（规则层）错误
------------------------------------------------
L1 错误（缺字段、越界、evidence 不是原文、禁令没执行）不泄露金标准数值，
可以安全回传给模型。一旦把 L2 的"你比金标准高了 1.2 分"喂回去，
修复轮就变成了对着答案改错，pass^k 会被系统性地虚高——
这是评测污染在 agent 循环里的变体，必须结构性避免。
"""

from __future__ import annotations

import json
import re

from . import pipeline
from . import reference as ref
from . import schema

SCAFFOLDS = ("naive_shot", "card_injected", "card_fewshot", "card_repair",
             "card_formula", "judge_plain", "judge_cue")

# 两层架构的两个臂：模型只判读，标定由 pipeline 完成
JUDGE_SCAFFOLDS = ("judge_plain", "judge_cue")


# --------------------------------------------------------------------------
# JSON 抽取（真实模型最常见的失败：散文里包 JSON / markdown 围栏）
# --------------------------------------------------------------------------
def extract_json(text: str) -> dict | None:
    if not isinstance(text, str):
        return None
    s = text.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)

    for candidate in (s, s[s.find("{"):s.rfind("}") + 1] if "{" in s else ""):
        if not candidate:
            continue
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    # 括号配平扫描：从第一个 { 开始找配对的 }
    start = s.find("{")
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(s)):
        c = s[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(s[start:i + 1])
                    return obj if isinstance(obj, dict) else None
                except json.JSONDecodeError:
                    return None
    return None


# --------------------------------------------------------------------------
# 提示词
# --------------------------------------------------------------------------
SYSTEM = (
    "你是漫剧的表演指导。给定角色卡与台词，为每条台词输出一份可执行的表演参数。"
    "只输出 JSON，不要解释。"
)

CONTRACT_DOC = f"""输出格式（每个镜头一次决策）：
{{"scene_id": "<场景id>", "shots": [
  {{"line_id": "<台词id>", "character": "<角色名>",
    "intensity": 0-10 的外部表演强度,
    "suppression": 0-1 的抑制系数（外部强度 / 内心强度）,
    "f0_offset": -4~6 的半音偏移（相对该角色基线）,
    "f0_range": 0.5~8 的音高动态范围（半音，喊叫的核心特征）,
    "rate": 2~9 的语速（字/秒）,
    "pause_after_ms": 0~2000 的句尾停顿,
    "expression": {{"style": "{' / '.join(schema.STYLES)}",
                    "au": {{"AU4":0~1,"AU12":0~1,"AU17":0~1,"AU20":0~1,"AU26":0~1}},
                    "gaze": "{' / '.join(schema.GAZES)}"}},
    "constraints": ["本镜头生效的禁令，如 no_shout"],
    "evidence": "必须是该台词原文的一个连续子串",
    "director_note": "一句人话导演备注"}}
]}}"""

PRINCIPLES = """表演原则（必须遵守）：
1. 内心激烈不等于外部激烈。越是内敛的角色，抑制系数越低。
   外部强度 = 内心强度 × 表现欲，再被「性格上限」和「场景上限」双重截断。
2. 私密场景要压低音量、以气声为主；公共场景可以放开，但仍有上限。
3. 场景规范里 allow_shout 为 false 时，**不许喊**：
   必须声明 no_shout，且 intensity 不超过 6.5、f0_range 不超过 5.0、
   style 不能是「外放」。
4. evidence 必须是台词原文子串——不要复述、不要改写、不要引用潜台词。"""

FORMULA = """参考实现的精确算法（照此计算，不要凭感觉）：
- 表现欲 expressiveness = 0.3·energy + 0.3·dominance + 0.2·neuroticism + 0.2·warmth
- 性格上限 = 3.5 + 4.5·expressiveness
- 场景上限 = ceiling + 关系修正（长辈-1.2 上级-1.0 陌生人-0.6 挚友+0.4 恋人+0.3 恋人未满-0.2 对手+0.2）
- 内心唤醒 arousal 由台词线索表决定：
  滚/闭嘴/你敢/最后一遍/再说一次→8.6；对不起/是我的错/我认→5.2；别走/等等/求你→7.4；
  我喜欢你/想你/别怕/我在→6.2；嗯/知道了/随便/随你→4.0；哈哈/笑死/真的假的→6.8；
  哇/好可爱/太棒/天哪/好美/好厉害→7.2；死/血/来不及/快跑→9.0；为什么/凭什么/你说什么→7.8；
  都不匹配→5.0；潜台词含 心跳/其实/忍住/怕/紧张 时 arousal +1.2（上限 10）
- intensity = max(1.0, min(arousal·(0.60+0.40·expressiveness), 性格上限, 场景上限))
  若 allow_shout=false 再取 min(intensity, 6.5)
- suppression = intensity / max(arousal, 0.1)
- f0_range = min(6.0, 0.9 + 0.50·intensity)
- f0_offset = 角色基线 f0_base + 0.45·(intensity - 4.0)，其中 f0_base = -1.5 + 1.5·(1-dominance)
- rate = rate_base · (0.85 + 0.035·intensity)，其中 rate_base = 3.6 + 1.2·energy
- pause_after_ms = 180 + (10-intensity)·35 + (260 if withheld≥2 else 0)，withheld = max(0, arousal-intensity)
- style/au/gaze 按 withheld 与 intensity 分档：
  withheld≥2 且 intensity<5 → 压抑；withheld≥2 → 克制；intensity≥7 → 外放；否则自然。
  压抑 AU4=.75 AU20=.10 AU26=.08 AU12=.05 AU17=.55 / 避开对视，短暂下移
  克制 AU4=.80 AU20=.35 AU26=.25 AU12=.08 AU17=.45 / 直视但眼神收紧
  外放 AU4=.55 AU20=.70 AU26=.55 AU12=.15 AU17=.20 / 直视对方
  自然 AU4=.35 AU20=.20 AU26=.15 AU12=.30 AU17=.10 / 自然对视"""


def _render_card(name: str, card: dict) -> str:
    p = card.get("personality", {})
    return (f"- {name}（{card.get('role','角色')}，音色：{card.get('voice','—')}）"
            f"warmth={p.get('warmth')} dominance={p.get('dominance')} "
            f"energy={p.get('energy')} neuroticism={p.get('neuroticism')}")


# --------------------------------------------------------------------------
# 判读层（两层架构的第一层）
# --------------------------------------------------------------------------
INTENT_GLOSS = {
    "威胁": "放狠话、压迫对方",
    "认错": "道歉、服软",
    "挽留": "求对方别走、别放弃",
    "表白/安抚": "表达喜欢，或安慰对方",
    "隐忍": "把情绪压下去，敷衍、应付",
    "玩笑": "打趣、调侃、自嘲",
    "惊喜": "惊叹、赞叹",
    "危机": "危险、紧急、来不及",
    "质疑": "反问、不服、追问",
    "忐忑": "紧张、不安、怕被拒绝",
    "陈述": "平铺直叙，没有明显情绪",
}

APPRAISAL_SYSTEM = (
    "你是漫剧的表演指导助手，本轮只负责一件事：**读出台词背后的内心状态**。"
    "外部表演参数（强度、音高、语速、停顿、表情）由下游系统按公式计算，"
    "你不需要、也不允许输出它们。只输出 JSON。"
)

APPRAISAL_DOC = """输出格式（一场戏一次）：
{"scene_id": "<场景id>", "lines": [
  {"line_id": "<台词id>",
   "arousal": 0-10 的**内心**唤醒强度（不是外部表现强度；
              内心汹涌但表面克制，arousal 仍然高）,
   "valence": -1~1 的情绪效价（负=难受/敌意，正=愉悦/亲近）,
   "intent": "从下面的枚举里选一个",
   "evidence": "该台词原文的一个连续子串，用来证明你的判断有依据"}
]}

intent 枚举（必须逐字使用，只能选一个）：
""" + "\n".join(f"- {k}：{v}" for k, v in INTENT_GLOSS.items())

CUE_TABLE_DOC = """参考：台词线索 → 内心唤醒（可借鉴，但不要机械套用）
威胁类（滚/闭嘴/你敢/最后一遍/再说一次）≈8.6；认错类（对不起/是我的错/我认）≈5.2；
挽留类（别走/等等/求你）≈7.4；表白安抚类（我喜欢你/想你/别怕/我在）≈6.2；
隐忍类（嗯/知道了/随便/随你）≈4.0；玩笑类（哈哈/笑死/真的假的）≈6.8；
惊喜类（哇/好可爱/太棒/天哪/好美/好厉害）≈7.2；危机类（死/血/来不及/快跑）≈9.0；
质疑类（为什么/凭什么/你说什么）≈7.8；都不匹配≈5.0。
潜台词里出现 心跳/其实/忍住/怕/紧张 时，arousal 再 +1.2（上限 10）。"""


def build_appraisal_prompt(task: dict, scaffold: str) -> list[dict]:
    lines = task["lines"]
    parts: list[str] = [f"场次：{task['title']}（scene_id={task['scene_id']}）", "", "角色卡："]
    for name, card in task["characters"].items():
        parts.append(_render_card(name, card))

    parts.append("")
    parts.append("台词列表：")
    for ln in lines:
        parts.append(
            f"- id={ln['id']} 角色={ln['character']} 场景={ln.get('scene_type')} "
            f"关系={ln.get('relation')} 对{ln.get('listener', '—')}说：「{ln['text']}」"
            + (f"（潜台词：{ln['subtext']}）" if ln.get("subtext") else "")
        )

    parts.append("")
    parts.append(
        "为每条台词判断：说这句话的人**内心**有多激烈（arousal）、"
        "情绪是正还是负（valence）、这句话在做什么（intent）。"
    )
    if scaffold == "judge_cue":
        parts.append("")
        parts.append(CUE_TABLE_DOC)
    parts.append("")
    parts.append(APPRAISAL_DOC)
    parts.append("")
    parts.append(f"现在为 {task['scene_id']} 的 {len(lines)} 条台词输出判读结果。")
    return [{"role": "system", "content": APPRAISAL_SYSTEM},
            {"role": "user", "content": "\n".join(parts)}]


def appraisal_feedback(output: dict, task: dict) -> list[str]:
    """判读层的 L1 反馈：结构、区间、枚举、归因。同样不泄露金标准。"""
    if not isinstance(output, dict):
        return ["输出不是 JSON 对象"]
    raw = output.get("lines") if isinstance(output.get("lines"), list) else output.get("shots")
    if not isinstance(raw, list):
        return ["输出缺少 lines 数组"]

    by_id = {x.get("line_id"): x for x in raw if isinstance(x, dict)}
    msgs: list[str] = []
    for ln in task["lines"]:
        item = by_id.get(ln["id"])
        if item is None:
            msgs.append(f"{ln['id']}: 缺这条台词的判读")
            continue
        msgs.extend(f"{ln['id']}: {e}" for e in schema.validate_appraisal(item))
        if isinstance(item.get("evidence"), str):
            msgs.extend(f"{ln['id']}: {e}"
                        for e in schema.check_evidence(item["evidence"], ln["text"]))
    return msgs


def build_prompt(task: dict, scaffold: str,
                 examples: list[dict] | None = None) -> list[dict]:
    if scaffold in JUDGE_SCAFFOLDS:
        return build_appraisal_prompt(task, scaffold)

    lines = task["lines"]
    scene_types = {ln.get("scene_type", "日常") for ln in lines}

    parts: list[str] = [f"场次：{task['title']}（scene_id={task['scene_id']}）"]

    if scaffold == "naive_shot":
        parts.append("台词列表：")
        for ln in lines:
            parts.append(f"- id={ln['id']} 角色={ln['character']} 台词：「{ln['text']}」")
        parts.append(
            "请为每句台词给出表演参数，输出 JSON："
            '{"scene_id": "...", "shots": [{"line_id": "...", "character": "...", '
            '"intensity": 数值, "evidence": "台词原文"}]}'
        )
        return [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": "\n".join(parts)}]

    parts.append("角色卡：")
    for name, card in task["characters"].items():
        parts.append(_render_card(name, card))

    parts.append("场景规范：")
    for st in sorted(scene_types):
        prof = ref.SCENE_PROFILE.get(st, ref.SCENE_PROFILE["日常"])
        parts.append(
            f"- {st}：ceiling={prof['ceiling']}, space={prof['space']}, "
            f"allow_shout={prof['allow_shout']} → 必守禁令 "
            f"{ref.required_constraints(st) or '无'}"
        )

    parts.append("台词列表：")
    for ln in lines:
        parts.append(
            f"- id={ln['id']} 角色={ln['character']} 场景={ln.get('scene_type')} "
            f"关系={ln.get('relation')} 对{ln.get('listener','—')}说：「{ln['text']}」"
            + (f"（潜台词：{ln['subtext']}）" if ln.get("subtext") else "")
        )

    parts.append(PRINCIPLES)
    if scaffold == "card_formula":
        parts.append(FORMULA)
    parts.append(CONTRACT_DOC)

    if examples:
        parts.append("参考示例（来自其他作品的已标注镜头，数值可直接借鉴）：")
        for ex in examples:
            parts.append(json.dumps(ex, ensure_ascii=False, indent=1))

    parts.append(f"现在为 {task['scene_id']} 的 {len(lines)} 条台词输出完整 JSON。")
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": "\n".join(parts)}]


# --------------------------------------------------------------------------
# 只回传 L1 错误（安全：不泄露金标准数值）
# --------------------------------------------------------------------------
def l1_feedback(output: dict, task: dict) -> list[str]:
    msgs: list[str] = []
    shots = output.get("shots") if isinstance(output, dict) else None
    if not isinstance(shots, list):
        return ["输出缺少 shots 数组"]

    by_id = {s.get("line_id"): s for s in shots if isinstance(s, dict)}
    for ln in task["lines"]:
        shot = by_id.get(ln["id"])
        if shot is None:
            msgs.append(f"{ln['id']}: 缺这个镜头")
            continue
        for e in schema.validate_shot(shot):
            msgs.append(f"{ln['id']}: {e}")
        for e in schema.check_evidence(shot.get("evidence", ""), ln["text"]):
            msgs.append(f"{ln['id']}: {e}")
        required = ref.required_constraints(ln.get("scene_type", "日常"))
        declared = set(shot.get("constraints") or [])
        for c in required:
            if c not in declared:
                msgs.append(f"{ln['id']}: 漏声明禁令 {c}")
        if "no_shout" in required:
            if isinstance(shot.get("intensity"), (int, float)) and shot["intensity"] > ref.SHOUT_CAP_INTENSITY:
                msgs.append(f"{ln['id']}: no_shout 场景 intensity 不得超过 {ref.SHOUT_CAP_INTENSITY}")
            if isinstance(shot.get("f0_range"), (int, float)) and shot["f0_range"] > ref.SHOUT_CAP_F0_RANGE:
                msgs.append(f"{ln['id']}: no_shout 场景 f0_range 不得超过 {ref.SHOUT_CAP_F0_RANGE}")
    return msgs


# --------------------------------------------------------------------------
# 执行
# --------------------------------------------------------------------------
def run_scaffold(task: dict, scaffold: str, llm, examples: list[dict] | None = None,
                 max_repair: int = 2, repeat: int = 0) -> dict:
    if scaffold not in SCAFFOLDS:
        raise ValueError(f"未知 scaffold: {scaffold}")

    judge_mode = scaffold in JUDGE_SCAFFOLDS
    messages = build_prompt(task, scaffold, examples)
    trace: list[dict] = []
    tokens_in = tokens_out = 0
    output: dict | None = None
    attempts = 0

    for attempt in range(max_repair + 1):
        attempts = attempt + 1
        res = llm.complete(messages, meta={"scene": task, "scaffold": scaffold,
                                           "attempt": attempt, "repeat": repeat})
        tokens_in += res.tokens_in
        tokens_out += res.tokens_out
        text = res.text
        # 判读层偶尔会把 JSON 包在散文里，且 reasoning 模型可能把答案放在 reasoning_content
        output = extract_json(text)
        trace.append({"attempt": attempt, "raw_head": text[:180]})

        if output is None:
            feedback = ["输出不是合法 JSON，请只输出 JSON 对象"]
        elif judge_mode:
            feedback = appraisal_feedback(output, task)
        else:
            feedback = l1_feedback(output, task)

        trace[-1]["l1_errors"] = feedback
        if not feedback:
            break
        if scaffold != "card_repair":
            break
        if attempt == max_repair:
            break
        messages = messages + [
            {"role": "assistant", "content": text},
            {"role": "user", "content":
                "上面的输出有这些问题，请修正后重新输出**完整** JSON：\n- "
                + "\n- ".join(feedback[:12])},
        ]

    appraisal = output if judge_mode else None
    if judge_mode:
        # 第二层：判读 → 标定 → 完整契约。走同一个 compose，参数不与模型协商。
        output = pipeline.compose_scene(appraisal or {}, task)

    return {
        "output": output,
        "appraisal": appraisal,
        "trace": trace,
        "attempts": attempts,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
    }
