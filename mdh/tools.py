"""Agent 循环：让模型自己决定调哪个工具，而不是一次性吐 JSON。

为什么要加这一层
----------------
原来的 scaffold 都是"一次 prompt → 一段 JSON"，本质是模板填空。
面试官问「你这里面的 Agent 循环在哪」，答案是「没有」。这个模块补的就是这一层。

三个工具，对应三种真实需要
------------------------
1. `get_character_card(角色名)`
   性格四维决定表演上限。**提示词里不直接给角色卡**，模型必须自己查——
   这才是"用工具获取信息"，而不是"信息已经在上下文里"。

2. `search_similar_shots(台词, k)`
   从**其他作品**的金标准库里检索相似镜头做 few-shot。这就是 RAG：
   检索源必须排除当前场景，否则是抄答案。

3. `validate_params(shots)`
   自检：结构、区间、归因、禁令。把 judge 的 L1 层借过来当工具，
   **注意只回传 L1（规则）错误，绝不回传与金标准的差值**——否则等于把答案给它。

协议
----
沿用项目二既有的 provider 无关动作协议（不依赖原生 function calling）：

    {"think": "...", "tool": "get_character_card", "args": {"name": "沈砚"}}
    {"think": "...", "tool": "finish", "args": {"shots": [...]}}

一次运行 = 最多 max_steps 轮「模型决定 → 执行工具 → 观测回灌」，
模型自己决定什么时候 finish。步数、每步调了什么，全部记进 trace。
"""

from __future__ import annotations

import json
from collections import Counter

from . import reference as ref
from . import schema

TOOL_SPEC = """你可以调用以下工具（每次只调一个，拿到结果后再决定下一步）：

1. get_character_card
   args: {"name": "<角色名>"}
   作用：取该角色的性格四维与音色。**表演上限由性格决定，不查就无法算准强度。**

2. search_similar_shots
   args: {"text": "<台词>", "k": 3}
   作用：从其他作品的已标注镜头里检索相似台词，返回它们的参数作为参考。

3. validate_params
   args: {"shots": [<按契约写好的镜头数组>]}
   作用：自检格式、区间、归因、禁令。返回错误列表；空列表表示通过。

4. finish
   args: {"shots": [<最终镜头数组>]}
   作用：交卷。调用后结束。

输出必须是单个 JSON 对象，形如：
{"think": "一句话说明你为什么要调这个工具", "tool": "<工具名>", "args": {...}}
"""


# --------------------------------------------------------------------------
# 工具实现
# --------------------------------------------------------------------------
def get_character_card(task: dict, args: dict) -> str:
    name = (args or {}).get("name", "")
    card = task["characters"].get(name)
    if not card:
        return json.dumps({"error": f"没有这个角色：{name}；可用角色 {list(task['characters'])}"},
                          ensure_ascii=False)
    return json.dumps({"name": name, **card}, ensure_ascii=False)


def _bigrams(s: str) -> set[str]:
    s = "".join(s.split())
    return {s[i:i + 2] for i in range(max(0, len(s) - 1))}


def search_similar_shots(library: list[dict], args: dict) -> str:
    """按字面二元组重合度检索。检索源必须已排除当前场景（由调用方保证）。"""
    text = (args or {}).get("text", "")
    k = int((args or {}).get("k", 3) or 3)
    q = _bigrams(text)
    scored = []
    for item in library:
        o = _bigrams(item["text"])
        if not q or not o:
            continue
        scored.append((len(q & o) / len(q | o), item))
    scored.sort(key=lambda x: -x[0])
    hits = [{"text": it["text"], "shot": it["shot"]} for _, it in scored[:k] if _ > 0]
    if not hits:
        return json.dumps({"results": [], "note": "库里没有相似台词，请自行判断"}, ensure_ascii=False)
    return json.dumps({"results": hits}, ensure_ascii=False)


def validate_params(task: dict, args: dict) -> str:
    shots = (args or {}).get("shots")
    if not isinstance(shots, list):
        return json.dumps({"errors": ["shots 必须是数组"]}, ensure_ascii=False)
    by_id = {ln["id"]: ln for ln in task["lines"]}
    errs: list[str] = []
    for s in shots:
        if not isinstance(s, dict):
            errs.append("镜头不是对象")
            continue
        errs.extend(schema.validate_shot(s))
        ln = by_id.get(s.get("line_id"))
        if ln is None:
            errs.append(f"未知 line_id：{s.get('line_id')}")
            continue
        errs.extend(schema.check_evidence(s.get("evidence", ""), ln["text"]))
        required = ref.required_constraints(ln.get("scene_type", "日常"))
        declared = set(s.get("constraints") or [])
        for c in required:
            if c not in declared:
                errs.append(f"{ln['id']}: 漏声明禁令 {c}")
        if "no_shout" in required:
            if isinstance(s.get("intensity"), (int, float)) and s["intensity"] > ref.SHOUT_CAP_INTENSITY:
                errs.append(f"{ln['id']}: no_shout 场景 intensity 不得超过 {ref.SHOUT_CAP_INTENSITY}")
            if isinstance(s.get("f0_range"), (int, float)) and s["f0_range"] > ref.SHOUT_CAP_F0_RANGE:
                errs.append(f"{ln['id']}: no_shout 场景 f0_range 不得超过 {ref.SHOUT_CAP_F0_RANGE}")
    # 只报错不够：必须把合法取值一起给它，否则模型会反复试错直到步数耗尽。
    return json.dumps({
        "errors": errs[:20],
        "allowed": {
            "style": list(schema.STYLES),
            "gaze": list(schema.GAZES),
            "ranges": {k: list(v) for k, v in schema.RANGES.items()},
            "au_keys": list(schema.AU_KEYS),
        },
    }, ensure_ascii=False)


# --------------------------------------------------------------------------
# 提示词
# --------------------------------------------------------------------------
def build_agent_prompt(task: dict) -> list[dict]:
    """注意：**故意不把角色卡写进提示词**，逼模型调用工具去查。"""
    parts = [
        f"场次：{task['title']}（scene_id={task['scene_id']}）",
        f"本场角色：{list(task['characters'])}",
        "台词列表：",
    ]
    for ln in task["lines"]:
        parts.append(
            f"- id={ln['id']} 角色={ln['character']} 场景类型={ln.get('scene_type')} "
            f"关系={ln.get('relation')} 台词：「{ln['text']}」"
            + (f"（潜台词：{ln['subtext']}）" if ln.get("subtext") else ""))
    parts.append(TOOL_SPEC)
    parts.append("参数契约（finish 时每个镜头都要满足）：" + json.dumps({
        "line_id": "str", "character": "str (0-10)", "intensity": "number",
        "suppression": "number", "f0_offset": "number", "f0_range": "number",
        "rate": "number", "pause_after_ms": "int",
        "expression": {"style": "压抑|克制|自然|外放", "au": {"AU4": 0}, "gaze": "枚举"},
        "constraints": ["no_shout"], "evidence": "台词原文子串", "director_note": "str",
    }, ensure_ascii=False))
    return [{"role": "system",
             "content": "你是漫剧的表演指导。你需要用工具获取信息，最后调用 finish 交卷。只输出 JSON。"},
            {"role": "user", "content": "\n".join(parts)}]


# --------------------------------------------------------------------------
# 循环
# --------------------------------------------------------------------------
def run_agent(task: dict, llm, library: list[dict] | None = None,
              max_steps: int = 10, repeat: int = 0) -> dict:
    from .agents import extract_json

    library = library or []
    messages = build_agent_prompt(task)
    trace: list[dict] = []
    tokens_in = tokens_out = 0
    shots = None

    for step in range(max_steps):
        res = llm.complete(messages, meta={"scene": task, "scaffold": "agent_loop",
                                           "attempt": step, "repeat": repeat})
        tokens_in += res.tokens_in
        tokens_out += res.tokens_out
        obj = extract_json(res.text)
        tool = (obj or {}).get("tool")
        args = (obj or {}).get("args") or {}
        trace.append({"step": step, "tool": tool,
                      "args_head": json.dumps(args, ensure_ascii=False)[:120]})

        if tool == "finish":
            shots = args.get("shots")
            break
        if tool == "get_character_card":
            obs = get_character_card(task, args)
        elif tool == "search_similar_shots":
            obs = search_similar_shots(library, args)
        elif tool == "validate_params":
            obs = validate_params(task, args)
        else:
            obs = json.dumps({"error": f"未知工具 {tool!r}，可用：get_character_card / "
                                       f"search_similar_shots / validate_params / finish"},
                             ensure_ascii=False)

        trace[-1]["observation_head"] = obs[:160]
        messages = messages + [
            {"role": "assistant", "content": res.text},
            {"role": "user", "content": "工具返回：\n" + obs},
        ]

    return {
        "output": {"scene_id": task["scene_id"], "shots": shots} if shots is not None else None,
        "trace": trace,
        "attempts": len(trace),
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "n_tools": Counter(t["tool"] for t in trace),
    }
