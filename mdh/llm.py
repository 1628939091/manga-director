"""模型后端：真实 API（OpenAI 兼容）+ Mock。

为什么 Mock 要"有缺陷"，而不是回一个正确答案
--------------------------------------------
离线跑通不等于评测有效。一个只会输出金标准的 Mock 会让所有指标看起来满分，
你就永远不知道自己有没有把判卷写对。这里的 Mock 注入的是真实模型最典型的
四类毛病（把情绪当强度、容差边缘抖动、结构漏字段、禁令漏声明），
**只在它们身上能复现已知结论，才算校准过**（对应项目一的 Mock sanity check）。
"""

from __future__ import annotations

import json
import os
import random
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from . import reference as ref
from . import schema


@dataclass
class LLMResult:
    text: str
    tokens_in: int = 0
    tokens_out: int = 0
    meta: dict = field(default_factory=dict)

    @property
    def tokens(self) -> int:
        return self.tokens_in + self.tokens_out


def _est_tokens(s: str) -> int:
    """粗估：中文约 1.6 字/token。只用于离线演示的成本口径，不用于计费。"""
    return max(1, int(len(s) / 1.6))


def _load_dotenv() -> None:
    """读取项目根目录的 .env（KEY=VALUE）。

    已存在的环境变量优先，所以临时 export 的 key 总能覆盖文件里的。
    key 只落在 .env 里，不进代码、不进仓库（.gitignore 已排除）。
    """
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"
    )
    if not os.path.exists(path):
        return
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k and k not in os.environ:
                    os.environ[k] = v
    except OSError:
        pass


# --------------------------------------------------------------------------
# 真实后端
# --------------------------------------------------------------------------
class OpenAICompat:
    """OpenAI 兼容端点（DeepSeek / Qwen / GLM / vLLM 均可）。纯标准库。"""

    def __init__(self, model: str, base_url: str | None = None,
                 api_key: str | None = None, temperature: float = 0.7,
                 seed: int | None = None, timeout: int = 120,
                 max_tokens: int | None = None, max_retries: int = 3):
        _load_dotenv()
        self.model = model
        self.base_url = (base_url or os.environ.get("MDH_BASE_URL")
                         or os.environ.get("DAH_BASE_URL")
                         or "https://api.deepseek.com/v1").rstrip("/")
        self.api_key = (api_key or os.environ.get("MDH_API_KEY")
                        or os.environ.get("DEEPSEEK_API_KEY")
                        or os.environ.get("OPENAI_API_KEY") or "")
        if not self.api_key:
            raise RuntimeError(
                "没有 API key：设置环境变量 DEEPSEEK_API_KEY（或 MDH_API_KEY / MDH_BASE_URL）"
            )
        self.temperature = temperature
        self.seed = seed
        self.timeout = timeout
        # 显式给输出上限。部分厂商把默认 max_tokens 设成上下文全长，
        # 不传就可能 400（项目一撞过 GLM 的这个坑）；一镜到底的 JSON 用 2048 绰绰有余。
        self.max_tokens = max_tokens or int(os.environ.get("MDH_MAX_TOKENS", "2048"))
        self.max_retries = max_retries

    def complete(self, messages: list[dict], meta: dict | None = None) -> LLMResult:
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.seed is not None:
            body["seed"] = self.seed
        data = json.dumps(body).encode("utf-8")

        payload = None
        for attempt in range(self.max_retries + 1):
            req = urllib.request.Request(
                f"{self.base_url}/chat/completions",
                data=data,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", "ignore")[:500]
                # 429 限流与 5xx 是"跑长任务必然遇到"的瞬时错误，退避重试；
                # 4xx 里的其余情况（401/400/402）重试没有意义，直接抛。
                if e.code in (408, 409, 429, 500, 502, 503, 504) and attempt < self.max_retries:
                    wait = min(2 ** attempt * 2, 30)
                    print(f"    [重试 {attempt + 1}/{self.max_retries}] HTTP {e.code}，{wait}s 后重试")
                    time.sleep(wait)
                    continue
                raise RuntimeError(f"HTTP {e.code}: {detail}") from e
            except (urllib.error.URLError, socket.timeout, TimeoutError) as e:
                if attempt < self.max_retries:
                    wait = min(2 ** attempt * 2, 30)
                    print(f"    [重试 {attempt + 1}/{self.max_retries}] 网络错误 {e}，{wait}s 后重试")
                    time.sleep(wait)
                    continue
                raise RuntimeError(f"网络错误：{e}") from e

        if payload is None:
            raise RuntimeError("请求失败：重试次数用尽")

        msg = payload["choices"][0]["message"]
        # 推理型模型有时把内容放在 reasoning_content，content 为空
        text = msg.get("content") or msg.get("reasoning_content") or ""
        usage = payload.get("usage", {})
        return LLMResult(
            text=text,
            tokens_in=usage.get("prompt_tokens", _est_tokens(str(messages))),
            tokens_out=usage.get("completion_tokens", _est_tokens(text)),
            meta={"model": self.model},
        )


# --------------------------------------------------------------------------
# Mock 后端（离线校准用）
# --------------------------------------------------------------------------
def _corrupt(gold: dict, field_name: str, rng: random.Random) -> dict:
    """按字段类型破坏一个字段，破坏幅度刻意落在容差之外。"""
    import copy
    out = copy.deepcopy(gold)
    if field_name == "intensity":
        out["intensity"] = min(10.0, round(out["intensity"] + rng.choice([1.3, -1.2, 1.6]), 2))
    elif field_name == "f0_range":
        out["f0_range"] = min(8.0, round(out["f0_range"] + rng.choice([1.4, -1.3]), 2))
    elif field_name == "f0_offset":
        out["f0_offset"] = round(min(6.0, max(-4.0, out["f0_offset"] + rng.choice([1.5, -1.4]))), 2)
    elif field_name == "rate":
        out["rate"] = round(min(9.0, max(2.0, out["rate"] + rng.choice([0.8, -0.7]))), 2)
    elif field_name == "pause_after_ms":
        out["pause_after_ms"] = int(min(2000, max(0, out["pause_after_ms"] + rng.choice([320, -350]))))
    elif field_name == "style":
        out["expression"]["style"] = rng.choice([s for s in ("压抑", "克制", "自然", "外放")
                                                if s != out["expression"]["style"]])
    elif field_name == "gaze":
        out["expression"]["gaze"] = rng.choice([g for g in (
            "直视对方", "直视但眼神收紧", "避开对视，短暂下移", "自然对视")
            if g != out["expression"]["gaze"]])
    elif field_name == "au":
        au = out["expression"]["au"]
        # 必须把 AU 平均误差推到 0.25 容差之外：改 3 个 × 0.8 → MAE ≈ 0.48
        for k in list(au)[:3]:
            au[k] = round(min(1.0, au[k] + 0.80), 2)
    return out


# 每个 scaffold 每轮的"坏字段数"分布（(坏字段数, 概率)）
# 注意：坏 1 个字段仍可能通过（8 个字段命中 7 个 = 0.875 ≥ τ=0.8），
# 这本身是设计意图——容差之内允许一个字段偏，但一场戏里每条台词都得如此。
DEFECTS = {
    "naive_shot": [(8, 1.0)],          # 全错（naive 参数）
    "card_injected": [(2, 0.35), (1, 0.65)],
    "card_fewshot": [(2, 0.18), (1, 0.82)],
    "card_repair": [(2, 0.35), (1, 0.65)],
    # 给了公式 ≈ 变成算术题：基本全对，偶发一个字段手滑
    "card_formula": [(2, 0.02), (1, 0.20), (0, 0.78)],
}

JUDGED = list(ref.JUDGED_NUMERIC) + ["style", "gaze", "au"]

# 每个 scaffold 出现 L1 缺陷（漏声明禁令 / 归因幻觉）的逐行概率。
# 这类错误才是修复轮能修的东西——L2 的数值偏差按纪律不回传，修不了。
L1_DEFECT_P = {
    "naive_shot": 0.0,
    "card_injected": 0.20,
    "card_fewshot": 0.12,
    "card_repair": 0.20,
    "card_formula": 0.03,
}


def _apply_l1_defect(shot: dict, line: dict, rng: random.Random) -> dict:
    """注入真实模型最常见的两类 L1 错误。"""
    import copy
    out = copy.deepcopy(shot)
    kind = rng.choice(["drop_constraint", "paraphrase_evidence"])
    if kind == "drop_constraint":
        out["constraints"] = []
    else:
        # 复述/改写台词 —— 看起来更"专业"，但已经不是原文子串
        out["evidence"] = line["text"].rstrip("。！？") + "（语气压低）"
    return out


class MockDirector:
    """模拟一个"懂任务但会抖"的模型。确定性：seed 由 scene_id + attempt 决定。"""

    def __init__(self, name: str = "mock"):
        self.name = name

    def complete(self, messages: list[dict], meta: dict | None = None) -> LLMResult:
        meta = meta or {}
        scene, scaffold = meta["scene"], meta["scaffold"]
        attempt = int(meta.get("attempt", 0))
        repeat = int(meta.get("repeat", 0))
        prompt = str(messages)

        if scaffold in ("judge_plain", "judge_cue"):
            text = self._appraisal(scene, scaffold, repeat, attempt)
            return LLMResult(
                text=text,
                tokens_in=_est_tokens(prompt),
                tokens_out=_est_tokens(text),
                meta={"model": self.name, "attempt": attempt},
            )

        shots = []
        for line in scene["lines"]:
            gold = ref.gold_shot(line, scene["characters"])
            rng = random.Random(f"{scene['scene_id']}|{line['id']}|{scaffold}|{repeat}|{attempt}")

            if scaffold == "naive_shot":
                shot = self._naive(line, scene, gold, rng)
            else:
                n_bad = self._draw_bad(scaffold, attempt, rng)
                shot = dict(gold)
                for f in rng.sample(JUDGED, n_bad):
                    shot = _corrupt(shot, f, rng)
                shot = {k: v for k, v in shot.items() if not k.startswith("_")}

                p_l1 = L1_DEFECT_P[scaffold]
                if scaffold == "card_repair" and attempt > 0:
                    p_l1 = 0.02
                if rng.random() < p_l1:
                    shot = _apply_l1_defect(shot, line, rng)

            # 结构缺陷：小概率把 JSON 包在散文里（真实模型非常常见的失败模式）
            shots.append(shot)

        obj = {"scene_id": scene["scene_id"], "shots": shots}
        text = json.dumps(obj, ensure_ascii=False, indent=2)
        if scaffold == "naive_shot" and random.Random(scene["scene_id"] + "wrap").random() < 0.5:
            text = "好的，下面是这场戏的表演设计：\n```json\n" + text + "\n```\n希望有帮助！"

        return LLMResult(
            text=text,
            tokens_in=_est_tokens(prompt),
            tokens_out=_est_tokens(text),
            meta={"model": self.name, "attempt": attempt},
        )

    def _appraisal(self, scene: dict, scaffold: str, repeat: int, attempt: int) -> str:
        """判读层的 Mock：注入"唤醒度读不准 + 意图误判 + 归因改写"三类缺陷。

        缺陷率按臂给：给了线索表的臂（judge_cue）读得更准一点，
        但仍然会偏——这是真实模型的形态，不是"给了表就满分"。
        """
        plain = scaffold == "judge_plain"
        p_shift = 0.55 if plain else 0.22
        p_intent = 0.35 if plain else 0.15
        p_ev = 0.05 if plain else 0.03

        out = []
        for line in scene["lines"]:
            g = ref.appraise(line["text"], line.get("subtext", ""))
            rng = random.Random(f"{scene['scene_id']}|{line['id']}|{scaffold}|{repeat}|{attempt}")
            arousal = g["arousal"]
            if rng.random() < p_shift:
                mag = rng.uniform(1.5, 3.5) if plain else rng.uniform(0.5, 1.8)
                arousal = round(min(10.0, max(0.0, arousal + (mag if rng.random() < 0.5 else -mag))), 2)

            intent = g["intent"]
            if rng.random() < p_intent:
                intent = rng.choice([x for x in schema.INTENTS if x != g["intent"]])

            evidence = line["text"] if rng.random() > p_ev else line["text"][:max(2, len(line["text"]) // 3)]

            out.append({
                "line_id": line["id"],
                "arousal": arousal,
                "valence": g["valence"],
                "intent": intent,
                "evidence": evidence,
            })

        return json.dumps({"scene_id": scene["scene_id"], "lines": out}, ensure_ascii=False, indent=2)

    @staticmethod
    def _draw_bad(scaffold: str, attempt: int, rng: random.Random) -> int:
        table = DEFECTS[scaffold]
        if scaffold == "card_repair" and attempt > 0:
            # 修复轮：模型看到了具体错误，显著收敛（但没全对）
            return 2 if rng.random() < 0.08 else 1
        r = rng.random()
        acc = 0.0
        for n, p in table:
            acc += p
            if r <= acc:
                return n
        return table[-1][0]

    @staticmethod
    def _naive(line: dict, scene: dict, gold: dict, rng: random.Random) -> dict:
        """AI 漫剧现状：内部情绪多少分，就演多少分（抑制系数恒等于 1）。"""
        internal = gold["_internal"]
        baseline = gold["_baseline"]
        arousal = internal["arousal"]
        intensity = round(max(1.0, arousal), 2)

        if intensity >= 7.0:
            style, gaze = "外放", "直视对方"
        elif intensity >= 5.0:
            style, gaze = "自然", "自然对视"
        else:
            style, gaze = "压抑", "避开对视，短暂下移"

        return {
            "line_id": line["id"],
            "character": line["character"],
            "intensity": intensity,
            "suppression": 1.0,
            "f0_offset": round(baseline["f0_base"] + 0.45 * (intensity - 4.0), 2),
            "f0_range": round(min(8.0, 1.0 + 0.55 * intensity), 2),
            "rate": round(baseline["rate_base"] * (0.90 + 0.040 * intensity), 2),
            "pause_after_ms": int(120 + (10 - intensity) * 25),
            "expression": {
                "style": style,
                "au": {"AU4": 0.55, "AU20": 0.70, "AU26": 0.55, "AU12": 0.15, "AU17": 0.20},
                "gaze": gaze,
            },
            "constraints": [],          # naive 模型根本不知道有禁令
            "evidence": line["text"],
            "director_note": "按台词情绪直接给强度。",
        }
