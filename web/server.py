"""MangaDirector Studio —— 单页端：把 mdh 包成一个真的能点的东西。

为什么用标准库而不是 Flask
--------------------------
这个项目的卖点是"判卷可复算、链路可审计"，不是 Web 框架。零第三方依赖意味着
任何人 clone 下来 `python web/server.py` 就能跑——评审者不会因为装不上环境而跳过它。
与 mdh/llm.py 用 urllib 而不是 requests 是同一个理由。

三个接口
--------
GET  /api/scenes          场景清单（13 场：1 手写 + 12 模板生成）
GET  /api/scene?id=       单个场景 + 金标准镜头 + naive（AI 直觉版）对照
POST /api/run             跑 k 次，返回每次的镜头、判卷结果、k 次分歧

一个刻意的设计：/api/run 一定返回**每次原始镜头**，不只是聚合指标。
页面要能让人看见"这 k 次到底哪里不一样"——只给一个 pass^k 数字，
等于又退回项目二那种"只看分数不看分歧"的老路。
"""

from __future__ import annotations

import json
import os
import sys
import time
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mdh import agents, judge, llm, metrics, runner, tasks  # noqa: E402
from mdh import reference as ref  # noqa: E402
from mdh import schema  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
# 页面默认只加载 13 场（12 生成 + 1 手写示例）：页面是交互式点选，同步请求扛不住 51 场。
# 要看全量就 MDH_N_SCENES=50 启动。
SCENES = tasks.build_tasks(n_generated=int(os.environ.get("MDH_N_SCENES", "12")))
BY_ID = {t["scene_id"]: t for t in SCENES}

RUN_K_MAX = 8        # 单场：一次只跑一场戏，k 大一点也还快
COMPARE_K_MAX = 5    # 全量：13 场 × 5 臂 × k，同步请求扛不住更大的 k

# 只缓存 mock：mock 是确定性的，重算没有新信息；
# 真实模型每次点击都应该是**新采样的 k 次**，缓存会悄悄毁掉 pass^k 的语义。
_COMPARE_CACHE: dict[str, dict] = {}

_PUBLIC_GOLD = ("line_id", "character", "intensity", "suppression", "f0_offset",
                "f0_range", "rate", "pause_after_ms", "expression", "constraints",
                "evidence", "director_note")


def _safe_static_path(rel: str) -> str | None:
    """/static/<rel> -> 安全路径；越界返回 None。

    只挡 ".." 是不够的：os.path.join 遇到绝对路径会**丢弃左边**，
    所以 /static/D:/secret.txt 会解析到静态目录之外。
    """
    static = os.path.realpath(STATIC)
    target = os.path.realpath(os.path.join(static, rel))
    if target == static or not target.startswith(static + os.sep):
        return None
    return target


def _invalid_scaffolds(names) -> list[str]:
    return [n for n in names if n not in agents.SCAFFOLDS]


def _gold_bundle(task: dict) -> list[dict]:
    """金标准 + naive 对照。naive_intensity = 内部唤醒直接当外部强度（抑制系数恒为 1）。"""
    out = []
    for line in task["lines"]:
        g = ref.gold_shot(line, task["characters"])
        out.append({
            "line_id": line["id"],
            "text": line["text"],
            "character": line["character"],
            "listener": line.get("listener", ""),
            "relation": line.get("relation", ""),
            "scene_type": line.get("scene_type", ""),
            "subtext": line.get("subtext", ""),
            "gold": {k: g[k] for k in _PUBLIC_GOLD},
            "internal": g["_internal"],
            "naive_intensity": g["_naive_intensity"],
            "withheld": g["_withheld"],
        })
    return out


def _scene_brief(t: dict) -> dict:
    return {
        "scene_id": t["scene_id"],
        "title": t["title"],
        "work_id": t["work_id"],
        "n_lines": len(t["lines"]),
        "scene_types": sorted({ln.get("scene_type", "日常") for ln in t["lines"]}),
        "cast": list(t["characters"]),
    }


def _diverge(runs: list[dict], lines: list[dict]) -> dict:
    """k 次运行的分歧：逐行逐字段收集取值，标出不一致的。"""
    fields = list(ref.JUDGED_NUMERIC) + ["style", "gaze"]
    out = {}
    for ln in lines:
        lid = ln["id"]
        rows = []
        for f in fields:
            vals = []
            for r in runs:
                shot = next((s for s in (r["shots"] or []) if s.get("line_id") == lid), None)
                if shot is None:
                    vals.append(None)
                elif f in ("style", "gaze"):
                    vals.append((shot.get("expression") or {}).get(f))
                else:
                    vals.append(shot.get(f))
            rows.append({
                "field": f,
                "values": vals,
                "distinct": len({json.dumps(v, ensure_ascii=False) for v in vals}),
            })
        out[lid] = rows
    return out


class Handler(BaseHTTPRequestHandler):
    server_version = "MDH-Studio/0.1"

    # ---------------- 基础响应 ----------------
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def _file(self, path: str) -> None:
        if not os.path.isfile(path):
            self._json({"error": "not found", "path": path}, 404)
            return
        ext = os.path.splitext(path)[1].lower()
        ctype = {".html": "text/html; charset=utf-8",
                 ".js": "text/javascript; charset=utf-8",
                 ".css": "text/css; charset=utf-8",
                 ".svg": "image/svg+xml",
                 ".json": "application/json; charset=utf-8"}.get(ext, "application/octet-stream")
        with open(path, "rb") as f:
            self._send(200, f.read(), ctype)

    def log_message(self, fmt, *args):  # 别把每次请求都刷到终端
        pass

    # ---------------- GET ----------------
    def do_GET(self):
        u = urlparse(self.path)
        p = u.path
        if p in ("/", "/index.html"):
            return self._file(os.path.join(STATIC, "index.html"))
        if p.startswith("/static/"):
            target = _safe_static_path(p[len("/static/"):])
            if target is None:
                return self._json({"error": "bad path"}, 400)
            return self._file(target)
        if p == "/api/scenes":
            return self._json({"scenes": [_scene_brief(t) for t in SCENES]})
        if p == "/api/scene":
            sid = (parse_qs(u.query).get("id") or [""])[0]
            t = BY_ID.get(sid)
            if not t:
                return self._json({"error": f"未知场景 {sid}"}, 404)
            return self._json({
                "scene": _scene_brief(t),
                "characters": t["characters"],
                "lines": _gold_bundle(t),
                "scene_profile": {st: ref.SCENE_PROFILE.get(st) for st in
                                  {ln.get("scene_type", "日常") for ln in t["lines"]}},
            })
        if p == "/api/meta":
            return self._json({
                "scaffolds": list(agents.SCAFFOLDS),
                "tau": judge.DEFAULT_TAU,
                "tol": judge.DEFAULT_TOL,
                "judged_fields": list(ref.JUDGED_FIELDS),
                "styles": list(schema.STYLES),
                "api_key_present": bool(
                    os.environ.get("MDH_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")
                    or os.environ.get("OPENAI_API_KEY")),
                "base_url": os.environ.get("MDH_BASE_URL", "https://api.deepseek.com/v1"),
            })
        return self._json({"error": "not found"}, 404)

    # ---------------- POST ----------------
    def do_POST(self):
        u = urlparse(self.path)
        if u.path == "/api/compare":
            return self._compare()
        if u.path != "/api/run":
            return self._json({"error": "not found"}, 404)

        try:
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n).decode("utf-8") or "{}")
        except (ValueError, UnicodeDecodeError) as e:
            return self._json({"error": f"请求体不是合法 JSON：{e}"}, 400)

        sid = body.get("scene_id") or ""
        scaffold = body.get("scaffold") or "card_injected"
        k = max(1, min(RUN_K_MAX, int(body.get("k") or 3)))
        backend_name = body.get("backend") or "mock"
        model = body.get("model") or "deepseek-chat"

        task = BY_ID.get(sid)
        if not task:
            return self._json({"error": f"未知场景 {sid}"}, 404)
        if scaffold not in agents.SCAFFOLDS:
            return self._json({"error": f"未知 scaffold {scaffold}"}, 400)

        if backend_name == "api":
            try:
                backend = llm.OpenAICompat(model=model, temperature=0.7, seed=20261030)
            except RuntimeError as e:
                return self._json({"error": str(e)}, 400)
        else:
            backend = llm.MockDirector()

        golds = {ln["id"]: ref.gold_shot(ln, task["characters"]) for ln in task["lines"]}
        examples = (runner.fewshot_examples(task, SCENES, 2)
                    if scaffold == "card_fewshot" else None)

        runs = []
        for rep in range(k):
            out = agents.run_scaffold(task, scaffold, backend, examples=examples,
                                      max_repair=2, repeat=rep)
            j = judge.judge_scene(out["output"], task, golds, tau=judge.DEFAULT_TAU)
            shots = (out["output"] or {}).get("shots")
            runs.append({
                "repeat": rep,
                "passed": j["passed"],
                "hit_rate": j["hit_rate"],
                "min_hit_rate": j["min_hit_rate"],
                "l1_clean": j["l1_clean"],
                "errors": j["errors"],
                "violations": j["violations"],
                "attempts": out["attempts"],
                "tokens": out["tokens_in"] + out["tokens_out"],
                "shots": shots if isinstance(shots, list) else [],
                "rows": [{
                    "line_id": r["line_id"], "hit_rate": r["hit_rate"],
                    "l1_clean": r["l1_clean"], "hits": r["hits"],
                    "deltas": r["deltas"], "errors": r["errors"],
                    "violations": r["violations"],
                } for r in j["rows"]],
                "trace": [t.get("l1_errors", []) for t in out["trace"]],
            })

        n_pass = sum(1 for r in runs if r["passed"])
        return self._json({
            "scene_id": sid,
            "scaffold": scaffold,
            "k": k,
            "backend": backend_name,
            "model": getattr(backend, "model", getattr(backend, "name", "-")),
            "tau": judge.DEFAULT_TAU,
            "summary": {
                "pass_at_1": round(n_pass / len(runs), 4),
                "pass_pow_k": 1.0 if n_pass == len(runs) else 0.0,
                "mean_hit_rate": round(sum(r["hit_rate"] for r in runs) / len(runs), 4),
                "violations": sum(len(r["violations"]) for r in runs),
                "l1_clean_rate": round(sum(1 for r in runs if r["l1_clean"]) / len(runs), 4),
            },
            "runs": runs,
            "diverge": _diverge(runs, task["lines"]),
        })


    # ---------------- 全量五臂对比 ----------------
    def _compare(self):
        """跑全部 13 场 × 5 个 scaffold，出消融表。这是能直接进简历的那张图。"""
        try:
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n).decode("utf-8") or "{}")
        except (ValueError, UnicodeDecodeError) as e:
            return self._json({"error": f"请求体不是合法 JSON：{e}"}, 400)

        k = max(1, min(COMPARE_K_MAX, int(body.get("k") or 3)))
        backend_name = body.get("backend") or "mock"
        model = body.get("model") or "deepseek-chat"
        only = body.get("scaffolds") or list(agents.SCAFFOLDS)
        if not isinstance(only, list) or not only:
            return self._json({"error": "scaffolds 必须是非空数组"}, 400)
        bad = _invalid_scaffolds(only)
        if bad:
            return self._json({"error": f"未知 scaffold: {bad}",
                               "valid": list(agents.SCAFFOLDS)}, 400)

        cache_key = json.dumps({"backend": backend_name, "model": model,
                                "k": k, "scaffolds": only}, sort_keys=True)
        if backend_name != "api":
            hit = _COMPARE_CACHE.get(cache_key)
            if hit is not None:
                return self._json({**hit, "cached": True})

        if backend_name == "api":
            try:
                backend = llm.OpenAICompat(model=model, temperature=0.7, seed=20261030)
            except RuntimeError as e:
                return self._json({"error": str(e)}, 400)
        else:
            backend = llm.MockDirector()

        rows = []
        t0 = time.time()
        try:
            for scaffold in only:
                records = runner.run_matrix(SCENES, [scaffold], backend, k=k,
                                            verbose=False,
                                            examples_for=runner.make_example_provider(SCENES, 2))
                lo, hi, point = metrics.pass_ci(records, scaffold)
                rows.append({
                    "scaffold": scaffold,
                    "pass_at_1": round(metrics.pass_at_1(records, scaffold), 4),
                    "pass_at_1_ci": [lo, hi],
                    "pass_pow_k": round(metrics.pass_pow_k(records, scaffold), 4),
                    "line_pass": round(metrics.line_level_pass(records, scaffold), 4),
                    "hit_rate": round(metrics.mean_hit_rate(records, scaffold), 4),
                    "l1_clean": round(metrics.l1_clean_rate(records, scaffold), 4),
                    "violation_rate": round(metrics.violation_rate(records, scaffold), 4),
                    "tokens": round(metrics.mean_tokens(records, scaffold), 1),
                    "attempts": round(metrics.mean_attempts(records, scaffold), 2),
                })
        except Exception as e:  # 不让一次失败把连接掐断（前端只会看到网络错误）
            return self._json(
                {"error": f"评测执行失败：{type(e).__name__}: {e}"}, 500)

        payload = {
            "backend": backend_name,
            "model": getattr(backend, "model", getattr(backend, "name", "-")),
            "k": k,
            "n_scenes": len(SCENES),
            "n_lines": sum(len(t["lines"]) for t in SCENES),
            "tau": judge.DEFAULT_TAU,
            "elapsed": round(time.time() - t0, 1),
            "rows": rows,
        }
        if backend_name != "api":
            _COMPARE_CACHE[cache_key] = payload
        return self._json(payload)


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    port = int(os.environ.get("MDH_PORT", "8770"))
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print("=" * 62)
    print(" MangaDirector Studio")
    print(f"   http://127.0.0.1:{port}")
    print(f"   场景 {len(SCENES)} 场 / scaffold {len(agents.SCAFFOLDS)} 个")
    print("   Ctrl+C 停止")
    print("=" * 62)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
