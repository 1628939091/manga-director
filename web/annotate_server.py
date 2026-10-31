"""标注页服务：一句一屏，键盘操作，写完直接算 κ（端口 8771）。

为什么单独起一个服务，而不是塞进 web/server.py
----------------------------------------------
那个页面是「给人看结果的」，这个页面是「给人干活的」。
干活的页面最怕两件事：误触改了别的状态、和看结果的页面抢同一个端口。
所以分开：8770 看结果，8771 填标注，互不影响。

为什么标注结果直接写回 form.csv
------------------------------
`python annotate.py score` 认的就是 `annotations/form.csv`。
如果这里另存一个 answers.jsonl，就会出现「填完了但 score 说没填」的错位。
所以每按一次键就整表重写一遍——166 行、每行几十字节，写一次不到 1ms，
用不着为了省这点 IO 引入第二个数据源。
"""

from __future__ import annotations

import contextlib
import csv
import io
import json
import os
import sys
import threading
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from mdh import annotate  # noqa: E402

STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
FORM = annotate.FORM
FIELDS = ["序号", "scene_id", "line_id", "作品", "场景", "场景类型",
          "角色", "听者", "关系", "台词", "潜台词",
          "arousal(1-10)", "valence(-1~1)", "intent", "备注"]


class Store:
    """form.csv 的内存镜像。写入即落盘，刷新页面不丢。"""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.rows: list[dict] = []
        self.idx: dict[str, int] = {}
        self.load()

    def load(self) -> None:
        if not os.path.exists(FORM):
            raise SystemExit(f"找不到 {FORM}，先跑 python annotate.py export")
        with open(FORM, encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                self.rows.append({k: (row.get(k) or "") for k in FIELDS})
        self.idx = {r["line_id"]: i for i, r in enumerate(self.rows)}

    def items(self) -> list[dict]:
        return [{
            "row": r["序号"], "line_id": r["line_id"], "scene_id": r["scene_id"],
            "scene": r["场景"], "scene_type": r["场景类型"],
            "character": r["角色"], "listener": r["听者"], "relation": r["关系"],
            "text": r["台词"], "subtext": r["潜台词"],
            "arousal": r["arousal(1-10)"], "valence": r["valence(-1~1)"],
            "intent": r["intent"], "note": r["备注"],
        } for r in self.rows]

    def save(self, line_id: str, arousal, valence, intent, note=None) -> dict:
        with self.lock:
            i = self.idx.get(line_id)
            if i is None:
                return {"ok": False, "error": f"未知 line_id：{line_id}"}
            r = self.rows[i]
            # 空字符串 = 明确清空（跳过/反悔），不能被当成「不修改」
            r["arousal(1-10)"] = "" if arousal in (None, "") else str(arousal)
            r["valence(-1~1)"] = "" if valence in (None, "") else str(valence)
            r["intent"] = "" if intent in (None, "") else str(intent)
            if note is not None:
                r["备注"] = str(note)
            self._flush()
            return {"ok": True, "filled": self.filled()}

    def _flush(self) -> None:
        tmp = FORM + ".tmp"
        with open(tmp, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(self.rows)
        os.replace(tmp, FORM)     # 原子替换：写一半断电不会毁掉整张表

    def filled(self) -> int:
        n = 0
        for r in self.rows:
            if r["arousal(1-10)"] or r["valence(-1~1)"] or r["intent"]:
                n += 1
        return n


STORE: Store | None = None


class Handler(BaseHTTPRequestHandler):
    server_version = "MDH-Annotate/1.0"

    def log_message(self, fmt, *args):    # 静音：每按一次键都会打一条，太吵
        pass

    # ---------------- 基础 ----------------
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

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        try:
            return json.loads(self.rfile.read(n).decode("utf-8"))
        except json.JSONDecodeError:
            return {}

    # ---------------- 路由 ----------------
    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            p = os.path.join(STATIC, "annotate.html")
            if not os.path.exists(p):
                return self._send(404, b"annotate.html missing", "text/plain")
            with open(p, "rb") as f:
                return self._send(200, f.read(), "text/html; charset=utf-8")
        if path == "/api/items":
            return self._json({
                "items": STORE.items(),
                "intents": annotate.INTENTS,
                "filled": STORE.filled(),
            })
        return self._send(404, b"not found", "text/plain")

    def do_POST(self):
        path = urlparse(self.path).path
        body = self._body()
        if path == "/api/save":
            return self._json(STORE.save(
                body.get("line_id", ""),
                body.get("arousal"),
                body.get("valence"),
                body.get("intent"),
                body.get("note"),
            ))
        if path == "/api/score":
            return self._json(_run_score())
        return self._send(404, b"not found", "text/plain")


def _run_score() -> dict:
    """跑一次 score，把报告原文回给页面。

    不在这里重新实现统计：κ 的算法只应该有一份（mdh/annotate.py），
    页面只是显示器。两边各写一套，迟早对不上。
    """
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            annotate.cmd_score()
    except SystemExit as e:
        return {"ok": False, "error": str(e)}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    md = ""
    if os.path.exists(annotate.OUT_MD):
        with open(annotate.OUT_MD, encoding="utf-8") as f:
            md = f.read()
    return {"ok": True, "log": buf.getvalue(), "md": md,
            "path": annotate.OUT_MD}


def main() -> int:
    global STORE
    STORE = Store()
    port = int(os.environ.get("MDH_ANNO_PORT", "8771"))
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print("=" * 56)
    print(" 标注台：一句一屏，键盘操作")
    print("=" * 56)
    print(f"   已填 {STORE.filled()} / {len(STORE.rows)} 行")
    print(f"   http://127.0.0.1:{port}")
    print("   关掉这个窗口即停止（填写内容已实时存进 form.csv）")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
