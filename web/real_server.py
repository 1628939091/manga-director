"""真实剧本标注页服务（端口 8773）：给 `annotations/real_form.csv` 打 arousal。

这个页面和另外两个标注页是三批完全独立的数据
--------------------------------------------------
- 8771：合成剧本主力表（170 行），第一次标注用的
- 8772：同一批合成的复测表（20 行），量自我一致性
- 8773：**真实剧本**（40 行，大话西游等真实台词），量外部效度

三批数据不能互相看见。这个服务只认 `annotations/real_form.csv` 一个文件，
不 import `mdh.annotate`（那个模块的 FORM 常量指着主力表），
也不 import 复测页的任何东西——连间接引用都不留。读错一个文件，
「真实剧本」这一批就退化成合成数据的副本，外部效度无从谈起。

只写一列
--------
`arousal(1-10)` 是唯一会被改的列，其余 9 列原样回写。
启动时逐字校验表头、行数 > 0、line_id 唯一；表不对就拒绝启动，
宁可起不来，也不要把错的行标进去。

为什么每按一键就整表重写
------------------------
40 行、写一次不到 1ms。标注最怕「标到第 30 句断线，前 29 句白标」。
先写 `.tmp` 再 `os.replace`，写一半崩了也不会留下半张表。

为什么页面不返回已有答案
------------------------
`/api/items` 只给 `done` 布尔，不给数值。重进已标过的行时若看见旧值，
人会顺着旧值走，金标就会被自己上一次的判断污染。
"""

from __future__ import annotations

import csv
import json
import os
import threading
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
REAL = os.path.join(ROOT, "annotations", "real_form.csv")

# 列序即契约：回写时按这个顺序写，不接受任何增删或换位
FIELDS = ["序号", "line_id", "出处", "场次", "说话人",
          "前一句", "台词", "后一句", "arousal(1-10)", "备注"]
COL_AROUSAL = "arousal(1-10)"


class Store:
    """real_form.csv 的内存镜像。写入即落盘，刷新页面不丢。"""

    def __init__(self, path: str = REAL) -> None:
        self.path = path
        self.lock = threading.Lock()
        self.rows: list[dict] = []
        self.idx: dict[str, int] = {}
        self.load()

    def load(self) -> None:
        if not os.path.exists(self.path):
            raise SystemExit(f"找不到 {self.path}")
        with open(self.path, encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames or []
            if header != FIELDS:
                raise SystemExit(
                    "real_form.csv 表头不对，拒绝启动。\n"
                    f"  期望：{FIELDS}\n  实际：{header}")
            for row in reader:
                self.rows.append({k: (row.get(k) or "") for k in FIELDS})
        if not self.rows:
            raise SystemExit("real_form.csv 是空表，拒绝启动")
        ids = [r["line_id"] for r in self.rows]
        if len(set(ids)) != len(ids):
            dup = sorted({x for x in ids if ids.count(x) > 1})
            raise SystemExit(f"line_id 有重复，拒绝启动：{dup}")
        self.idx = {r["line_id"]: i for i, r in enumerate(self.rows)}

    def items(self) -> list[dict]:
        """给前端的行数据。**不含 arousal 值**，只给 done 布尔。"""
        return [{
            "row": r["序号"],
            "line_id": r["line_id"],
            "source": r["出处"],
            "scene": r["场次"],
            "speaker": r["说话人"],
            "prev": r["前一句"],
            "text": r["台词"],
            "next": r["后一句"],
            "done": bool(r[COL_AROUSAL].strip()),
        } for r in self.rows]

    def save(self, line_id: str, arousal) -> dict:
        with self.lock:
            i = self.idx.get(line_id)
            if i is None:
                return {"ok": False, "error": f"未知 line_id：{line_id}"}
            try:
                v = int(str(arousal).strip())
            except (TypeError, ValueError):
                return {"ok": False, "error": f"arousal 不是整数：{arousal!r}"}
            if not 1 <= v <= 10:
                return {"ok": False, "error": f"arousal 超出 1-10：{v}"}
            self.rows[i][COL_AROUSAL] = str(v)
            self._flush()
            return {"ok": True, "filled": self.filled(), "total": len(self.rows)}

    def _flush(self) -> None:
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(self.rows)
        os.replace(tmp, self.path)      # 原子替换

    def filled(self) -> int:
        return sum(1 for r in self.rows if r[COL_AROUSAL].strip())


STORE: Store | None = None


class Handler(BaseHTTPRequestHandler):
    server_version = "MDH-Real/1.0"

    def log_message(self, fmt, *args):     # 静音：每按一键都会打一条
        pass

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

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            p = os.path.join(STATIC, "real.html")
            if not os.path.exists(p):
                return self._send(404, b"real.html missing", "text/plain")
            with open(p, "rb") as f:
                return self._send(200, f.read(), "text/html; charset=utf-8")
        if path == "/api/items":
            return self._json({
                "items": STORE.items(),
                "total": len(STORE.rows),
                "filled": STORE.filled(),
            })
        return self._send(404, b"not found", "text/plain")

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/api/save":
            body = self._body()
            return self._json(STORE.save(body.get("line_id", ""), body.get("arousal")))
        return self._send(404, b"not found", "text/plain")


def main() -> int:
    global STORE
    STORE = Store()
    port = int(os.environ.get("MDH_REAL_PORT", "8773"))
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print("=" * 56)
    print(" 真实剧本标注：一句一屏，带前后文")
    print("=" * 56)
    print(f"   已填 {STORE.filled()} / {len(STORE.rows)} 行")
    print(f"   表：{STORE.path}")
    print(f"   http://127.0.0.1:{port}")
    print("   关掉这个窗口即停止（填写内容已实时存进 real_form.csv）")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
