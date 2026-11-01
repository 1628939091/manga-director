"""demo 服务（端口 8774）。

两个视图：
  /pair          参数对比（外放 vs 内敛，看的是参数条，不是耳朵）
  /demo.html     音频对比（naive vs calibrated，声音差异很细微）

为什么参数视图是主视图：
    模型的音高公式 f0_offset = 0.45 × (intensity − 4)，
    每 1 分强度只换 0.45 个半音。强度从 8.0 压到 3.7，音高差约 2 个半音（12%），
    而真实喊叫是 8–12 个半音。参数层有效、声学层不够——所以让人「看参数」而不是「听」。

只读 outputs/ 下的产出文件，不 import mdh、不碰 annotations。零第三方依赖。
启动：python web/demo_server.py  →  http://127.0.0.1:8774
"""

from __future__ import annotations

import http.server
import json
import os
import socketserver

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIO = os.path.join(ROOT, "outputs", "audio")
AUDIO_EDGE = os.path.join(ROOT, "outputs", "audio_edge")
AUDIO_PAIR = os.path.join(ROOT, "outputs", "audio_pair")
STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
PORT = int(os.environ.get("MDH_DEMO_PORT", "8774"))


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=STATIC, **kw)

    def log_message(self, fmt, *args):
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json_file(self, path: str) -> None:
        if not os.path.isfile(path):
            self._send(500, b'{"error":"missing file"}',
                       "application/json; charset=utf-8")
            return
        self._send(200, open(path, "rb").read(), "application/json; charset=utf-8")

    def do_GET(self) -> None:
        path = self.path.replace("\\", "/").split("?")[0].split("#")[0]

        if path == "/api/pair":
            return self._json_file(os.path.join(AUDIO_PAIR, "manifest.json"))
        if path == "/api/data":
            return self._json_file(os.path.join(AUDIO, "manifest.json"))
        if path == "/api/oai":
            return self._json_file(os.path.join(AUDIO, "oai_table.json"))

        if "/pair_audio/" in path:
            return self._serve_file(AUDIO_PAIR, path.rsplit("/", 1)[-1])
        if "/audio_edge/" in path:
            return self._serve_file(AUDIO_EDGE, path.rsplit("/", 1)[-1])
        if "/audio/" in path:
            return self._serve_file(AUDIO, path.rsplit("/", 1)[-1])

        if path in ("/", "/index.html", "/pair"):
            return self._serve_html("pair.html")
        if path in ("/demo", "/demo.html"):
            return self._serve_html("demo.html")
        super().do_GET()

    def _serve_html(self, name: str) -> None:
        p = os.path.join(STATIC, name)
        if not os.path.isfile(p):
            self._send(404, b"not found", "text/plain")
            return
        self._send(200, open(p, "rb").read(), "text/html; charset=utf-8")

    def _serve_file(self, base: str, name: str) -> None:
        path = os.path.normpath(os.path.join(base, os.path.basename(name)))
        if not path.startswith(base) or not os.path.isfile(path):
            self._send(404, b"not found", "text/plain")
            return
        self._send(200, open(path, "rb").read(), "audio/wav")


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


if __name__ == "__main__":
    with Server(("127.0.0.1", PORT), Handler) as httpd:
        print(f"参数对比: http://127.0.0.1:{PORT}/pair")
        print(f"音频对比: http://127.0.0.1:{PORT}/demo.html")
        httpd.serve_forever()
