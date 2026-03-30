from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional

from ui.game_events import STORE


HTML_PATH = Path(__file__).parent / "static" / "index.html"


class UIRequestHandler(BaseHTTPRequestHandler):
    def _send(self, status: int, body: bytes, content_type: str = "text/plain; charset=utf-8"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ["/", "/index.html"]:
            html = HTML_PATH.read_bytes()
            return self._send(200, html, "text/html; charset=utf-8")

        if self.path == "/api/state":
            payload = json.dumps(STORE.get_state()).encode("utf-8")
            return self._send(200, payload, "application/json; charset=utf-8")

        return self._send(404, b"Not found")

    def log_message(self, format: str, *args):
        return


_server: Optional[ThreadingHTTPServer] = None
_thread: Optional[threading.Thread] = None


def start_ui_server(host: str = "127.0.0.1", port: int = 8000):
    global _server, _thread
    if _server is not None:
        return _server

    _server = ThreadingHTTPServer((host, port), UIRequestHandler)
    _thread = threading.Thread(target=_server.serve_forever, daemon=True)
    _thread.start()
    return _server


def stop_ui_server():
    global _server, _thread
    if _server is not None:
        _server.shutdown()
        _server.server_close()
        _server = None
    if _thread is not None:
        _thread.join(timeout=1)
        _thread = None
