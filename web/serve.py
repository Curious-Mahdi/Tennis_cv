"""
Local server for the Tennis Vision demo.

    python web/serve.py            # then open http://localhost:8000

Why not `python -m http.server`: on the development machine it intermittently truncated
responses (a 88 KB JSON arrived as 65 KB despite a correct Content-Length), which breaks
the page. This server reads each file fully and sends it with one sendall(), and answers
HTTP Range requests so <video> seeking works.
"""
from __future__ import annotations

import mimetypes
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parent
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("video/mp4", ".mp4")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_HEAD(self):
        self._serve(head=True)

    def do_GET(self):
        self._serve(head=False)

    def _serve(self, head: bool) -> None:
        rel = unquote(urlparse(self.path).path).lstrip("/") or "index.html"
        path = (ROOT / rel).resolve()
        if ROOT not in path.parents and path != ROOT or not path.is_file() or path.name == "serve.py":
            self.send_error(404)
            return
        data = path.read_bytes()
        size, start, end, status = len(data), 0, len(data) - 1, 200
        m = re.match(r"bytes=(\d*)-(\d*)$", self.headers.get("Range", ""))
        if m and (m.group(1) or m.group(2)):
            if m.group(1):
                start = int(m.group(1))
                end = int(m.group(2)) if m.group(2) else size - 1
            else:
                start = max(0, size - int(m.group(2)))
            end = min(end, size - 1)
            if start > end:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            status = 206
        body = data[start:end + 1]
        self.send_response(status)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-store")
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if not head:
            self.wfile.write(body)
            self.wfile.flush()

    def log_message(self, fmt, *args):
        pass


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Tennis Vision demo: http://localhost:{port}   (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
