"""
range_server.py  -  a tiny local web server that understands HTTP range requests (like the public server of the
expert benchmark). Used only by the smoke test, to test tools/get_benchmark.py without the internet.
"""
import http.server
import os
import re
import threading


class _Handler(http.server.BaseHTTPRequestHandler):
    root = "."

    def log_message(self, *args):   # keep the smoke test output clean
        pass

    def do_GET(self):
        rel = self.path.split("?", 1)[0].lstrip("/")
        path = os.path.normpath(os.path.join(self.root, *rel.split("/")))
        if not path.startswith(os.path.normpath(self.root)) or not os.path.isfile(path):
            self.send_error(404, "Not Found")
            return
        size = os.path.getsize(path)
        m = re.match(r"bytes=(\d+)-(\d*)$", self.headers.get("Range", ""))
        if m:
            start = int(m.group(1))
            end = min(int(m.group(2)) if m.group(2) else size - 1, size - 1)
            if start > end:
                self.send_error(416, "Requested Range Not Satisfiable")
                return
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            start, end = 0, size - 1
            self.send_response(200)
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Content-Type", "application/octet-stream")
        self.end_headers()
        with open(path, "rb") as fh:
            fh.seek(start)
            left = end - start + 1
            while left > 0:
                chunk = fh.read(min(1 << 20, left))
                if not chunk:
                    break
                self.wfile.write(chunk)
                left -= len(chunk)


def start(root):
    """Serve the folder `root` on 127.0.0.1 (a free port). Returns (server, base url); server.shutdown() stops it."""
    handler = type("Handler", (_Handler,), {"root": os.path.abspath(root)})
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}/"
