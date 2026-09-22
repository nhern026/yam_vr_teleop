"""Live teleop dashboard — zero external dependencies.

Serves a single-page dashboard over HTTP and streams live telemetry via
Server-Sent Events (SSE). The teleop loop calls ``push()`` at ~10 Hz; every
connected browser receives the JSON snapshot immediately.

Architecture:
  - One background thread runs an ``http.server`` on ``0.0.0.0:8080``.
  - ``GET /`` serves the embedded HTML dashboard.
  - ``GET /events`` is an SSE endpoint; the browser's ``EventSource`` reconnects
    automatically on drop.
  - ``GET /demos`` returns a JSON list of raw episode directories in the record dir.
  - ``push(snapshot_dict)`` is called from the teleop thread; it serialises to
    JSON once and fans out to every waiting SSE connection via a threading.Event
    per client.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

_DASHBOARD_HTML: bytes | None = None


def _load_html() -> bytes:
    global _DASHBOARD_HTML
    if _DASHBOARD_HTML is None:
        html_path = Path(__file__).with_name("dashboard.html")
        _DASHBOARD_HTML = html_path.read_bytes()
    return _DASHBOARD_HTML


class _Broker:
    """Fan-out: one producer, many SSE consumers."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._clients: list[queue.Queue[str]] = []
        self._last_snapshot: str | None = None

    def subscribe(self) -> queue.Queue[str]:
        q: queue.Queue[str] = queue.Queue(maxsize=4)
        with self._lock:
            self._clients.append(q)
            if self._last_snapshot is not None:
                try:
                    q.put_nowait(self._last_snapshot)
                except queue.Full:
                    pass
        return q

    def unsubscribe(self, q: queue.Queue[str]) -> None:
        with self._lock:
            try:
                self._clients.remove(q)
            except ValueError:
                pass

    def publish(self, data: str) -> None:
        with self._lock:
            self._last_snapshot = data
            dead: list[queue.Queue[str]] = []
            for q in self._clients:
                try:
                    # Drop old frames rather than blocking the producer.
                    while not q.empty():
                        try:
                            q.get_nowait()
                        except queue.Empty:
                            break
                    q.put_nowait(data)
                except queue.Full:
                    dead.append(q)
            for q in dead:
                self._clients.remove(q)

    @property
    def client_count(self) -> int:
        with self._lock:
            return len(self._clients)


_broker = _Broker()
_record_dir: Path | None = None


def _list_demos() -> list[dict[str, Any]]:
    if _record_dir is None or not _record_dir.is_dir():
        return []
    demos = []
    paths = sorted((p for p in _record_dir.iterdir()
                    if p.is_dir() and (p / "metadata.json").is_file()),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    for p in paths:
        try:
            metadata = json.loads((p / "metadata.json").read_bytes())
            counts = metadata.get("counts", {})
            configs = metadata.get("configs", [])
            demos.append({
                "name": p.name,
                "ticks": int(counts.get("valid", 0)),
                "duration_s": round(float(metadata.get("duration_s", 0)), 1),
                "arms": [cfg.get("teleop", {}).get("hand", "?") for cfg in configs if cfg],
                "task": str(metadata.get("task", "")),
                "status": str(metadata.get("status", "unknown")),
            })
        except Exception:
            demos.append({"name": p.name, "error": "could not read"})
    return demos


import socket
import socketserver


class _ReusableHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, server_address: tuple[str, int], RequestHandlerClass: type) -> None:
        if server_address[0] == "::":
            self.address_family = socket.AF_INET6
        else:
            self.address_family = socket.AF_INET
        super().__init__(server_address, RequestHandlerClass)

    def server_bind(self) -> None:
        if self.address_family == socket.AF_INET6:
            try:
                self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
            except Exception:
                pass
        super().server_bind()


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:
        pass

    def do_HEAD(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.flush()

    def do_GET(self) -> None:
        if self.path == "/":
            self._serve_html()
        elif self.path == "/events":
            self._serve_sse()
        elif self.path == "/demos":
            self._serve_demos()
        else:
            self.send_error(404)

    def _serve_html(self) -> None:
        body = _load_html()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()

    def _serve_demos(self) -> None:
        body = json.dumps(_list_demos()).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()

    def _serve_sse(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.flush()

        q = _broker.subscribe()
        try:
            while True:
                try:
                    data = q.get(timeout=15.0)
                except queue.Empty:
                    # Keep-alive comment so the browser doesn't time out.
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
                    continue
                self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            _broker.unsubscribe(q)


class Dashboard:
    """Manages the dashboard HTTP server lifecycle."""

    def __init__(self, port: int = 8080, record_dir: Path | None = None):
        global _record_dir
        _record_dir = record_dir
        self._port = port
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        try:
            self._server = _ReusableHTTPServer(("::", self._port), _Handler)
        except Exception:
            self._server = _ReusableHTTPServer(("0.0.0.0", self._port), _Handler)
        self._server.timeout = 0.5
        self._thread = threading.Thread(target=self._serve, name="dashboard", daemon=True)
        self._thread.start()
        import socket
        try:
            ip = socket.gethostbyname(socket.gethostname())
        except Exception:
            ip = "JETSON_IP"
        print(f"Dashboard: http://localhost:{self._server.server_port} (or http://{ip}:{self._server.server_port} from external browser)", flush=True)

    def _serve(self) -> None:
        assert self._server is not None
        self._server.serve_forever()

    def push(self, snapshot: dict[str, Any]) -> None:
        """Push a state snapshot to all connected browsers. Call from any thread."""
        try:
            _broker.publish(json.dumps(snapshot, default=_json_default))
        except Exception:
            pass

    def close(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    @property
    def client_count(self) -> int:
        return _broker.client_count


def _json_default(obj: Any) -> Any:
    """JSON serialiser for numpy and non-standard types."""
    import numpy as np
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (set, tuple)):
        return list(obj)
    return str(obj)
