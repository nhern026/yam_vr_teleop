"""Live teleop dashboard — zero external dependencies.

Serves a single-page dashboard over HTTP and streams live telemetry via
Server-Sent Events (SSE). The teleop loop calls ``push()`` at ~10 Hz; every
connected browser receives the JSON snapshot immediately.

Architecture:
  - One background thread runs an ``http.server`` on ``0.0.0.0:8080``.
  - ``GET /`` serves the embedded HTML dashboard.
  - ``GET /events`` is an SSE endpoint; the browser's ``EventSource`` reconnects
    automatically on drop.
  - ``GET /demos`` returns a JSON list of saved HDF5 files in the record dir.
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

_DASHBOARD_HTML: str | None = None


def _load_html() -> str:
    global _DASHBOARD_HTML
    if _DASHBOARD_HTML is None:
        html_path = Path(__file__).with_name("dashboard.html")
        _DASHBOARD_HTML = html_path.read_text()
    return _DASHBOARD_HTML


class _Broker:
    """Fan-out: one producer, many SSE consumers."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._clients: list[queue.Queue[str]] = []

    def subscribe(self) -> queue.Queue[str]:
        q: queue.Queue[str] = queue.Queue(maxsize=4)
        with self._lock:
            self._clients.append(q)
        return q

    def unsubscribe(self, q: queue.Queue[str]) -> None:
        with self._lock:
            try:
                self._clients.remove(q)
            except ValueError:
                pass

    def publish(self, data: str) -> None:
        with self._lock:
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
    if _record_dir is None:
        return []
    demos = []
    for p in sorted(_record_dir.glob("*.hdf5"), reverse=True):
        try:
            import h5py
            with h5py.File(p, "r") as f:
                demos.append({
                    "name": p.name,
                    "ticks": int(f.attrs.get("num_ticks", 0)),
                    "duration_s": round(float(f.attrs.get("duration_s", 0)), 1),
                    "arms": list(f.attrs.get("arms", [])),
                    "start_time": str(f.attrs.get("start_time", "")),
                    "size_kb": round(p.stat().st_size / 1024, 1),
                })
        except Exception:
            demos.append({"name": p.name, "error": "could not read"})
    return demos


import socketserver


class _ReusableHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    allow_reuse_address = True
    daemon_threads = True


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        pass

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
        body = _load_html().encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_demos(self) -> None:
        body = json.dumps(_list_demos()).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def _serve_sse(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

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
        self._server = _ReusableHTTPServer(("0.0.0.0", self._port), _Handler)
        self._server.timeout = 0.5
        self._thread = threading.Thread(target=self._serve, name="dashboard", daemon=True)
        self._thread.start()
        print(f"Dashboard: http://localhost:{self._port}", flush=True)

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

    @property
    def client_count(self) -> int:
        return _broker.client_count


def _json_default(obj: Any) -> Any:
    """JSON serialiser for numpy types."""
    import numpy as np
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    raise TypeError(f"Not JSON serializable: {type(obj)}")
