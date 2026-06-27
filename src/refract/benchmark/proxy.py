from __future__ import annotations

import http.server
import json
import socketserver
import threading
from urllib import request as urllib_request
from urllib.error import HTTPError


class ProxyStats:
    def __init__(self) -> None:
        self.api_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self._lock = threading.Lock()

    def record(self, body: bytes, content_type: str = "") -> None:
        with self._lock:
            self.api_calls += 1
            text = body.decode("utf-8", errors="replace")
            if "event-stream" in content_type:
                self._record_stream(text)
            else:
                self._record_json(text)

    def _record_stream(self, text: str) -> None:
        # SSE stream: hunt the data lines for a usage object. Chat Completions
        # keeps it on the chunk, the Responses API tucks it under chunk.response.
        for line in text.splitlines():
            if not line.startswith("data:"):
                continue
            raw = line[5:].strip()
            if raw == "[DONE]":
                continue
            try:
                chunk = json.loads(raw)
            except json.JSONDecodeError:
                continue
            usage = chunk.get("usage") or (chunk.get("response") or {}).get("usage") or {}
            self._add_usage(usage)

    def _record_json(self, text: str) -> None:
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return
        self._add_usage(data.get("usage") or {})

    def _add_usage(self, usage: dict) -> None:
        self.input_tokens += usage.get("input_tokens", usage.get("prompt_tokens", 0))
        self.output_tokens += usage.get("output_tokens", usage.get("completion_tokens", 0))


class _ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


# canned model list so codex's GET /models check passes without a real call
_MODELS_RESPONSE = json.dumps(
    {
        "models": [
            {"id": "gpt-4o", "object": "model"},
            {"id": "gpt-4o-mini", "object": "model"},
            {"id": "o1", "object": "model"},
            {"id": "o3-mini", "object": "model"},
            {"id": "o4-mini", "object": "model"},
        ]
    }
).encode("utf-8")


def _make_handler(stats: ProxyStats, upstream: str) -> type:
    class _Handler(http.server.BaseHTTPRequestHandler):
        def _forward(self, method: str) -> None:
            if method == "GET" and self.path.rstrip("/").endswith("/models"):
                self._respond(200, "application/json", _MODELS_RESPONSE)
                return

            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length) if length else b""

            target_url = upstream.rstrip("/") + self.path
            fwd_headers = {k: v for k, v in self.headers.items() if k.lower() != "host"}
            req = urllib_request.Request(target_url, data=body, headers=fwd_headers, method=method)

            try:
                with urllib_request.urlopen(req, timeout=180) as resp:
                    resp_body = resp.read()
                    content_type = resp.headers.get("Content-Type", "")

                    self.send_response(resp.status)
                    for key, value in resp.getheaders():
                        if key.lower() != "transfer-encoding":
                            self.send_header(key, value)
                    self.send_header("Content-Length", str(len(resp_body)))
                    self.end_headers()
                    self.wfile.write(resp_body)

                    stats.record(resp_body, content_type)
            except HTTPError as exc:
                self._respond(exc.code, "application/json", exc.read())

        def _respond(self, status: int, content_type: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:  # noqa: N802
            self._forward("POST")

        def do_GET(self) -> None:  # noqa: N802
            self._forward("GET")

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            pass  # quiet down the per-request logging

    return _Handler


class CountingProxy:
    """Counts API calls/tokens, then forwards to upstream."""

    def __init__(self, upstream: str = "https://api.openai.com") -> None:
        self.stats = ProxyStats()
        self._server = _ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(self.stats, upstream))
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def port(self) -> int:
        return self._server.server_address[1]  # type: ignore[index]

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._server.shutdown()
