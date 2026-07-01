from __future__ import annotations

import http.server
import json
import socketserver
import threading
from urllib import request as urllib_request
from urllib.error import HTTPError, URLError


class ProxyStats:
    def __init__(self) -> None:
        self.api_calls = 0
        self.failed_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self._lock = threading.Lock()

    def record(self, body: bytes, content_type: str = "") -> None:
        with self._lock:
            self.api_calls += 1
            text = body.decode("utf-8", errors="replace")
            usage = (
                self._stream_usage(text)
                if "event-stream" in content_type
                else self._json_usage(text)
            )
            self.input_tokens += self._input(usage)
            self.output_tokens += self._output(usage)

    def record_failure(self) -> None:
        """Count an upstream call that errored (5xx/429/connection). Without
        this, failed calls vanish from api_calls -- undercounting exactly when
        the run is struggling. No tokens: an error body carries no usage."""
        with self._lock:
            self.api_calls += 1
            self.failed_calls += 1

    def _stream_usage(self, text: str) -> dict:
        # SSE stream: keep the last usage object seen. OpenAI emits it once on the
        # final chunk; Gemini repeats it cumulatively on every chunk, so last wins
        # for both. Chat Completions puts it on the chunk, the Responses API tucks
        # it under chunk.response, and Gemini uses chunk.usageMetadata.
        last: dict = {}
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
            usage = self._extract(chunk)
            if usage:
                last = usage
        return last

    def _json_usage(self, text: str) -> dict:
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return {}
        if isinstance(data, list):
            # Gemini's streamGenerateContent (without ?alt=sse) returns a JSON
            # array of chunks rather than an SSE stream; keep the last usage.
            last: dict = {}
            for chunk in data:
                usage = self._extract(chunk)
                if usage:
                    last = usage
            return last
        return self._extract(data)

    @staticmethod
    def _extract(obj: object) -> dict:
        if not isinstance(obj, dict):
            return {}
        return (
            obj.get("usage")
            or (obj.get("response") or {}).get("usage")
            or obj.get("usageMetadata")
            or {}
        )

    @staticmethod
    def _input(usage: dict) -> int:
        return usage.get(
            "input_tokens", usage.get("prompt_tokens", usage.get("promptTokenCount", 0))
        )

    @staticmethod
    def _output(usage: dict) -> int:
        # Direct output count across OpenAI (output_tokens / completion_tokens)
        # and Gemini (candidatesTokenCount). Gemini 2.5 "thinking" models bill
        # reasoning as output but report it separately under thoughtsTokenCount,
        # so add it in -- otherwise a run that spends most of its budget
        # thinking looks nearly free.
        direct = (
            usage.get("output_tokens")
            or usage.get("completion_tokens")
            or usage.get("candidatesTokenCount")
            or 0
        )
        direct += usage.get("thoughtsTokenCount", 0)
        if direct:
            return direct
        # Fallback: some Gemini streaming responses leave candidatesTokenCount
        # off the final usageMetadata, carrying only prompt+total -- derive
        # output as total-minus-input so it never silently reads zero.
        total = usage.get("total_tokens") or usage.get("totalTokenCount") or 0
        return max(0, total - ProxyStats._input(usage)) if total else 0


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
                # 4xx/5xx from upstream: still a call the tool made -- count it,
                # then relay the error body so the tool sees the real status.
                stats.record_failure()
                self._respond(exc.code, "application/json", exc.read())
            except URLError as exc:
                # Connection reset / DNS / timeout: never reached upstream, but
                # it's still an attempt. Count it and surface a 502 rather than
                # letting the handler thread crash and hang the tool's request.
                stats.record_failure()
                body = json.dumps({"error": {"message": str(exc.reason)}}).encode("utf-8")
                self._respond(502, "application/json", body)

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
