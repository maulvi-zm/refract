from __future__ import annotations

import http.server
import json
import os
import socketserver
import threading
import time
from pathlib import Path
from urllib import request as urllib_request
from urllib.error import HTTPError, URLError

# Server-side backoff so a 429 burst is absorbed here, not by the tool's own
# (much smaller) retry budget.
_MAX_RETRIES = 6
_BACKOFF_BASE = 2.0  # seconds; grows 2,4,8,... per attempt unless Retry-After given
_BACKOFF_MAX = 60.0  # cap per sleep


class ProxyStats:
    def __init__(self, log_path: Path | None = None) -> None:
        self.api_calls = 0
        self.failed_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self._lock = threading.Lock()
        # Optional per-call audit trail (JSONL): one line per forwarded call with
        # the upstream's own token counts, so the totals above can be re-derived
        # by summing the lines. Truncated here so each run starts clean.
        self._log_path = log_path
        if log_path is not None:
            log_path.write_text("")
        # Opt-in capture of the *request* side. The token log above shows input
        # tokens climbing call after call, but not why; the request body carries
        # the actual conversation the tool replays each turn, which is the thing
        # that grows. Off by default: these files are large and hold source code.
        self._req_path: Path | None = None
        if log_path is not None and os.environ.get("REFRACT_PROXY_CAPTURE_BODIES"):
            self._req_path = log_path.with_name(
                log_path.name.replace("_proxy.jsonl", "_requests.jsonl")
            )
            self._req_path.write_text("")

    @staticmethod
    def _redact(path: str) -> str:
        """Strip secrets from a URL path before it hits the on-disk log: some
        providers authenticate via a query param (Gemini's ?key=) instead of a
        header, so the raw path can carry a live API key."""
        if "?" not in path:
            return path
        base, _, query = path.partition("?")
        secret = {"key", "apikey", "api_key", "access_token", "token"}
        parts = []
        for pair in query.split("&"):
            name, sep, _val = pair.partition("=")
            parts.append(f"{name}{sep}***REDACTED***" if sep and name.lower() in secret else pair)
        return f"{base}?{'&'.join(parts)}"

    def _log(self, record: dict) -> None:
        if self._log_path is None:
            return
        if "path" in record:
            record = {**record, "path": self._redact(record["path"])}
        try:
            with self._log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"ts": round(time.time(), 3), **record}) + "\n")
        except OSError:
            pass  # never let audit logging break a run

    def record(self, body: bytes, content_type: str = "", path: str = "") -> None:
        with self._lock:
            self.api_calls += 1
            text = body.decode("utf-8", errors="replace")
            usage = (
                self._stream_usage(text)
                if "event-stream" in content_type
                else self._json_usage(text)
            )
            inp, out = self._input(usage), self._output(usage)
            self.input_tokens += inp
            self.output_tokens += out
            # Log derived input/output next to the raw upstream fields, so the
            # per-call arithmetic can be checked.
            self._log(
                {
                    "call": self.api_calls,
                    "ok": True,
                    "path": path,
                    "input_tokens": inp,
                    "output_tokens": out,
                    "promptTokenCount": usage.get("promptTokenCount")
                    or usage.get("prompt_tokens")
                    or usage.get("input_tokens"),
                    "candidatesTokenCount": usage.get("candidatesTokenCount")
                    or usage.get("completion_tokens")
                    or usage.get("output_tokens"),
                    "thoughtsTokenCount": usage.get("thoughtsTokenCount"),
                    "totalTokenCount": usage.get("totalTokenCount")
                    or usage.get("total_tokens"),
                }
            )

    def record_failure(self, path: str = "", status: int | None = None) -> None:
        """Count an upstream call that errored (5xx/429/connection). Without
        this, failed calls vanish from api_calls -- undercounting exactly when
        the run is struggling. No tokens: an error body carries no usage."""
        with self._lock:
            self.api_calls += 1
            self.failed_calls += 1
            self._log({"call": self.api_calls, "ok": False, "path": path, "status": status})

    def record_request(self, body: bytes, path: str = "") -> None:
        """Log the outgoing conversation for one call, when body capture is on.

        Written before the response comes back, so this call's number is one
        past api_calls. Keeps the whole body plus a per-turn digest, so the
        transcript can be read directly and the growth can be charted without
        re-parsing megabytes of JSON."""
        if self._req_path is None:
            return
        text = body.decode("utf-8", errors="replace")
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = None
        turns: list[dict] = []
        if isinstance(data, dict):
            # OpenAI Chat Completions uses "messages", the Responses API uses
            # "input", Gemini uses "contents" -- all are the replayed history.
            raw = data.get("messages") or data.get("input") or data.get("contents") or []
            if isinstance(raw, list):
                for turn in raw:
                    if not isinstance(turn, dict):
                        continue
                    turns.append(
                        {
                            "role": turn.get("role"),
                            "chars": len(json.dumps(turn, ensure_ascii=False)),
                        }
                    )
        with self._lock:
            record = {
                "call": self.api_calls + 1,
                "path": self._redact(path),
                "body_bytes": len(body),
                "turns": len(turns),
                "turn_digest": turns,
                "body": data if data is not None else text,
            }
        try:
            with self._req_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            pass  # never let audit logging break a run

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
            stats.record_request(body, self.path)

            target_url = upstream.rstrip("/") + self.path
            fwd_headers = {k: v for k, v in self.headers.items() if k.lower() != "host"}
            req = urllib_request.Request(target_url, data=body, headers=fwd_headers, method=method)

            # Retry 429/503 here so the tool doesn't burn its own retry budget on
            # a rate-limit burst. Every attempt is counted as a real call; a 429
            # carries no tokens, so cost accounting stays honest.
            for attempt in range(_MAX_RETRIES + 1):
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

                        stats.record(resp_body, content_type, self.path)
                    return
                except HTTPError as exc:
                    # 4xx/5xx from upstream: still a call the tool made -- count it.
                    stats.record_failure(self.path, exc.code)
                    if exc.code in (429, 503) and attempt < _MAX_RETRIES:
                        retry_after = exc.headers.get("Retry-After") if exc.headers else None
                        try:
                            delay = float(retry_after) if retry_after else _BACKOFF_BASE * (2**attempt)
                        except (TypeError, ValueError):
                            delay = _BACKOFF_BASE * (2**attempt)
                        time.sleep(min(delay, _BACKOFF_MAX))
                        continue
                    # Non-retryable, or retries exhausted: relay the real status.
                    self._respond(exc.code, "application/json", exc.read())
                    return
                except URLError as exc:
                    # Connection reset / DNS / timeout: never reached upstream, but
                    # it's still an attempt. Count it and surface a 502 rather than
                    # letting the handler thread crash and hang the tool's request.
                    stats.record_failure(self.path, 502)
                    body = json.dumps({"error": {"message": str(exc.reason)}}).encode("utf-8")
                    self._respond(502, "application/json", body)
                    return

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

    def __init__(self, upstream: str = "https://api.openai.com", log_path: Path | None = None) -> None:
        self.stats = ProxyStats(log_path)
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
