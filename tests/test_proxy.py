import json

from refract.benchmark.proxy import ProxyStats


def _sse(*chunks: dict) -> bytes:
    lines = [f"data: {json.dumps(c)}" for c in chunks]
    lines.append("data: [DONE]")
    return ("\n\n".join(lines)).encode("utf-8")


def test_openai_chat_json_usage() -> None:
    stats = ProxyStats()
    body = json.dumps({"usage": {"prompt_tokens": 12, "completion_tokens": 5}}).encode()
    stats.record(body, "application/json")
    assert stats.api_calls == 1
    assert stats.input_tokens == 12
    assert stats.output_tokens == 5


def test_openai_chat_sse_last_chunk_usage() -> None:
    stats = ProxyStats()
    body = _sse(
        {"choices": [{"delta": {"content": "hi"}}]},
        {"choices": [{"delta": {}}], "usage": {"prompt_tokens": 8, "completion_tokens": 4}},
    )
    stats.record(body, "text/event-stream")
    assert stats.input_tokens == 8
    assert stats.output_tokens == 4


def test_responses_api_nested_usage() -> None:
    stats = ProxyStats()
    body = json.dumps({"response": {"usage": {"input_tokens": 7, "output_tokens": 3}}}).encode()
    stats.record(body, "application/json")
    assert stats.input_tokens == 7
    assert stats.output_tokens == 3


def test_gemini_sse_usage_metadata() -> None:
    stats = ProxyStats()
    body = _sse(
        {"usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 3, "totalTokenCount": 13}},
        {"usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 6, "totalTokenCount": 16}},
    )
    stats.record(body, "text/event-stream")
    assert stats.input_tokens == 10
    assert stats.output_tokens == 6  # last (cumulative) chunk wins


def test_gemini_thinking_tokens_count_as_output() -> None:
    stats = ProxyStats()
    body = json.dumps(
        {
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 5,
                "thoughtsTokenCount": 20,
                "totalTokenCount": 35,
            }
        }
    ).encode()
    stats.record(body, "application/json")
    assert stats.output_tokens == 25  # candidates + thoughts, not just candidates


def test_gemini_output_falls_back_to_total_minus_prompt() -> None:
    # final usageMetadata carrying no candidatesTokenCount -- the exact shape
    # that used to make output read 0
    stats = ProxyStats()
    body = json.dumps(
        {"usageMetadata": {"promptTokenCount": 10, "totalTokenCount": 30}}
    ).encode()
    stats.record(body, "application/json")
    assert stats.output_tokens == 20


def test_gemini_json_array_stream() -> None:
    stats = ProxyStats()
    body = json.dumps(
        [
            {"usageMetadata": {"promptTokenCount": 9, "candidatesTokenCount": 2, "totalTokenCount": 11}},
            {"usageMetadata": {"promptTokenCount": 9, "candidatesTokenCount": 7, "totalTokenCount": 16}},
        ]
    ).encode()
    stats.record(body, "application/json")
    assert stats.input_tokens == 9
    assert stats.output_tokens == 7


def test_record_failure_counts_call_without_tokens() -> None:
    stats = ProxyStats()
    stats.record(json.dumps({"usage": {"prompt_tokens": 5, "completion_tokens": 2}}).encode())
    stats.record_failure()
    stats.record_failure()
    assert stats.api_calls == 3  # 1 ok + 2 failed
    assert stats.failed_calls == 2
    assert stats.input_tokens == 5  # failures add no tokens
    assert stats.output_tokens == 2


def test_malformed_body_is_ignored_not_fatal() -> None:
    stats = ProxyStats()
    stats.record(b"not json at all", "application/json")
    assert stats.api_calls == 1
    assert stats.input_tokens == 0
    assert stats.output_tokens == 0
