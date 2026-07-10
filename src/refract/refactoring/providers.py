from __future__ import annotations

import json
import os
import time
from typing import Any
from urllib import request
from urllib.error import HTTPError, URLError

from refract.refactoring.proposal import ProviderConfig, ProviderName, RefactorProposal

# Transient upstream failures worth retrying: rate limiting and server-side
# overload. A single 503 shouldn't permanently skip a target -- observed on a
# live call where the identical request failed once then succeeded immediately
# on retry with no changes at all, confirming it's provider-side, not us.
_RETRYABLE_STATUSES = {429, 500, 502, 503, 504}
_MAX_ATTEMPTS = 3
_BACKOFF_SECONDS = 1.0
# Reasoning models (e.g. deepseek-v4-pro) can spend well over a minute thinking
# before the first token; a 60s ceiling was cutting otherwise-clean fixes off
# mid-flight. Generous enough to let them finish, still bounded so a truly hung
# call doesn't wait forever.
_TIMEOUT_SECONDS = 240

DEFAULT_MODELS: dict[ProviderName, str] = {
    ProviderName.OPENAI: "gpt-4.1",
    ProviderName.GEMINI: "gemini-2.5-pro",
}

_ENV_KEYS: dict[ProviderName, str] = {
    ProviderName.OPENAI: "OPENAI_API_KEY",
    ProviderName.GEMINI: "GEMINI_API_KEY",
}


def config_from_env(provider: str | None, model: str | None = None) -> ProviderConfig:
    name = ProviderName(provider or os.getenv("REFRACT_PROVIDER", ProviderName.OPENAI.value))
    model_name = model or os.getenv("REFRACT_MODEL") or DEFAULT_MODELS[name]
    return ProviderConfig(
        provider=name,
        model=model_name,
        api_key=os.getenv(_ENV_KEYS[name]),
    )


def provider_from_config(config: ProviderConfig) -> HttpJsonProvider:
    if not config.api_key:
        raise ValueError(
            f"Missing API key for provider {config.provider.value} "
            f"(set {_ENV_KEYS[config.provider]})"
        )

    providers: dict[ProviderName, type[HttpJsonProvider]] = {
        ProviderName.OPENAI: OpenAIProvider,
        ProviderName.GEMINI: GeminiProvider,
    }
    return providers[config.provider](config)


class HttpJsonProvider:
    name: ProviderName

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config

    def propose(self, system_prompt: str, user_prompt: str) -> RefactorProposal:
        payload = self._payload(system_prompt, user_prompt)
        response = _post_json(self._url(), self._headers(), payload)
        return RefactorProposal.from_json(self._extract_json(response))

    def _url(self) -> str:
        raise NotImplementedError

    def _headers(self) -> dict[str, str]:
        return {"Content-Type": "application/json"}

    def _payload(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        raise NotImplementedError

    def _extract_json(self, response: dict[str, Any]) -> str | dict[str, Any]:
        raise NotImplementedError


class OpenAIProvider(HttpJsonProvider):
    name = ProviderName.OPENAI

    def _url(self) -> str:
        base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com").rstrip("/")
        return f"{base}/v1/responses"

    def _headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}",
        }

    def _payload(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        return {
            "model": self.config.model,
            "input": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "refactor_proposal",
                    "schema": _proposal_schema(),
                    "strict": True,
                }
            },
        }

    def _extract_json(self, response: dict[str, Any]) -> str | dict[str, Any]:
        if "output_text" in response:
            return response["output_text"]
        for item in response.get("output", []):
            for content in item.get("content", []):
                if content.get("type") in {"output_text", "text"}:
                    return content.get("text", "")
        raise ValueError("OpenAI response did not contain output text")


class GeminiProvider(HttpJsonProvider):
    name = ProviderName.GEMINI

    def _url(self) -> str:
        base = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com").rstrip("/")
        return f"{base}/v1beta/models/{self.config.model}:generateContent?key={self.config.api_key}"

    def _payload(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        return {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": _proposal_schema_gemini(),
            },
        }

    def _extract_json(self, response: dict[str, Any]) -> str | dict[str, Any]:
        candidates = response.get("candidates", [])
        if not candidates:
            raise ValueError("Gemini response did not contain candidates")
        parts = candidates[0].get("content", {}).get("parts", [])
        if not parts:
            raise ValueError("Gemini response did not contain text parts")
        return parts[0].get("text", "")


def _post_json(url: str, headers: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    for attempt in range(_MAX_ATTEMPTS):
        req = request.Request(url, data=body, headers=headers, method="POST")
        try:
            with request.urlopen(req, timeout=_TIMEOUT_SECONDS) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            if exc.code not in _RETRYABLE_STATUSES or attempt == _MAX_ATTEMPTS - 1:
                raise
            time.sleep(_BACKOFF_SECONDS * (2**attempt))
        except (TimeoutError, URLError) as exc:
            # A socket timeout or dropped connection isn't an HTTP status, so the
            # branch above never sees it -- yet it's the same kind of transient,
            # retryable failure (a slow reasoning model, a flaky network) and
            # shouldn't permanently skip the target. Retry with backoff; only the
            # final attempt propagates.
            if attempt == _MAX_ATTEMPTS - 1:
                raise
            time.sleep(_BACKOFF_SECONDS * (2**attempt))
    raise AssertionError("unreachable: loop always returns or raises")


def _proposal_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "explanation": {"type": "string"},
            "constant_name": {"type": "string"},
            "edits": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "old_snippet": {"type": "string"},
                        "new_snippet": {"type": "string"},
                    },
                    "required": ["old_snippet", "new_snippet"],
                },
            },
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["explanation", "constant_name", "edits", "confidence"],
    }


def _proposal_schema_gemini() -> dict[str, Any]:
    # Gemini's responseSchema is a restricted OpenAPI subset that rejects
    # unknown fields like "additionalProperties" outright (400 INVALID_ARGUMENT).
    schema = _proposal_schema()
    schema.pop("additionalProperties", None)
    schema["properties"]["edits"]["items"].pop("additionalProperties", None)
    return schema
