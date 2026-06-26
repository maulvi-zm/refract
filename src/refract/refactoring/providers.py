from __future__ import annotations

import json
import os
from typing import Any
from urllib import request

from refract.refactoring.proposal import ProviderConfig, ProviderName, RefactorProposal

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
        return (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.config.model}:generateContent?key={self.config.api_key}"
        )

    def _payload(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        return {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": _proposal_schema(),
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
    req = request.Request(url, data=body, headers=headers, method="POST")
    with request.urlopen(req, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def _proposal_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "explanation": {"type": "string"},
            "old_snippet": {"type": "string"},
            "new_snippet": {"type": "string"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["explanation", "old_snippet", "new_snippet", "confidence"],
    }
