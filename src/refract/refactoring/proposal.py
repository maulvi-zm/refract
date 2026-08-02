from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol


class ProviderName(str, Enum):
    OPENAI = "openai"
    GEMINI = "gemini"


@dataclass(frozen=True)
class ProviderConfig:
    provider: ProviderName
    model: str
    api_key: str | None


@dataclass(frozen=True)
class RefactorProposal:
    explanation: str
    old_snippet: str
    new_snippet: str
    confidence: float

    @classmethod
    def from_json(cls, payload: str | dict[str, Any]) -> RefactorProposal:
        data = json.loads(payload) if isinstance(payload, str) else payload

        required = {"explanation", "old_snippet", "new_snippet", "confidence"}
        missing = sorted(required - set(data))
        if missing:
            raise ValueError(f"Provider response missing fields: {', '.join(missing)}")

        for key in ("old_snippet", "new_snippet"):
            if not isinstance(data[key], str) or not data[key].strip():
                raise ValueError(f"Provider response must include a non-empty {key}")

        confidence = float(data["confidence"])
        if not 0 <= confidence <= 1:
            raise ValueError("Provider confidence must be between 0 and 1")

        return cls(
            explanation=str(data["explanation"]),
            old_snippet=data["old_snippet"],
            new_snippet=data["new_snippet"],
            confidence=confidence,
        )


class RefactorProvider(Protocol):
    name: ProviderName

    def propose(self, system_prompt: str, user_prompt: str) -> RefactorProposal: ...


# a new_snippet containing any of these is a stub, not a refactoring
_STUB_PATTERNS = (
    "// implement",
    "// todo",
    "// add logic",
    "// your code",
    "# implement",
    "# todo",
    "# your code",
    "pass  # ",
)


def validate_replacement(source: str, proposal: RefactorProposal) -> None:
    occurrences = source.count(proposal.old_snippet)
    if occurrences == 0:
        raise ValueError("Provider old_snippet was not found in the target file")
    if occurrences > 1:
        raise ValueError("Provider old_snippet is ambiguous in the target file")

    if proposal.old_snippet == proposal.new_snippet:
        raise ValueError("Provider proposed no source change")

    lowered = proposal.new_snippet.lower()
    for pattern in _STUB_PATTERNS:
        if pattern in lowered:
            raise ValueError(f"Provider new_snippet contains a stub placeholder: {pattern!r}")
