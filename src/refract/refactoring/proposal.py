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
class SnippetEdit:
    """One verbatim old->new replacement within a single file.

    A refactoring often needs several of these at once -- extract-constant is a
    new `const` definition *plus* the rewritten use site; extract-method adds a
    helper *plus* replaces the block with a call; rename-identifier touches every
    occurrence. Expressing each as its own edit is what lets those land as valid
    code instead of being crammed into one contiguous snippet (the Run 1 failure
    mode -- see references/README.md (findings)).
    """

    old_snippet: str
    new_snippet: str


@dataclass(frozen=True)
class RefactorProposal:
    explanation: str
    edits: tuple[SnippetEdit, ...]
    confidence: float
    # For an extract-constant fix the model supplies only the constant's name
    # here (UPPER_SNAKE_CASE) and lets refract place the definition and rewrite
    # the literal deterministically -- see refactoring/extract_constant.py. Empty
    # for every other fix, where ``edits`` carries the whole change.
    constant_name: str = ""

    @property
    def old_snippet(self) -> str:
        """First edit's old snippet -- convenience for the common single-edit
        case and for reporting. Multi-location callers iterate ``edits``."""
        return self.edits[0].old_snippet

    @property
    def new_snippet(self) -> str:
        return self.edits[0].new_snippet

    @classmethod
    def from_json(cls, payload: str | dict[str, Any]) -> RefactorProposal:
        data = json.loads(payload) if isinstance(payload, str) else payload

        if "explanation" not in data:
            raise ValueError("Provider response missing fields: explanation")
        if "confidence" not in data:
            raise ValueError("Provider response missing fields: confidence")

        edits = _parse_edits(data)

        confidence = float(data["confidence"])
        if not 0 <= confidence <= 1:
            raise ValueError("Provider confidence must be between 0 and 1")

        return cls(
            explanation=str(data["explanation"]),
            edits=edits,
            confidence=confidence,
            constant_name=str(data.get("constant_name", "")),
        )


def _parse_edits(data: dict[str, Any]) -> tuple[SnippetEdit, ...]:
    """Read the edit list from either the multi-edit ``edits`` shape or the
    legacy single ``old_snippet``/``new_snippet`` pair, so a provider (or an old
    cached response) using either form parses the same way."""
    if "edits" in data:
        raw_edits = data["edits"]
        if not isinstance(raw_edits, list) or not raw_edits:
            raise ValueError("Provider response 'edits' must be a non-empty list")
        return tuple(_parse_edit(item) for item in raw_edits)

    missing = sorted({"old_snippet", "new_snippet"} - set(data))
    if missing:
        raise ValueError(f"Provider response missing fields: {', '.join(missing)}")
    return (_parse_edit(data),)


def _parse_edit(item: dict[str, Any]) -> SnippetEdit:
    for key in ("old_snippet", "new_snippet"):
        if key not in item:
            raise ValueError(f"Provider edit missing field: {key}")
        if not isinstance(item[key], str) or not item[key].strip():
            raise ValueError(f"Provider edit must include a non-empty {key}")
    return SnippetEdit(old_snippet=item["old_snippet"], new_snippet=item["new_snippet"])


class RefactorProvider(Protocol):
    name: ProviderName

    def propose(self, system_prompt: str, user_prompt: str) -> RefactorProposal: ...


# placeholder bodies to reject instead of applying
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
    """Reject a proposal whose edits can't be applied cleanly against ``source``.

    Each edit must resolve to exactly one location and be a real change with no
    stub placeholder. Uniqueness is checked here against the pristine source as a
    pre-flight; the patcher re-checks each edit against the running text as it
    applies them in order, since one edit can change what a later one matches.
    """
    for edit in proposal.edits:
        occurrences = source.count(edit.old_snippet)
        if occurrences == 0:
            raise ValueError("Provider old_snippet was not found in the target file")
        if occurrences > 1:
            raise ValueError("Provider old_snippet is ambiguous in the target file")

        if edit.old_snippet == edit.new_snippet:
            raise ValueError("Provider proposed no source change")

        lowered = edit.new_snippet.lower()
        for pattern in _STUB_PATTERNS:
            if pattern in lowered:
                raise ValueError(f"Provider new_snippet contains a stub placeholder: {pattern!r}")
