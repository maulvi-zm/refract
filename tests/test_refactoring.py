from pathlib import Path

import pytest

from refract.core.models import MethodInfo, RepositoryIndex, SmellLocation, SmellType
from refract.refactoring.patcher import apply_snippet_replacement
from refract.refactoring.pipeline import run_refactor
from refract.refactoring.proposal import (
    ProviderName,
    RefactorProposal,
    validate_replacement,
)
from refract.refactoring.providers import config_from_env


def _proposal(old: str, new: str) -> RefactorProposal:
    return RefactorProposal(explanation="x", old_snippet=old, new_snippet=new, confidence=0.9)


def test_proposal_from_json_rejects_missing_fields() -> None:
    with pytest.raises(ValueError):
        RefactorProposal.from_json({"explanation": "x", "old_snippet": "a"})


def test_proposal_from_json_rejects_out_of_range_confidence() -> None:
    with pytest.raises(ValueError):
        RefactorProposal.from_json(
            {"explanation": "x", "old_snippet": "a", "new_snippet": "b", "confidence": 2}
        )


def test_validate_replacement_rejects_ambiguous_match() -> None:
    with pytest.raises(ValueError):
        validate_replacement("int a = 1;\nint a = 1;", _proposal("int a = 1;", "int b = 1;"))


def test_validate_replacement_rejects_stub_body() -> None:
    with pytest.raises(ValueError):
        validate_replacement("def f(): return 1", _proposal("return 1", "pass  # todo"))


def test_patcher_dry_run_does_not_write(tmp_path: Path) -> None:
    source = tmp_path / "Example.java"
    source.write_text("class Example { int a = 1; }\n", encoding="utf-8")

    updated = apply_snippet_replacement(
        source, _proposal("int a = 1;", "int count = 1;"), apply=False
    )

    assert "int count = 1;" in updated
    # dry run, so the file on disk is untouched
    assert "int a = 1;" in source.read_text(encoding="utf-8")


def test_config_from_env_reads_key_and_model_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    config = config_from_env("openai", "model-x")

    assert config.provider is ProviderName.OPENAI
    assert config.model == "model-x"
    assert config.api_key == "test-key"


def test_pipeline_dry_run_uses_provider_without_writing(tmp_path: Path) -> None:
    source = tmp_path / "example.py"
    source.write_text("def target():\n    value = 42\n    return value\n", encoding="utf-8")

    index = RepositoryIndex(
        methods=[MethodInfo("target", "<unknown>", source, 1, 3, 1)],
        smells=[SmellLocation(SmellType.MAGIC_NUMBER, source, 2, "42", "magic")],
    )

    captured: dict[str, str] = {}

    class FakeProvider:
        name = ProviderName.OPENAI

        def propose(self, system_prompt: str, user_prompt: str) -> RefactorProposal:
            captured["system"] = system_prompt
            captured["user"] = user_prompt
            return _proposal("value = 42", "value = LIMIT")

    results = run_refactor(index, tmp_path, SmellType.MAGIC_NUMBER, 1, FakeProvider(), apply=False)

    assert len(results) == 1
    assert "python refactoring assistant" in captured["system"]
    assert "Smell: magic_number" in captured["user"]
    assert "value = 42" in source.read_text(encoding="utf-8")
