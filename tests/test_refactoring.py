from pathlib import Path

import pytest

from refract.core.models import MethodInfo, RepositoryIndex, SmellLocation, SmellType
from refract.refactoring.patcher import apply_snippet_replacement
from refract.refactoring.pipeline import run_refactor
from refract.refactoring.proposal import (
    ProviderName,
    RefactorProposal,
    SnippetEdit,
    validate_replacement,
)
from refract.refactoring.providers import config_from_env


def _proposal(old: str, new: str) -> RefactorProposal:
    return RefactorProposal(explanation="x", edits=(SnippetEdit(old, new),), confidence=0.9)


def test_proposal_from_json_rejects_missing_fields() -> None:
    with pytest.raises(ValueError):
        RefactorProposal.from_json({"explanation": "x", "old_snippet": "a"})


def test_proposal_from_json_parses_multi_edit_list() -> None:
    proposal = RefactorProposal.from_json(
        {
            "explanation": "extract constant",
            "edits": [
                {"old_snippet": "MAX = 0", "new_snippet": "MAX = 10  # limit"},
                {"old_snippet": "x > 10", "new_snippet": "x > MAX"},
            ],
            "confidence": 0.8,
        }
    )
    assert len(proposal.edits) == 2
    # convenience accessors point at the first edit
    assert proposal.old_snippet == "MAX = 0"
    assert proposal.new_snippet == "MAX = 10  # limit"


def test_proposal_from_json_accepts_legacy_single_snippet() -> None:
    proposal = RefactorProposal.from_json(
        {"explanation": "x", "old_snippet": "a", "new_snippet": "b", "confidence": 0.5}
    )
    assert proposal.edits == (SnippetEdit("a", "b"),)


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


def test_patcher_applies_multiple_edits_in_one_file(tmp_path: Path) -> None:
    source = tmp_path / "example.py"
    source.write_text("LIMIT = 0\n\n\ndef check(x):\n    return x > 10\n", encoding="utf-8")

    proposal = RefactorProposal(
        explanation="extract constant",
        edits=(
            SnippetEdit("LIMIT = 0", "LIMIT = 10"),
            SnippetEdit("return x > 10", "return x > LIMIT"),
        ),
        confidence=0.9,
    )
    apply_snippet_replacement(source, proposal, apply=True)

    text = source.read_text(encoding="utf-8")
    assert "LIMIT = 10" in text
    assert "return x > LIMIT" in text


def test_patcher_rejects_edit_that_breaks_syntax(tmp_path: Path) -> None:
    source = tmp_path / "example.py"
    original = "def f():\n    return 1\n"
    source.write_text(original, encoding="utf-8")

    # an unbalanced paren makes the file unparseable -- the guardrail must refuse
    broken = _proposal("return 1", "return (1")

    with pytest.raises(ValueError):
        apply_snippet_replacement(source, broken, apply=True)
    # do no harm: the file on disk is untouched
    assert source.read_text(encoding="utf-8") == original


def test_patcher_rejects_java_constant_hoisted_outside_class(tmp_path: Path) -> None:
    # tree-sitter parses a `static final` field at file scope as a valid
    # local_variable_declaration (no ERROR node), so the syntax guard alone lets
    # it through -- but javac won't compile it. The structural guard must reject
    # a declaration placed directly under the compilation unit.
    source = tmp_path / "Example.java"
    original = (
        "import java.io.IOException;\n\npublic class Example {\n    int f() { return 8; }\n}\n"
    )
    source.write_text(original, encoding="utf-8")

    hoisted = RefactorProposal(
        explanation="x",
        edits=(
            SnippetEdit(
                "import java.io.IOException;",
                "import java.io.IOException;\n\nprivate static final int BITS = 8;",
            ),
            SnippetEdit("return 8", "return BITS"),
        ),
        confidence=0.9,
    )

    with pytest.raises(ValueError):
        apply_snippet_replacement(source, hoisted, apply=True)
    # do no harm: the file on disk is untouched
    assert source.read_text(encoding="utf-8") == original


def test_patcher_rejects_edit_that_orphans_code_after_return(tmp_path: Path) -> None:
    # The pint:long_method failure mode reduced to its fingerprint: after a
    # botched extract-method the block has a `return` followed by a live
    # statement -- unreachable code that runs None-returning behaviour. Valid
    # syntax + a "shorter" method, so only the dead-code guard catches it.
    source = tmp_path / "mod.py"
    original = "def prepare(data):\n    out = list(data)\n    return finalize(out)\n"
    source.write_text(original, encoding="utf-8")

    botched = RefactorProposal(
        explanation="x",
        edits=(
            SnippetEdit(
                "    return finalize(out)\n",
                "    return finalize(out)\n    out = extra(out)\n",
            ),
        ),
        confidence=0.9,
    )

    with pytest.raises(ValueError):
        apply_snippet_replacement(source, botched, apply=True)
    assert source.read_text(encoding="utf-8") == original


def test_dead_code_guard_allows_early_returns(tmp_path: Path) -> None:
    # A guard clause (return nested inside an `if`, followed by more code) is not
    # dead code -- the guard must not fire on a benign edit to such a function.
    source = tmp_path / "mod.py"
    source.write_text(
        "def f(x):\n    if x < 0:\n        return 0\n    return x + 1\n", encoding="utf-8"
    )
    edit = RefactorProposal(
        explanation="x",
        edits=(SnippetEdit("    return x + 1\n", "    return x + STEP\n"),),
        confidence=0.9,
    )
    # STEP is undefined, but that's a name error at runtime, not a parse/dead-code
    # problem -- the structural guards must let it through.
    apply_snippet_replacement(source, edit, apply=True)
    assert "return x + STEP" in source.read_text(encoding="utf-8")


def test_config_from_env_reads_key_and_model_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    config = config_from_env("openai", "model-x")

    assert config.provider is ProviderName.OPENAI
    assert config.model == "model-x"
    assert config.api_key == "test-key"


class _QueueProvider:
    """Returns queued proposals in order (repeating the last), recording each
    user prompt so a test can assert the repair feedback was fed back."""

    name = ProviderName.OPENAI

    def __init__(self, proposals: list[RefactorProposal]) -> None:
        self._proposals = proposals
        self.calls = 0
        self.prompts: list[str] = []

    def propose(self, system_prompt: str, user_prompt: str) -> RefactorProposal:
        self.prompts.append(user_prompt)
        proposal = self._proposals[min(self.calls, len(self._proposals) - 1)]
        self.calls += 1
        return proposal


def _magic_index(source: Path) -> RepositoryIndex:
    return RepositoryIndex(
        methods=[MethodInfo("target", "<unknown>", source, 1, 3, 1)],
        smells=[SmellLocation(SmellType.MAGIC_NUMBER, source, 2, "42", "magic")],
    )


def test_run_refactor_retries_with_feedback_then_applies(tmp_path: Path) -> None:
    source = tmp_path / "mod.py"
    source.write_text("def target():\n    value = 42\n    return value\n", encoding="utf-8")

    rejected = _proposal("value = 999", "value = LIMIT")  # 999 not in source -> rejected
    valid = _proposal("value = 42", "value = LIMIT")
    provider = _QueueProvider([rejected, valid])

    results = run_refactor(
        _magic_index(source), tmp_path, SmellType.MAGIC_NUMBER, 1, provider, True
    )

    assert provider.calls == 2  # first rejected, retried once
    assert len(results) == 1
    assert results[0].attempts == 2
    assert "value = LIMIT" in source.read_text(encoding="utf-8")
    # the retry prompt carried the rejection feedback back to the model
    assert "REJECTED" in provider.prompts[1]
    assert "not found" in provider.prompts[1].lower()


def test_run_refactor_gives_up_after_max_attempts_leaving_file_intact(tmp_path: Path) -> None:
    source = tmp_path / "mod.py"
    original = "def target():\n    value = 42\n    return value\n"
    source.write_text(original, encoding="utf-8")

    # unbalanced paren: every attempt leaves the file unparseable, so all are rejected
    always_bad = _proposal("value = 42", "value = (42")
    provider = _QueueProvider([always_bad])

    results = run_refactor(
        _magic_index(source), tmp_path, SmellType.MAGIC_NUMBER, 1, provider, True, max_attempts=3
    )

    assert provider.calls == 3  # exhausted the budget
    assert results == []  # target skipped
    assert source.read_text(encoding="utf-8") == original  # do no harm: untouched


def test_max_attempts_one_disables_retry(tmp_path: Path) -> None:
    source = tmp_path / "mod.py"
    source.write_text("def target():\n    value = 42\n    return value\n", encoding="utf-8")

    provider = _QueueProvider([_proposal("value = 999", "value = LIMIT")])

    results = run_refactor(
        _magic_index(source), tmp_path, SmellType.MAGIC_NUMBER, 1, provider, True, max_attempts=1
    )

    assert provider.calls == 1  # single-shot: no retry
    assert results == []


def test_run_refactor_prefers_deterministic_constant_extraction(tmp_path: Path) -> None:
    source = tmp_path / "mod.py"
    source.write_text(
        "import os\n\n\ndef target():\n    return os.read(0, 4096)\n", encoding="utf-8"
    )

    index = RepositoryIndex(
        methods=[MethodInfo("target", "<unknown>", source, 4, 5, 1)],
        smells=[SmellLocation(SmellType.MAGIC_NUMBER, source, 5, "4096", "magic")],
    )
    # the model only names the constant and ships a deliberately non-matching edit;
    # the deterministic path must rescue it and land on the first attempt.
    proposal = RefactorProposal(
        explanation="x",
        edits=(SnippetEdit("this does not match anything", "nope"),),
        confidence=0.9,
        constant_name="BUFFER_SIZE",
    )
    provider = _QueueProvider([proposal])

    results = run_refactor(index, tmp_path, SmellType.MAGIC_NUMBER, 1, provider, True)

    text = source.read_text(encoding="utf-8")
    assert provider.calls == 1  # no retries: the extraction applies immediately
    assert results[0].attempts == 1
    assert "BUFFER_SIZE = 4096" in text
    assert text.index("BUFFER_SIZE = 4096") < text.index("def target()")
    assert "os.read(0, BUFFER_SIZE)" in text


def test_build_repair_prompt_includes_error_and_failed_edit() -> None:
    from refract.refactoring.prompt import build_repair_prompt

    text = build_repair_prompt("BASE CONTEXT", _proposal("missing_snippet", "replacement"), "boom")

    assert "BASE CONTEXT" in text
    assert "boom" in text
    assert "missing_snippet" in text  # the rejected edit is shown back


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
