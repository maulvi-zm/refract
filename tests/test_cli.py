import argparse
from pathlib import Path
from unittest.mock import patch

import pytest

from refract.cli import _cmd_benchmark


def _args(**overrides) -> argparse.Namespace:
    defaults = dict(
        repo=Path("."),
        smell="long_method",
        model="gpt-4o-mini",
        limit=10,
        verbose=False,
        tools="codex",
        gemini_model="gemini-2.5-flash",
        codex_api_key_mode=False,
        refract_provider="openai",
        test_command="auto",
        keep_workdir=None,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_openai_provider_requires_openai_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(SystemExit, match="OPENAI_API_KEY is not set"):
        _cmd_benchmark(_args(refract_provider="openai"))


def test_gemini_provider_requires_gemini_key_not_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-unused")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(SystemExit, match="GEMINI_API_KEY is not set"):
        _cmd_benchmark(_args(refract_provider="gemini"))


def test_gemini_provider_and_test_command_reach_run_benchmark(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")

    with patch("refract.benchmark.runner.run_benchmark", return_value=[]) as mock_run:
        _cmd_benchmark(_args(refract_provider="gemini", test_command="mvn test -pl gson -am"))

    assert mock_run.call_args.kwargs["refract_provider"] == "gemini"
    assert mock_run.call_args.kwargs["test_command"] == "mvn test -pl gson -am"
    assert mock_run.call_args.kwargs["api_key"] == "fake-key"
    assert mock_run.call_args.kwargs["workdir"] is None  # not requested by default


def test_keep_workdir_flag_reaches_run_benchmark(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    kept = tmp_path / "kept"

    with patch("refract.benchmark.runner.run_benchmark", return_value=[]) as mock_run:
        _cmd_benchmark(_args(refract_provider="gemini", keep_workdir=kept))

    assert mock_run.call_args.kwargs["workdir"] == kept.resolve()
