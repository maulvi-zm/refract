from pathlib import Path

from refract.verification.runner import detect_test_command, parse_test_command


def test_detects_maven_wrapper_first(tmp_path: Path) -> None:
    wrapper = tmp_path / "mvnw"
    wrapper.write_text("#!/bin/sh\n", encoding="utf-8")
    (tmp_path / "pom.xml").write_text("<project/>", encoding="utf-8")

    # the wrapper should win over a bare mvn
    assert detect_test_command(tmp_path) == [str(wrapper), "test"]


def test_detects_python_tests_directory(tmp_path: Path) -> None:
    (tmp_path / "tests").mkdir()

    command = detect_test_command(tmp_path)

    assert command is not None
    assert command[0] == "pytest" or command[-2:] == ["unittest", "discover"]


def test_no_command_for_empty_repo(tmp_path: Path) -> None:
    assert detect_test_command(tmp_path) is None


def test_prefers_repo_local_venv_pytest(tmp_path: Path) -> None:
    (tmp_path / "tests").mkdir()
    venv_pytest = tmp_path / ".venv" / "bin" / "pytest"
    venv_pytest.parent.mkdir(parents=True)
    venv_pytest.write_text("#!/bin/sh\n", encoding="utf-8")

    assert detect_test_command(tmp_path) == [str(venv_pytest)]


def test_parse_explicit_command_splits_words(tmp_path: Path) -> None:
    assert parse_test_command(tmp_path, "pytest -q") == ["pytest", "-q"]
