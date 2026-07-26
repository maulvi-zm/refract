from pathlib import Path

from refract.verification.runner import (
    _compile_only_command,
    detect_test_command,
    parse_test_command,
)


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


def test_compile_only_maps_maven_test_to_test_compile_keeping_args() -> None:
    assert _compile_only_command(["mvn", "test"]) == ["mvn", "test-compile"]
    # gson's reactor args must survive the rewrite
    assert _compile_only_command(["mvn", "test", "-pl", "gson", "-am"]) == [
        "mvn",
        "test-compile",
        "-pl",
        "gson",
        "-am",
    ]
    assert _compile_only_command(["/repo/mvnw", "test"]) == ["/repo/mvnw", "test-compile"]


def test_compile_only_maps_gradle_test_to_test_classes() -> None:
    assert _compile_only_command(["gradle", "test"]) == ["gradle", "testClasses"]
    assert _compile_only_command(["/repo/gradlew", "test"]) == ["/repo/gradlew", "testClasses"]


def test_compile_only_is_none_for_python_or_unknown() -> None:
    # pytest / unittest compile nothing -> no cheap tier; gate falls through to verify()
    assert _compile_only_command(["pytest"]) is None
    assert _compile_only_command(["pytest", "--benchmark-disable"]) is None
    assert _compile_only_command(["python", "-m", "unittest", "discover"]) is None
    assert _compile_only_command([]) is None
