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


def _package_json(tmp_path: Path, scripts: dict[str, str]) -> None:
    import json

    (tmp_path / "package.json").write_text(json.dumps({"scripts": scripts}), encoding="utf-8")


def test_detects_node_test_script(tmp_path: Path) -> None:
    _package_json(tmp_path, {"test": "jest"})
    (tmp_path / "package-lock.json").write_text("{}", encoding="utf-8")

    command = detect_test_command(tmp_path)

    assert command is None or command == ["npm", "test"]  # None when npm isn't installed


def test_no_node_command_without_a_test_script(tmp_path: Path) -> None:
    # npm invents a "Missing script" failure for a package.json with no test
    # script, which would read as a behavioural regression on every target.
    _package_json(tmp_path, {"build": "nest build"})

    assert detect_test_command(tmp_path) is None


def test_malformed_package_json_does_not_crash_detection(tmp_path: Path) -> None:
    (tmp_path / "package.json").write_text("{ not json", encoding="utf-8")

    assert detect_test_command(tmp_path) is None


def test_node_package_manager_follows_the_lockfile(tmp_path: Path) -> None:
    from refract.verification.runner import _node_package_manager

    (tmp_path / "pnpm-lock.yaml").write_text("", encoding="utf-8")

    manager = _node_package_manager(tmp_path)
    assert manager in (None, "pnpm")  # None when pnpm isn't installed


def test_compile_only_typechecks_typescript_with_the_local_tsc(tmp_path: Path) -> None:
    (tmp_path / "tsconfig.json").write_text("{}", encoding="utf-8")
    tsc = tmp_path / "node_modules" / ".bin" / "tsc"
    tsc.parent.mkdir(parents=True)
    tsc.write_text("#!/bin/sh\n", encoding="utf-8")

    assert _compile_only_command(["npm", "test"], tmp_path) == [
        str(tsc),
        "--noEmit",
        "-p",
        str(tmp_path / "tsconfig.json"),
    ]


def test_compile_only_skips_typecheck_without_a_local_compiler(tmp_path: Path) -> None:
    # npx would fetch a compiler whose version doesn't match the project, so a
    # plain-JS repo (or one with no tsc installed) falls through to the suite.
    (tmp_path / "tsconfig.json").write_text("{}", encoding="utf-8")

    assert _compile_only_command(["npm", "test"], tmp_path) is None
    assert _compile_only_command(["npm", "test"]) is None
