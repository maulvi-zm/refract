import subprocess
from pathlib import Path

import pytest

from refract.benchmark import oracle
from refract.benchmark.oracle import OracleError, count_smells
from refract.core.models import SmellType


@pytest.fixture(autouse=True)
def _clear_oracle_env(monkeypatch):
    """Start every test from a deliberately-unconfigured oracle so a stray env
    var from the surrounding shell can't change strict/None behavior."""
    for var in ("REFRACT_DESIGNITE_JAR", "REFRACT_DPY_BIN", "REFRACT_DPY_CONFIG"):
        monkeypatch.delenv(var, raising=False)

_JAVA_CSV = (
    "Project Name,Package Name,Type Name,Method Name,Code Smell\n"
    "p,pkg,A,foo,Long Method\n"
    "p,pkg,A,bar,Long Method\n"
    "p,pkg,A,baz,Magic Number\n"
    "p,pkg,B,qux,Long Identifier\n"
)

_PY_JSON = (
    '[{"Smell": "Long method", "File": "a.py"},'
    ' {"Smell": "Long method", "File": "b.py"},'
    ' {"Smell": "Magic number", "File": "c.py"},'
    ' {"Smell": "Long statement", "File": "d.py"}]'
)


def _java_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "A.java").write_text("class A {}")
    return repo


def _python_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("x = 1\n")
    return repo


def test_count_smells_none_when_no_oracle_configured(tmp_path) -> None:
    # env cleared by the autouse fixture -> oracle deliberately disabled
    assert count_smells(_java_repo(tmp_path), SmellType.LONG_METHOD) is None


def test_count_smells_raises_when_binary_configured_but_missing(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("REFRACT_DESIGNITE_JAR", str(tmp_path / "nope.jar"))
    with pytest.raises(OracleError, match="not found"):
        count_smells(_java_repo(tmp_path), SmellType.LONG_METHOD)


def test_count_java_parses_csv_and_filters_smell_type(tmp_path, monkeypatch) -> None:
    jar = tmp_path / "DesigniteJava.jar"
    jar.write_text("")  # only its existence is checked
    monkeypatch.setenv("REFRACT_DESIGNITE_JAR", str(jar))

    def fake_run(cmd, **kwargs):
        out = Path(cmd[cmd.index("-o") + 1])
        (out / "implementationCodeSmells.csv").write_text(_JAVA_CSV)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(oracle.subprocess, "run", fake_run)
    repo = _java_repo(tmp_path)
    assert count_smells(repo, SmellType.LONG_METHOD) == 2
    assert count_smells(repo, SmellType.MAGIC_NUMBER) == 1
    assert count_smells(repo, SmellType.LONG_IDENTIFIER) == 1


def test_count_python_parses_json_and_filters_smell_type(tmp_path, monkeypatch) -> None:
    dpy = tmp_path / "DPy"
    dpy.write_text("")
    config = tmp_path / "dpy_config.json"
    config.write_text("{}")
    monkeypatch.setenv("REFRACT_DPY_BIN", str(dpy))
    monkeypatch.setenv("REFRACT_DPY_CONFIG", str(config))

    def fake_run(cmd, **kwargs):
        out = Path(cmd[cmd.index("-o") + 1])
        (out / "proj_implementation_smells.json").write_text(_PY_JSON)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(oracle.subprocess, "run", fake_run)
    repo = _python_repo(tmp_path)
    assert count_smells(repo, SmellType.LONG_METHOD) == 2
    assert count_smells(repo, SmellType.MAGIC_NUMBER) == 1


def test_count_smells_raises_on_nonzero_oracle_exit(tmp_path, monkeypatch) -> None:
    jar = tmp_path / "DesigniteJava.jar"
    jar.write_text("")
    monkeypatch.setenv("REFRACT_DESIGNITE_JAR", str(jar))

    def boom(cmd, **kwargs):
        raise subprocess.CalledProcessError(1, cmd, stderr="detector blew up")

    monkeypatch.setattr(oracle.subprocess, "run", boom)
    with pytest.raises(OracleError, match="exited 1"):
        count_smells(_java_repo(tmp_path), SmellType.LONG_METHOD)


def test_count_smells_raises_when_oracle_produces_no_output(tmp_path, monkeypatch) -> None:
    jar = tmp_path / "DesigniteJava.jar"
    jar.write_text("")
    monkeypatch.setenv("REFRACT_DESIGNITE_JAR", str(jar))

    def no_output(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(oracle.subprocess, "run", no_output)
    with pytest.raises(OracleError, match="no implementationCodeSmells.csv"):
        count_smells(_java_repo(tmp_path), SmellType.LONG_METHOD)


def test_count_smells_raises_when_language_unknown_but_oracle_configured(
    tmp_path, monkeypatch
) -> None:
    jar = tmp_path / "DesigniteJava.jar"
    jar.write_text("")
    monkeypatch.setenv("REFRACT_DESIGNITE_JAR", str(jar))
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(OracleError, match="cannot determine language"):
        count_smells(empty, SmellType.LONG_METHOD)


def test_language_none_returns_none_when_oracle_disabled(tmp_path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    assert count_smells(empty, SmellType.LONG_METHOD) is None


def test_preflight_raises_when_configured_jar_missing(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("REFRACT_DESIGNITE_JAR", str(tmp_path / "nope.jar"))
    with pytest.raises(OracleError, match="not found"):
        oracle.preflight()


def test_preflight_noop_when_oracle_disabled() -> None:
    # env cleared by the autouse fixture -> nothing configured -> no raise
    oracle.preflight()


def test_attach_oracle_records_after_failure_without_raising(monkeypatch, tmp_path) -> None:
    from refract.benchmark import runner
    from refract.benchmark.runner import ToolResult, _attach_oracle

    def boom(_repo, _smell):
        raise OracleError("DesigniteJava exited 1: compilation unit conversion")

    monkeypatch.setattr(runner, "oracle_count_smells", boom)
    result = ToolResult(
        tool="refract",
        model="m",
        api_calls=1,
        input_tokens=1,
        output_tokens=1,
        smells_before=5,
        smells_after=3,
    )
    # pristine before-count succeeded (passed in); the patched after-count blows up
    _attach_oracle(result, tmp_path, SmellType.MAGIC_NUMBER, oracle_before=42)

    assert result.oracle_smells_before == 42
    assert result.oracle_smells_after is None  # unmeasurable, not silently 0
    assert "oracle could not analyze patched repo" in result.error  # surfaced, not hidden
