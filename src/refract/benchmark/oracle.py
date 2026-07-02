"""Independent post-hoc smell validation via DesigniteJava (Java) / DPy (Python).

The live benchmark scores smells with refract's own tree-sitter detector, which
is also what drives the agentic tools' ``refract check`` loop. To confirm a
"fix" against a detector that isn't the one under test, these helpers re-run the
independent oracle used to build the dataset and count the same smell type on a
given repo snapshot -- so a tool's before/after can be checked for agreement
with something other than refract itself.

Binaries are resolved from env vars:

    REFRACT_DESIGNITE_JAR  path to DesigniteJava.jar   (Java repos)
    REFRACT_DPY_BIN        path to the DPy binary      (Python repos)
    REFRACT_DPY_CONFIG     path to dpy_config.json     (Python repos)

Failure policy is strict and loud, because a run that silently drops the
independent validation produces results we can't trust:

  * env var **unset** -> oracle deliberately disabled -> returns ``None``.
  * env var **set but** the binary/config is missing, the tool exits non-zero,
    times out, or produces no output file -> raises :class:`OracleError`.

So the only way to get ``None`` back is to explicitly not configure the oracle
(e.g. unit tests, ad-hoc ``refract benchmark`` runs). During the real
experiment the env vars are set, so every cell either yields a real count or
aborts the run with a clear reason.

Counts are whole-repo for the given smell type, mirroring the live
``smells_before``/``smells_after`` scope. Absolute totals will not match
refract's own detector (see dataset/README.md on the LOC- vs statement-based
divergence) -- the validation signal is whether the count moves in the same
direction, not that the two detectors agree on magnitude.
"""

from __future__ import annotations

import csv
import glob
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from refract.core.models import SmellType


class OracleError(RuntimeError):
    """Oracle validation was requested (a binary is configured) but could not
    complete. Raised rather than swallowed so a run never silently omits the
    independent validation it depends on."""


# DesigniteJava's "Code Smell" column values (implementationCodeSmells.csv).
_JAVA_SMELL_NAMES = {
    SmellType.LONG_METHOD: "Long Method",
    SmellType.LONG_IDENTIFIER: "Long Identifier",
    SmellType.MAGIC_NUMBER: "Magic Number",
}
# DPy's "Smell" key values (*_implementation_smells.json). Note the different
# capitalisation from DesigniteJava -- these strings are not interchangeable.
_PY_SMELL_NAMES = {
    SmellType.LONG_METHOD: "Long method",
    SmellType.LONG_IDENTIFIER: "Long identifier",
    SmellType.MAGIC_NUMBER: "Magic number",
}

_ORACLE_TIMEOUT_SECONDS = 600


def java_oracle_configured() -> bool:
    return bool(os.getenv("REFRACT_DESIGNITE_JAR"))


def python_oracle_configured() -> bool:
    return bool(os.getenv("REFRACT_DPY_BIN") or os.getenv("REFRACT_DPY_CONFIG"))


def preflight() -> None:
    """Fail fast if a configured oracle can't actually run, before any refactor
    burns LLM calls. Checks binary/config existence and (for Java) that a JVM is
    on PATH. No-op for oracles that are deliberately unconfigured."""
    if java_oracle_configured():
        jar = os.environ["REFRACT_DESIGNITE_JAR"]
        if not Path(jar).exists():
            raise OracleError(f"REFRACT_DESIGNITE_JAR set but not found: {jar}")
        if shutil.which("java") is None:
            raise OracleError("REFRACT_DESIGNITE_JAR is set but no `java` on PATH to run it")
    if python_oracle_configured():
        dpy = os.getenv("REFRACT_DPY_BIN")
        config = os.getenv("REFRACT_DPY_CONFIG")
        if not dpy or not Path(dpy).exists():
            raise OracleError(f"REFRACT_DPY_BIN set but not found: {dpy}")
        if not config or not Path(config).exists():
            raise OracleError(f"REFRACT_DPY_CONFIG set but not found: {config}")


def count_smells(repo_dir: Path, smell_type: SmellType) -> int | None:
    """Count instances of ``smell_type`` in ``repo_dir`` via the language's oracle.

    Language is inferred from the source files present (java wins ties). Returns
    ``None`` only when the matching oracle is deliberately unconfigured; any
    other failure raises :class:`OracleError`.
    """
    language = _language_for(repo_dir)
    if language == "java":
        return _count_java(repo_dir, smell_type)
    if language == "python":
        return _count_python(repo_dir, smell_type)
    # No source we know how to validate. If an oracle is configured the caller
    # expected a count, so don't quietly return None.
    if java_oracle_configured() or python_oracle_configured():
        raise OracleError(f"cannot determine language (no .java/.py) under {repo_dir}")
    return None


def _language_for(repo_dir: Path) -> str | None:
    if next(repo_dir.rglob("*.java"), None) is not None:
        return "java"
    if next(repo_dir.rglob("*.py"), None) is not None:
        return "python"
    return None


def _count_java(repo_dir: Path, smell_type: SmellType) -> int | None:
    jar = os.getenv("REFRACT_DESIGNITE_JAR")
    if not jar:
        return None  # oracle deliberately disabled
    if not Path(jar).exists():
        raise OracleError(f"REFRACT_DESIGNITE_JAR set but not found: {jar}")
    target = _JAVA_SMELL_NAMES[smell_type]
    with tempfile.TemporaryDirectory(prefix="refract_oracle_") as out:
        _run_oracle(["java", "-jar", jar, "-i", str(repo_dir), "-o", out], "DesigniteJava", repo_dir)
        csv_path = Path(out) / "implementationCodeSmells.csv"
        if not csv_path.exists():
            raise OracleError(
                f"DesigniteJava produced no implementationCodeSmells.csv for {repo_dir}"
            )
        with csv_path.open(newline="") as handle:
            return sum(1 for row in csv.DictReader(handle) if row.get("Code Smell") == target)


def _count_python(repo_dir: Path, smell_type: SmellType) -> int | None:
    dpy = os.getenv("REFRACT_DPY_BIN")
    config = os.getenv("REFRACT_DPY_CONFIG")
    if not dpy and not config:
        return None  # oracle deliberately disabled
    if not dpy or not Path(dpy).exists():
        raise OracleError(f"REFRACT_DPY_BIN set but not found: {dpy}")
    if not config or not Path(config).exists():
        raise OracleError(f"REFRACT_DPY_CONFIG set but not found: {config}")
    target = _PY_SMELL_NAMES[smell_type]
    with tempfile.TemporaryDirectory(prefix="refract_oracle_") as out:
        _run_oracle([dpy, "analyze", "-i", str(repo_dir), "-o", out, "-c", config], "DPy", repo_dir)
        matches = glob.glob(str(Path(out) / "*_implementation_smells.json"))
        if not matches:
            raise OracleError(f"DPy produced no *_implementation_smells.json for {repo_dir}")
        count = 0
        for path in matches:
            with open(path) as handle:
                count += sum(1 for row in json.load(handle) if row.get("Smell") == target)
        return count


def _run_oracle(cmd: list[str], name: str, repo_dir: Path) -> None:
    """Run an oracle subprocess, turning any non-zero exit or timeout into a
    loud OracleError instead of a silently missing count."""
    try:
        subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=_ORACLE_TIMEOUT_SECONDS,
            check=True,
        )
    except subprocess.TimeoutExpired as exc:
        raise OracleError(
            f"{name} timed out after {_ORACLE_TIMEOUT_SECONDS}s on {repo_dir}"
        ) from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()[:500]
        raise OracleError(f"{name} exited {exc.returncode} on {repo_dir}: {stderr}") from exc
