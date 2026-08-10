from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from refract.core.models import RepositoryIndex, SmellLocation
from refract.indexing.repository import index_repository

_TEST_TIMEOUT_SECONDS = 300


@dataclass(frozen=True)
class VerificationResult:
    command: list[str] | None
    returncode: int
    stdout: str
    stderr: str
    index: RepositoryIndex  # repo state after the tests ran

    @property
    def passed(self) -> bool:
        return self.command is not None and self.returncode == 0

    @property
    def smells(self) -> list[SmellLocation]:
        return self.index.smells


def detect_test_command(repo_root: Path) -> list[str] | None:
    return (
        _detect_java_command(repo_root)
        or _detect_node_command(repo_root)
        or _detect_python_command(repo_root)
    )


def _detect_java_command(repo_root: Path) -> list[str] | None:
    if (repo_root / "mvnw").exists():
        return [str(repo_root / "mvnw"), "test"]
    if (repo_root / "pom.xml").exists():
        return ["mvn", "test"] if shutil.which("mvn") else None
    if (repo_root / "gradlew").exists():
        return [str(repo_root / "gradlew"), "test"]

    has_gradle = any((repo_root / name).exists() for name in ("build.gradle", "build.gradle.kts"))
    if has_gradle and shutil.which("gradle"):
        return ["gradle", "test"]

    return None


def _detect_node_command(repo_root: Path) -> list[str] | None:
    """``<package manager> test`` when package.json declares a test script.

    The script itself is not inspected beyond its existence -- a NestJS repo
    runs jest, a Vite one runs vitest, and either way the package manager is the
    stable entry point. We require the script because npm's built-in fallback for
    a missing one exits non-zero with "Missing script", which would read as a
    behavioural failure on every single target.
    """
    manifest = repo_root / "package.json"
    if not manifest.exists():
        return None

    try:
        scripts = json.loads(manifest.read_text(encoding="utf-8")).get("scripts") or {}
    except (json.JSONDecodeError, OSError, UnicodeDecodeError, AttributeError):
        return None
    if "test" not in scripts:
        return None

    manager = _node_package_manager(repo_root)
    return [manager, "test"] if manager else None


def _node_package_manager(repo_root: Path) -> str | None:
    """The package manager this repo is locked to, if it is installed.

    Lockfile-driven rather than npm-by-default: running ``npm test`` in a pnpm
    workspace resolves dependencies differently (or not at all), so a wrong guess
    fails the suite for reasons that have nothing to do with the refactor.
    """
    for lockfile, manager in (
        ("pnpm-lock.yaml", "pnpm"),
        ("yarn.lock", "yarn"),
        ("bun.lockb", "bun"),
        ("package-lock.json", "npm"),
    ):
        if (repo_root / lockfile).exists():
            return manager if shutil.which(manager) else None
    return "npm" if shutil.which("npm") else None


def _detect_python_command(repo_root: Path) -> list[str] | None:
    if not _has_python_tests(repo_root):
        return None
    # a repo-local venv (its own dependencies installed) beats whatever pytest
    # happens to be on PATH -- otherwise this repo's tests fail on
    # ModuleNotFoundError for its own package before a single line of the
    # actual refactor is evaluated.
    venv_pytest = repo_root / ".venv" / "bin" / "pytest"
    if venv_pytest.exists():
        return [str(venv_pytest)]
    if shutil.which("pytest"):
        return ["pytest"]
    return [sys.executable, "-m", "unittest", "discover"]


def _has_python_tests(repo_root: Path) -> bool:
    if (repo_root / "tests").is_dir():
        return True
    return any(repo_root.rglob("test_*.py")) or any(repo_root.rglob("*_test.py"))


def parse_test_command(repo_root: Path, command: str) -> list[str] | None:
    if command == "auto":
        return detect_test_command(repo_root)
    return command.split()


def _compile_only_command(command: list[str], repo_root: Path | None = None) -> list[str] | None:
    """A cheap "does it still compile?" variant of ``command``, or None when the
    toolchain has no separate compile phase (Python -- pytest compiles nothing).

    Tier 1 of the two-tier behavioural gate: a Java edit that won't compile (a
    hallucinated import, a wrong inferred type, a severed reference) is caught in
    seconds without running the suite. Maven's ``test`` becomes ``test-compile``,
    Gradle's becomes ``testClasses``; other args (``-pl gson -am``) are kept.
    TypeScript has no such sub-phase, so it gets a type-check instead, which
    needs ``repo_root`` to find the project's own compiler."""
    if not command:
        return None
    exe = Path(command[0]).name
    if exe.startswith("mvn"):
        return [command[0]] + ["test-compile" if a == "test" else a for a in command[1:]]
    if exe.startswith("gradle"):
        return [command[0]] + ["testClasses" if a == "test" else a for a in command[1:]]
    if exe in ("npm", "pnpm", "yarn", "bun") and repo_root is not None:
        return _typecheck_command(repo_root)
    return None  # pytest / unittest / unknown: no separate compile step


def _typecheck_command(repo_root: Path) -> list[str] | None:
    """``tsc --noEmit`` against the repo's own compiler, or None when this isn't
    a TypeScript project (plain JavaScript has nothing to check ahead of time).

    Deliberately restricted to the locally installed binary rather than ``npx``,
    which would reach for the network to fetch a compiler whose version doesn't
    match the project -- a slow tier-1 gate that reports type errors the repo's
    real build would never raise.
    """
    if not (repo_root / "tsconfig.json").exists():
        return None
    tsc = repo_root / "node_modules" / ".bin" / "tsc"
    if not tsc.exists():
        return None
    return [str(tsc), "--noEmit", "-p", str(repo_root / "tsconfig.json")]


def verify_compiles(repo_root: Path, test_command: str = "auto") -> bool | None:
    """Whether the repo still compiles, via the compile-only phase of its test
    command. None when the toolchain has no compile phase (Python), so the caller
    should fall through to ``verify``."""
    command = parse_test_command(repo_root, command=test_command)
    compile_cmd = _compile_only_command(command, repo_root) if command else None
    if compile_cmd is None:
        return None
    return _run_tests(repo_root, compile_cmd).returncode == 0


def verify(repo_root: Path, test_command: str = "auto") -> VerificationResult:
    command = parse_test_command(repo_root, command=test_command)
    completed = _run_tests(repo_root, command)

    return VerificationResult(
        command=command,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        index=index_repository(repo_root),
    )


def _test_env(repo_root: Path) -> dict[str, str]:
    """Force the repo copy's own source onto PYTHONPATH ahead of everything else.

    A copied repo (``copytree``) breaks a uv venv, which is not relocatable:
    the copy's console scripts keep an absolute shebang to the
    original venv's python, whose editable-install .pth injects the original source
    into sys.path -- so the test-gate would pass no matter what the tool edited.
    PYTHONPATH precedes site-packages .pth additions, so prepending the copy's roots
    makes the edited source win. Harmless for non-Python repos."""
    env = os.environ.copy()
    roots = [str(repo_root)]
    src = repo_root / "src"
    if src.is_dir():
        roots.insert(0, str(src))  # src-layout: the package lives under src/
    existing = env.get("PYTHONPATH")
    if existing:
        roots.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(roots)
    return env


def _run_tests(repo_root: Path, command: list[str] | None) -> subprocess.CompletedProcess[str]:
    if not command:
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

    try:
        return subprocess.run(
            command,
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
            timeout=_TEST_TIMEOUT_SECONDS,
            env=_test_env(repo_root),
        )
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(
            args=command,
            returncode=1,
            stdout="",
            stderr=f"timeout: test command exceeded {_TEST_TIMEOUT_SECONDS} seconds",
        )
