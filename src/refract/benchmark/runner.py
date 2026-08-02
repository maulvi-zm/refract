from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from refract.benchmark.proxy import CountingProxy, ProxyStats
from refract.core.models import RepositoryIndex, SmellType
from refract.indexing.database import save
from refract.indexing.repository import index_repository
from refract.refactoring.pipeline import run_refactor
from refract.refactoring.providers import config_from_env, provider_from_config

_TIMEOUT_SECONDS = 300


@dataclass
class ToolResult:
    tool: str
    model: str
    api_calls: int
    input_tokens: int
    output_tokens: int
    smells_before: int
    smells_after: int
    exit_code: int = 0
    error: str = ""

    @property
    def fixed(self) -> int:
        return max(0, self.smells_before - self.smells_after)


def run_benchmark(
    repo: Path,
    smell_type: SmellType,
    model: str,
    api_key: str,
    limit: int = 10,
    codex_api_key_mode: bool = False,
    verbose: bool = False,
) -> list[ToolResult]:
    """Run each tool on its own copy of the repo.

    codex_api_key_mode routes codex through the proxy; otherwise it uses its
    own ChatGPT auth and calls are counted from its JSONL output.
    """
    with tempfile.TemporaryDirectory(prefix="refract_bench_") as tmp:
        tmp_path = Path(tmp)
        refract_dir = tmp_path / "refract_copy"
        codex_dir = tmp_path / "codex_copy"
        shutil.copytree(repo, refract_dir)
        shutil.copytree(repo, codex_dir)

        initial_index = index_repository(refract_dir)
        smells_before = len(initial_index.smells_by_type(smell_type))

        return [
            _run_refract(
                refract_dir, initial_index, smell_type, model, api_key, limit, smells_before
            ),
            _run_codex(
                codex_dir, smell_type, model, api_key, smells_before, codex_api_key_mode, verbose
            ),
        ]


def _run_refract(
    repo_dir: Path,
    initial_index: RepositoryIndex,
    smell_type: SmellType,
    model: str,
    api_key: str,
    limit: int,
    smells_before: int,
) -> ToolResult:
    # refract talks to the proxy, which counts usage and forwards to OpenAI
    proxy = CountingProxy("https://api.openai.com")
    proxy.start()

    saved_env = _set_env(
        OPENAI_BASE_URL=proxy.base_url,
        OPENAI_API_KEY=api_key,
        REFRACT_PROVIDER="openai",
        REFRACT_MODEL=model,
    )

    error = ""
    try:
        provider = provider_from_config(config_from_env("openai", model))
        save(initial_index, repo_dir.parent / "refract.db")
        run_refactor(
            index=initial_index,
            repo_root=repo_dir,
            smell_type=smell_type,
            limit=limit,
            provider=provider,
            apply=True,
        )
    except Exception as exc:  # noqa: BLE001 - report any failure in the result
        error = str(exc)
    finally:
        stats = _snapshot(proxy.stats)
        proxy.stop()
        _restore_env(saved_env)

    smells_after = len(index_repository(repo_dir).smells_by_type(smell_type))
    return ToolResult(
        tool="refract",
        model=model,
        api_calls=stats["calls"],
        input_tokens=stats["in"],
        output_tokens=stats["out"],
        smells_before=smells_before,
        smells_after=smells_after,
        error=error,
    )


def _codex_prompt(smell_type: SmellType) -> str:
    return (
        f"Fix all '{smell_type.value}' code smells in this repository. "
        "Refactor by extracting helpers/constants as appropriate. "
        "Apply all changes directly to the source files."
    )


def _run_codex(
    repo_dir: Path,
    smell_type: SmellType,
    model: str,
    api_key: str,
    smells_before: int,
    api_key_mode: bool,
    verbose: bool,
) -> ToolResult:
    codex_bin = shutil.which("codex")
    if not codex_bin:
        return ToolResult(
            tool="codex",
            model=model,
            api_calls=0,
            input_tokens=0,
            output_tokens=0,
            smells_before=smells_before,
            smells_after=smells_before,
            exit_code=-1,
            error="codex binary not found on PATH",
        )

    prompt = _codex_prompt(smell_type)
    if api_key_mode:
        return _run_codex_api_key_mode(
            codex_bin, repo_dir, smell_type, model, api_key, smells_before, prompt, verbose
        )
    return _run_codex_chatgpt_mode(
        codex_bin, repo_dir, smell_type, api_key, smells_before, prompt, verbose
    )


def _run_codex_api_key_mode(
    codex_bin: str,
    repo_dir: Path,
    smell_type: SmellType,
    model: str,
    api_key: str,
    smells_before: int,
    prompt: str,
    verbose: bool,
) -> ToolResult:
    # codex appends /responses to base_url, so the upstream must end in /v1
    proxy = CountingProxy("https://api.openai.com/v1")
    proxy.start()

    error = ""
    exit_code = 0
    try:
        cmd = [
            codex_bin,
            "exec",
            "--json",
            "-c",
            "model_provider=openai-direct",
            "-c",
            "model_providers.openai-direct.name=OpenAI Direct",
            "-c",
            f"model_providers.openai-direct.base_url={proxy.base_url}",
            "-c",
            "model_providers.openai-direct.env_key=OPENAI_API_KEY",
            "-m",
            model,
            "--dangerously-bypass-approvals-and-sandbox",
            "--skip-git-repo-check",
            "-C",
            str(repo_dir),
            prompt,
        ]
        proc = subprocess.run(
            cmd,
            env={**os.environ, "OPENAI_API_KEY": api_key},
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_SECONDS,
        )
        exit_code = proc.returncode
        if proc.returncode != 0 and proc.stderr:
            error = proc.stderr.strip()
        if verbose:
            print(proc.stdout)
    except subprocess.TimeoutExpired:
        error, exit_code = f"codex timed out after {_TIMEOUT_SECONDS} s", -1
    except Exception as exc:  # noqa: BLE001
        error, exit_code = str(exc), -1
    finally:
        stats = _snapshot(proxy.stats)
        proxy.stop()

    smells_after, error = _count_after(repo_dir, smell_type, smells_before, error)
    return ToolResult(
        tool="codex",
        model=model,
        api_calls=stats["calls"],
        input_tokens=stats["in"],
        output_tokens=stats["out"],
        smells_before=smells_before,
        smells_after=smells_after,
        exit_code=exit_code,
        error=error,
    )


def _run_codex_chatgpt_mode(
    codex_bin: str,
    repo_dir: Path,
    smell_type: SmellType,
    api_key: str,
    smells_before: int,
    prompt: str,
    verbose: bool,
) -> ToolResult:
    # no proxy here: each turn.completed event in the JSONL output is one call
    api_calls = input_tokens = output_tokens = 0
    error = ""
    exit_code = 0
    try:
        cmd = [
            codex_bin,
            "exec",
            "--json",
            "--dangerously-bypass-approvals-and-sandbox",
            "--skip-git-repo-check",
            "-C",
            str(repo_dir),
            prompt,
        ]
        proc = subprocess.run(
            cmd,
            env={**os.environ, "OPENAI_API_KEY": api_key},
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_SECONDS,
        )
        exit_code = proc.returncode
        for line in proc.stdout.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "turn.completed":
                api_calls += 1
                usage = event.get("usage") or {}
                input_tokens += usage.get("input_tokens", usage.get("prompt_tokens", 0))
                output_tokens += usage.get("output_tokens", usage.get("completion_tokens", 0))
        if exit_code != 0 and proc.stderr:
            error = proc.stderr.strip()
        if verbose:
            print(proc.stdout)
    except subprocess.TimeoutExpired:
        error, exit_code = f"codex timed out after {_TIMEOUT_SECONDS} s", -1
    except Exception as exc:  # noqa: BLE001
        error, exit_code = str(exc), -1

    smells_after, error = _count_after(repo_dir, smell_type, smells_before, error)
    return ToolResult(
        tool="codex",
        model="codex-default",
        api_calls=api_calls,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        smells_before=smells_before,
        smells_after=smells_after,
        exit_code=exit_code,
        error=error,
    )


def _count_after(
    repo_dir: Path, smell_type: SmellType, smells_before: int, error: str
) -> tuple[int, str]:
    try:
        return len(index_repository(repo_dir).smells_by_type(smell_type)), error
    except Exception as exc:  # noqa: BLE001
        fallback_error = error or f"re-index failed (codex may have broken syntax): {exc}"
        return smells_before, fallback_error


def _snapshot(stats: ProxyStats) -> dict[str, int]:
    return {"calls": stats.api_calls, "in": stats.input_tokens, "out": stats.output_tokens}


def _set_env(**values: str) -> dict[str, str | None]:
    saved = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    return saved


def _restore_env(saved: dict[str, str | None]) -> None:
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
