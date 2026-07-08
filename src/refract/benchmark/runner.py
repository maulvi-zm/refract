from __future__ import annotations

import contextlib
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

from refract.benchmark.oracle import OracleError
from refract.benchmark.oracle import count_smells as oracle_count_smells
from refract.benchmark.proxy import CountingProxy, ProxyStats
from refract.core.models import MethodInfo, RepositoryIndex, SmellLocation, SmellType
from refract.indexing.database import save
from refract.indexing.parser import parse
from refract.indexing.repository import index_repository, source_files
from refract.languages.registry import spec_for_path
from refract.refactoring.pipeline import run_refactor
from refract.refactoring.providers import config_from_env, provider_from_config
from refract.verification.runner import verify

# Agentic tools run a verify loop against `refract check`, so give them headroom.
# 600s wasn't enough in practice -- gemini-cli hit it on 3/3 pilot runs while still
# making real incremental progress each time, not stuck.
_TIMEOUT_SECONDS = 1800


@dataclass
class ToolResult:
    tool: str
    model: str
    api_calls: int
    input_tokens: int
    output_tokens: int
    smells_before: int
    smells_after: int
    # Subset of api_calls that returned an upstream error (5xx/429/connection).
    # Counted so cost/volume isn't silently understated when a run struggles.
    failed_api_calls: int = 0
    # File-scoped cyclomatic complexity: the sum over *every* method in each
    # edited file, before/after. Unlike a target-only sum this counts helpers a
    # tool extracts (they're new methods in the same file), so extract-method
    # redistribution shows up honestly -- a pure move nets ~0, only a real
    # simplification drops. See references/run1-refract-baseline.md §5.
    complexity_before: int = 0
    complexity_after: int = 0
    # Target methods that no longer exist after the refactor (renamed/removed);
    # excluded from the after distribution rather than counted as CC 0.
    complexity_unmatched: int = 0
    # Distribution of the flagged (target) methods' own complexity -- what
    # actually matters for long_method: did the smelly method drop below
    # threshold. Median/max over targets before, and over the matched targets
    # after. 0.0 when there were no in-method targets.
    target_cc_before_median: float = 0.0
    target_cc_after_median: float = 0.0
    target_cc_before_max: int = 0
    target_cc_after_max: int = 0
    syntax_broken_files: int = 0
    tests_passed: bool | None = None  # None: no test command detected, not "passed"
    # Independent oracle (DesigniteJava/DPy) whole-repo counts, filled in only
    # when an oracle is configured (REFRACT_DESIGNITE_JAR / REFRACT_DPY_BIN);
    # None means "not validated", not zero. See benchmark/oracle.py.
    oracle_smells_before: int | None = None
    oracle_smells_after: int | None = None
    exit_code: int = 0
    error: str = ""

    @property
    def fixed(self) -> int:
        return max(0, self.smells_before - self.smells_after)

    @property
    def oracle_fixed(self) -> int | None:
        """Instances the independent oracle agrees were removed, or None if the
        oracle didn't run. Can be negative if the tool introduced new smells."""
        if self.oracle_smells_before is None or self.oracle_smells_after is None:
            return None
        return self.oracle_smells_before - self.oracle_smells_after

    @property
    def complexity_delta(self) -> int:
        """Negative means file-scoped complexity dropped. Counts extracted helpers,
        so a redistribution nets ~0 rather than reading as a reduction."""
        return self.complexity_after - self.complexity_before


@dataclass
class TargetMethod:
    """A flagged smell's enclosing method, resolved once against the pre-refactor
    index so every tool's after-state can be compared against the same baseline.

    class_name is part of the identity, not just decoration: files with several
    classes routinely repeat method names (__init__, convert, edit, ...), and
    matching on (file, name) alone silently pairs the target with an unrelated
    same-named method elsewhere in the file after re-indexing.
    """

    file: Path  # relative to the repo root
    class_name: str
    name: str
    start_line: int  # tiebreaker when (file, class_name, name) still isn't unique
    complexity_before: int


@dataclass
class Baseline:
    """Precomputed once from the pristine repo copy, before any tool -- including
    refract's own run -- has touched a single file. Shared read-only across every
    tool's after-analysis so they're all judged against the same starting point.

    tests_passed is the pristine repo's own test-suite result: a flaky or
    already-broken test suite shouldn't be blamed on whichever tool happened
    to run against it. None means no test command could be detected at all.
    """

    target_methods: list[TargetMethod]
    # For each file that holds a target method, the summed CC of *all* its
    # methods pre-refactor -- the file-scoped baseline the after-state is diffed
    # against so extracted helpers are counted (see ToolResult.complexity_before).
    file_complexity_before: dict[Path, int]
    broken_files: frozenset[Path]  # relative paths tree-sitter already can't parse
    tests_passed: bool | None
    test_command: str  # "auto", or a repo-specific override (e.g. a multi-module
    # Maven project where the generic `mvn test` pulls in unrelated submodules)


AGENTIC_TOOLS = ("codex", "opencode", "gemini")

# Gemini talks to Google, not OpenAI, so it needs its own model + key.
_DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"


def run_benchmark(
    repo: Path,
    smell_type: SmellType,
    model: str,
    api_key: str,
    limit: int = 10,
    tools: list[str] | None = None,
    codex_api_key_mode: bool = False,
    codex_base_url: str = "https://api.openai.com/v1",
    codex_api_key: str = "",
    gemini_model: str = _DEFAULT_GEMINI_MODEL,
    gemini_api_key: str = "",
    refract_provider: str = "openai",
    test_command: str = "auto",
    verbose: bool = False,
    done_tools: set[str] | None = None,
    on_result: Callable[[ToolResult], None] | None = None,
    workdir: Path | None = None,
) -> list[ToolResult]:
    """Run refract plus each requested agentic tool on its own copy of the repo.

    Every tool gets an isolated ``shutil.copytree`` copy so they don't step on
    each other. ``tools`` selects which agentic competitors to run alongside
    refract (any of :data:`AGENTIC_TOOLS`); refract is always the baseline.

    refract_provider selects which provider backs refract's own baseline run:
    "openai" (default) interprets model/api_key as an OpenAI model+key; "gemini"
    interprets them as a Gemini model+key instead, so a whole run can go through
    a single Gemini key with no OpenAI account involved at all.

    codex_api_key_mode True routes codex through the proxy with the OpenAI key
    (needs a verified org). False lets codex use its ChatGPT auth and we count
    calls from its JSONL turn.completed events instead. opencode and gemini both
    run natively on Gemini (gemini_model/gemini_api_key), each routed through its
    own counting proxy that forwards to Google's generativelanguage endpoint.

    test_command overrides how the test suite is run, same syntax as `refract
    verify --test-command`. "auto" (default) detects mvn/gradle/pytest, which
    works for most repos, but not e.g. a multi-module Maven project where the
    generic `mvn test` also builds unrelated integration-test submodules --
    pass an explicit command (e.g. "mvn test -pl gson -am") for those.

    done_tools names tools ("refract"/"codex"/"opencode"/"gemini") whose result
    is already available (e.g. from a checkpoint) -- they're skipped, and only
    the remaining tools run. on_result, if given, is called with each ToolResult
    the moment that tool finishes, so a caller can persist per-tool progress and
    resume a partially-completed cell without re-spending the finished tools'
    calls. The shared per-cell setup (index, baseline test run, oracle-before)
    still runs, since the remaining tools need it to be scored.

    workdir, if given, is used as the parent for every tool's repo copy instead
    of a self-deleting temp dir, so the actual patched code + per-target diffs
    survive the run and each edit is auditable (roadmap #6). None keeps the old
    behavior: an auto-cleaned tempfile.TemporaryDirectory.
    """
    done = done_tools or set()
    selected = tools if tools is not None else ["codex"]
    unknown = [t for t in selected if t not in AGENTIC_TOOLS]
    if unknown:
        raise ValueError(f"unknown benchmark tool(s): {', '.join(unknown)}")

    with _benchmark_workdir(workdir) as tmp_path:
        refract_dir = tmp_path / "refract_copy"
        _fresh_copytree(repo, refract_dir)

        initial_index = index_repository(refract_dir)
        flagged = initial_index.smells_by_type(smell_type)
        smells_before = len(flagged)  # repo-wide, uncapped: catches regressions too
        capped_flagged = _cap_flagged(flagged, limit)
        target_count = len(capped_flagged)
        targets = _target_list(capped_flagged, refract_dir)
        baseline_test_result = verify(refract_dir, test_command)
        target_methods = _target_methods(initial_index, capped_flagged, refract_dir)
        baseline = Baseline(
            target_methods=target_methods,
            file_complexity_before=_file_complexity(
                initial_index, refract_dir, {t.file for t in target_methods}
            ),
            broken_files=_syntax_error_files(refract_dir),
            tests_passed=baseline_test_result.passed
            if baseline_test_result.command is not None
            else None,
            test_command=test_command,
        )

        # Independent oracle count on the pristine repo, shared across all tools
        # (they each start from an identical copy). None when no oracle is
        # configured -- see benchmark/oracle.py.
        oracle_before = oracle_count_smells(refract_dir, smell_type)

        results: list[ToolResult] = []
        if "refract" not in done:
            refract_result = _run_refract(
                refract_dir,
                initial_index,
                smell_type,
                model,
                api_key,
                limit,
                smells_before,
                baseline,
                refract_provider,
            )
            _attach_oracle(refract_result, refract_dir, smell_type, oracle_before)
            if on_result is not None:
                on_result(refract_result)
            results.append(refract_result)

        for tool in selected:
            if tool in done:
                continue
            tool_dir = tmp_path / f"{tool}_copy"
            _fresh_copytree(repo, tool_dir)
            if tool == "codex":
                result = _run_codex(
                    tool_dir,
                    smell_type,
                    model,
                    codex_api_key or api_key,
                    smells_before,
                    targets,
                    target_count,
                    baseline,
                    codex_api_key_mode,
                    codex_base_url,
                    verbose,
                )
            elif tool == "opencode":
                result = _run_opencode(
                    tool_dir,
                    smell_type,
                    gemini_model,
                    gemini_api_key,
                    smells_before,
                    targets,
                    target_count,
                    baseline,
                    verbose,
                )
            elif tool == "gemini":
                result = _run_gemini(
                    tool_dir,
                    smell_type,
                    gemini_model,
                    gemini_api_key,
                    smells_before,
                    targets,
                    target_count,
                    baseline,
                    verbose,
                )
            else:  # pragma: no cover -- guarded by the AGENTIC_TOOLS check above
                continue
            _attach_oracle(result, tool_dir, smell_type, oracle_before)
            if on_result is not None:
                on_result(result)
            results.append(result)
        return results


def _attach_oracle(
    result: ToolResult, repo_dir: Path, smell_type: SmellType, oracle_before: int | None
) -> None:
    """Record the independent oracle's before/after counts on a tool's result.

    ``oracle_before`` is measured once on the pristine repo and reused; the
    after-count is measured on this tool's own patched copy. Both are None when
    no oracle is configured, so this is a no-op in that case.

    The after-count is measured on a repo the tool just edited, so the oracle's
    parser can legitimately choke on it when the tool produced code the parser
    rejects (Eclipse JDT in DesigniteJava is stricter than our tree-sitter
    detector). That's a fact about the tool, not an infra failure -- so we
    record ``oracle_smells_after=None`` plus a visible note and carry on, rather
    than aborting the whole run. Misconfiguration and an unanalyzable *pristine*
    repo still raise loudly (that happens earlier, on ``oracle_before``).
    """
    result.oracle_smells_before = oracle_before
    try:
        result.oracle_smells_after = oracle_count_smells(repo_dir, smell_type)
    except OracleError as exc:
        result.oracle_smells_after = None
        note = f"oracle could not analyze patched repo (tool may have broken it): {str(exc)[:200]}"
        result.error = f"{result.error}; {note}" if result.error else note


@contextlib.contextmanager
def _benchmark_workdir(workdir: Path | None) -> Iterator[Path]:
    """Yield the parent dir for each tool's repo copy.

    None -> a self-deleting temp dir (default, nothing to inspect afterwards).
    A path -> that dir, created if needed and left in place, so the patched
    repos and diffs survive for auditing (roadmap #6).
    """
    if workdir is not None:
        workdir.mkdir(parents=True, exist_ok=True)
        yield workdir
    else:
        with tempfile.TemporaryDirectory(prefix="refract_bench_") as tmp:
            yield Path(tmp)


def _fresh_copytree(src: Path, dst: Path) -> None:
    """copytree that tolerates a pre-existing target -- a persistent --keep-workdir
    can already hold a previous run's copy, and copytree refuses an existing dst."""
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _file_complexity(
    index: RepositoryIndex, repo_dir: Path, rel_files: set[Path]
) -> dict[Path, int]:
    """Sum the cyclomatic complexity of *all* methods in each given file.

    Keyed by repo-relative path so the after-run can recompute the same scope on
    its own copy. Counting the whole file (not just the target methods) is what
    makes extracted helpers visible in the delta -- see ToolResult.complexity_before.
    """
    return {
        rel: sum(m.cyclomatic_complexity for m in index.methods_by_file(repo_dir / rel))
        for rel in rel_files
    }


def _cap_flagged(flagged: list[SmellLocation], limit: int) -> list[SmellLocation]:
    """Deterministically cap the flagged set to the same first-`limit` instances
    refract's own run_refactor will attempt (identical sort key), so the
    agentic-tool prompt lists exactly what refract is scoped to -- not the
    full repo-wide list while refract only attempts `limit` of them.
    """
    return sorted(flagged, key=lambda s: (str(s.file), s.line, s.identifier))[: max(limit, 0)]


def _target_list(flagged: list, repo_dir: Path) -> str:
    """Render the flagged smells as repo-relative bullet lines for the prompt.

    Paths are relative so the same text applies to every tool's copy of the repo.
    """
    lines = []
    for smell in flagged:
        try:
            location = smell.file.resolve().relative_to(repo_dir.resolve())
        except ValueError:
            location = smell.file.name
        lines.append(f"- {location}:{smell.line} - {smell.detail}")
    return "\n".join(lines)


def _enclosing_method(index: RepositoryIndex, file: Path, line: int) -> MethodInfo | None:
    """The innermost method containing a smell's line, by line-range containment.

    Works for all three smell types: long_method's identifier is already the
    method name, but long_identifier/magic_number point at an arbitrary line,
    so line-range containment is the only lookup that generalizes.
    """
    candidates = [m for m in index.methods_by_file(file) if m.start_line <= line <= m.end_line]
    if not candidates:
        return None
    return min(candidates, key=lambda m: m.end_line - m.start_line)


def _target_methods(
    index: RepositoryIndex, flagged: list[SmellLocation], repo_dir: Path
) -> list[TargetMethod]:
    """Resolve each flagged smell to its enclosing method's pre-refactor complexity.

    Smells that fall outside any method (e.g. module-level magic numbers) are
    skipped -- there's no method-level complexity to track for them.
    """
    targets: list[TargetMethod] = []
    seen: set[tuple[Path, str, str]] = set()
    for smell in flagged:
        method = _enclosing_method(index, smell.file, smell.line)
        if method is None:
            continue
        try:
            rel = method.file.resolve().relative_to(repo_dir.resolve())
        except ValueError:
            rel = method.file
        key = (rel, method.class_name, method.name)
        if key in seen:  # multiple smells in the same method: count it once
            continue
        seen.add(key)
        targets.append(
            TargetMethod(
                file=rel,
                class_name=method.class_name,
                name=method.name,
                start_line=method.start_line,
                complexity_before=method.cyclomatic_complexity,
            )
        )
    return targets


@dataclass
class ComplexityResult:
    """The redesigned complexity signal (roadmap #5).

    file_before/file_after are summed over *all* methods in each edited file, so
    extract-method redistribution is visible (a pure move nets ~0). The target_*
    fields describe the flagged methods' own complexity distribution -- what
    matters for long_method is whether the smelly method itself dropped, not a
    sum. unmatched counts targets that no longer exist after the refactor.
    """

    file_before: int
    file_after: int
    unmatched: int
    target_before_median: float
    target_after_median: float
    target_before_max: int
    target_after_max: int


def _complexity_after(
    after_index: RepositoryIndex, repo_dir: Path, baseline: Baseline
) -> ComplexityResult:
    """Measure complexity change file-scoped, plus the target-method distribution.

    File scope counts helpers the tool extracts (new methods in the same file),
    so a redistribution no longer reads as a reduction. Each target is matched
    after the refactor by (file, class, name); a renamed/removed one is counted
    as unmatched and left out of the after distribution rather than treated as
    CC 0. class_name is part of the key because files with several classes reuse
    method names (__init__, convert, ...); ties among same-named free functions
    (e.g. several closures literally named "decorator") break to whichever
    candidate's start_line is closest to the pre-refactor line.
    """
    file_before = sum(baseline.file_complexity_before.values())
    file_after = sum(
        sum(m.cyclomatic_complexity for m in after_index.methods_by_file(repo_dir / rel))
        for rel in baseline.file_complexity_before
    )

    before_ccs = [t.complexity_before for t in baseline.target_methods]
    after_ccs: list[int] = []
    unmatched = 0
    for t in baseline.target_methods:
        matches = [
            m
            for m in after_index.methods_by_file(repo_dir / t.file)
            if m.name == t.name and m.class_name == t.class_name
        ]
        if not matches:
            unmatched += 1
            continue
        closest = min(matches, key=lambda m: abs(m.start_line - t.start_line))
        after_ccs.append(closest.cyclomatic_complexity)

    return ComplexityResult(
        file_before=file_before,
        file_after=file_after,
        unmatched=unmatched,
        target_before_median=_median(before_ccs),
        target_after_median=_median(after_ccs),
        target_before_max=max(before_ccs, default=0),
        target_after_max=max(after_ccs, default=0),
    )


def _median(values: list[int]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _syntax_error_files(repo_dir: Path) -> frozenset[Path]:
    """Relative paths of source files tree-sitter can't cleanly parse.

    tree-sitter is deliberately error-tolerant -- it always produces *a* tree,
    marking unparseable regions with ERROR nodes instead of raising. That means
    a broken patch (bad indentation, an orphaned statement left behind by a
    half-finished extraction) re-indexes "successfully" with a silently wrong
    parse. has_error is the only thing that actually notices. Checked against
    a pre-refactor baseline so a file the repo already parses oddly isn't
    blamed on whichever tool happened to run.
    """
    broken = set()
    for path in source_files(repo_dir):
        spec = spec_for_path(path)
        assert spec is not None
        root = parse(path.read_bytes(), spec.language)
        if root.has_error:
            try:
                broken.add(path.resolve().relative_to(repo_dir.resolve()))
            except ValueError:
                broken.add(path)
    return frozenset(broken)


def _run_refract(
    repo_dir: Path,
    initial_index: RepositoryIndex,
    smell_type: SmellType,
    model: str,
    api_key: str,
    limit: int,
    smells_before: int,
    baseline: Baseline,
    provider_name: str = "openai",
) -> ToolResult:
    # point refract at the proxy, which forwards to the real upstream so calls
    # get counted regardless of which provider is backing this run
    if provider_name == "gemini":
        proxy = CountingProxy("https://generativelanguage.googleapis.com")
        saved_env_values = {
            "GEMINI_BASE_URL": proxy.base_url,
            "GEMINI_API_KEY": api_key,
            "REFRACT_PROVIDER": "gemini",
            "REFRACT_MODEL": model,
        }
    else:
        proxy = CountingProxy("https://api.openai.com")
        saved_env_values = {
            "OPENAI_BASE_URL": proxy.base_url,
            "OPENAI_API_KEY": api_key,
            "REFRACT_PROVIDER": "openai",
            "REFRACT_MODEL": model,
        }
    proxy.start()
    saved_env = _set_env(**saved_env_values)

    error = ""
    try:
        provider = provider_from_config(config_from_env(provider_name, model))
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

    analysis = _analyze_after(repo_dir, smell_type, smells_before, baseline, error)
    return ToolResult(
        tool="refract",
        model=model,
        api_calls=stats["calls"],
        failed_api_calls=stats["failed"],
        input_tokens=stats["in"],
        output_tokens=stats["out"],
        smells_before=smells_before,
        smells_after=analysis.smells_after,
        **_complexity_kwargs(analysis.complexity),
        syntax_broken_files=analysis.syntax_broken_files,
        tests_passed=analysis.tests_passed,
        error=analysis.error,
    )


def _checker_cmd(repo_dir: Path, smell_type: SmellType) -> str:
    # invoke the same detector the benchmark scores with, via `python -m refract`
    # so it works regardless of PATH inside the agent's shell.
    return f'{sys.executable} -m refract check "{repo_dir}" --smell {smell_type.value}'


def _agentic_prompt(
    smell_type: SmellType,
    targets: str,
    target_count: int,
    smells_before: int,
    checker_cmd: str,
) -> str:
    """Prompt that hands the agent the exact metric, the flagged targets, and a
    ground-truth checker to loop against — the same information refract has.

    The checker is repo-wide, but `targets` may be a deterministic subset
    (matching whatever refract's own --limit attempts, so neither side gets an
    easier or harder job). Telling the agent to loop until the repo-wide count
    hits *0* is only correct when targets is the full flagged set -- otherwise
    it's an unreachable goal and the agent burns its whole budget looping.
    Give it the reachable number instead.
    """
    goal_count = smells_before - target_count
    stop_condition = (
        f"it reports 0 smells (all {target_count} are repo-wide, so 0 is reachable)"
        if target_count >= smells_before
        else (
            f"it reports {goal_count} or fewer (there are {smells_before} total "
            f"'{smell_type.value}' smells repo-wide; you are only asked to fix the "
            f"{target_count} listed above, not the rest, so {goal_count} remaining "
            "is success, not 0)"
        )
    )
    return (
        f"This repository has '{smell_type.value}' code smells, flagged by this "
        "project's own detector. The currently flagged instances are:\n\n"
        f"{targets}\n\n"
        "Refactor each flagged method by extracting helper methods/constants so "
        "that the ORIGINAL method drops below the detector's threshold. "
        "IMPORTANT: actually remove the statements you extract from the original "
        "method body — do NOT leave them as unreachable/dead code after a return "
        "statement, or the detector will still count them. Do not touch any "
        "other instance of this smell outside the list above.\n\n"
        "After editing, verify your work by running this exact command:\n\n"
        f"    {checker_cmd}\n\n"
        f"It reindexes the repo and prints the repo-wide remaining "
        f"'{smell_type.value}' count. Keep editing and re-running it until "
        f"{stop_condition}. Apply all changes directly to the source files."
    )


def _run_codex(
    repo_dir: Path,
    smell_type: SmellType,
    model: str,
    api_key: str,
    smells_before: int,
    targets: str,
    target_count: int,
    baseline: Baseline,
    api_key_mode: bool,
    base_url: str,
    verbose: bool,
) -> ToolResult:
    codex_bin = _resolve_binary("REFRACT_CODEX_BIN", "codex")
    if not codex_bin:
        return _missing_binary(
            "codex", model, smells_before, baseline, "codex binary not found on PATH"
        )

    prompt = _agentic_prompt(
        smell_type, targets, target_count, smells_before, _checker_cmd(repo_dir, smell_type)
    )
    if api_key_mode:
        return _run_codex_api_key_mode(
            codex_bin,
            repo_dir,
            smell_type,
            model,
            api_key,
            smells_before,
            baseline,
            prompt,
            base_url,
            verbose,
        )
    return _run_codex_chatgpt_mode(
        codex_bin, repo_dir, smell_type, api_key, smells_before, baseline, prompt, verbose
    )


def _run_codex_api_key_mode(
    codex_bin: str,
    repo_dir: Path,
    smell_type: SmellType,
    model: str,
    api_key: str,
    smells_before: int,
    baseline: Baseline,
    prompt: str,
    base_url: str,
    verbose: bool,
) -> ToolResult:
    # codex tacks /responses onto base_url, so it must already end in /v1.
    # Point this at CCX (e.g. http://localhost:3000/v1) instead of real OpenAI
    # to put codex on the same model as everything else without an OpenAI key.
    proxy = CountingProxy(base_url)
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
            "-c",
            "model_providers.openai-direct.wire_api=responses",
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

    analysis = _analyze_after(repo_dir, smell_type, smells_before, baseline, error)
    return ToolResult(
        tool="codex",
        model=model,
        api_calls=stats["calls"],
        failed_api_calls=stats["failed"],
        input_tokens=stats["in"],
        output_tokens=stats["out"],
        smells_before=smells_before,
        smells_after=analysis.smells_after,
        **_complexity_kwargs(analysis.complexity),
        syntax_broken_files=analysis.syntax_broken_files,
        tests_passed=analysis.tests_passed,
        exit_code=exit_code,
        error=analysis.error,
    )


def _run_codex_chatgpt_mode(
    codex_bin: str,
    repo_dir: Path,
    smell_type: SmellType,
    api_key: str,
    smells_before: int,
    baseline: Baseline,
    prompt: str,
    verbose: bool,
) -> ToolResult:
    # one turn.completed event in the JSONL output = one inference call
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

    analysis = _analyze_after(repo_dir, smell_type, smells_before, baseline, error)
    return ToolResult(
        tool="codex",
        model="codex-default",
        api_calls=api_calls,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        smells_before=smells_before,
        smells_after=analysis.smells_after,
        **_complexity_kwargs(analysis.complexity),
        syntax_broken_files=analysis.syntax_broken_files,
        tests_passed=analysis.tests_passed,
        exit_code=exit_code,
        error=analysis.error,
    )


def _run_opencode(
    repo_dir: Path,
    smell_type: SmellType,
    model: str,
    api_key: str,
    smells_before: int,
    targets: str,
    target_count: int,
    baseline: Baseline,
    verbose: bool,
) -> ToolResult:
    """opencode (https://opencode.ai) driven headlessly with `opencode run`.

    Uses opencode's native `google` provider (backed by @ai-sdk/google) so it
    speaks to Gemini directly instead of being shimmed through an OpenAI-shaped
    endpoint. A per-run config file (OPENCODE_CONFIG) overrides that provider's
    baseURL to point at the counting proxy, which forwards to Google's
    generativelanguage endpoint and counts usageMetadata -- mirroring
    _run_gemini so opencode's cost/token stats land in the same ProxyStats.
    """
    opencode_bin = _resolve_binary("REFRACT_OPENCODE_BIN", "opencode")
    if not opencode_bin:
        return _missing_binary(
            "opencode", model, smells_before, baseline, "opencode binary not found on PATH"
        )
    if not api_key:
        return _missing_binary(
            "opencode", model, smells_before, baseline, "GEMINI_API_KEY is not set"
        )

    proxy = CountingProxy("https://generativelanguage.googleapis.com")
    proxy.start()

    # @ai-sdk/google builds request URLs as `{baseURL}/models/{model}:generate...`
    # against a default baseURL of .../v1beta, so the override must keep the
    # /v1beta suffix for the proxied path to match Google's real endpoint layout.
    config = {
        "$schema": "https://opencode.ai/config.json",
        "provider": {
            "google": {
                "options": {"baseURL": f"{proxy.base_url}/v1beta", "apiKey": api_key},
                "models": {model: {}},
            }
        },
    }
    config_path = repo_dir.parent / "opencode.json"
    config_path.write_text(json.dumps(config))

    prompt = _agentic_prompt(
        smell_type, targets, target_count, smells_before, _checker_cmd(repo_dir, smell_type)
    )
    error = ""
    exit_code = 0
    try:
        # opencode's `run` has no directory flag -- it uses the process cwd as
        # the project root (confirmed against the pinned build; passing --dir
        # makes it print usage and exit without doing anything).
        cmd = [
            opencode_bin,
            "run",
            "--format",
            "json",
            "-m",
            f"google/{model}",
            prompt,
        ]
        proc = subprocess.run(
            cmd,
            cwd=str(repo_dir),
            env={
                **os.environ,
                "GOOGLE_GENERATIVE_AI_API_KEY": api_key,
                "OPENCODE_CONFIG": str(config_path),
            },
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_SECONDS,
        )
        exit_code = proc.returncode
        if exit_code != 0:
            error = _clean_subprocess_stderr(proc.stderr)
        if verbose:
            print(proc.stdout)
    except subprocess.TimeoutExpired:
        error, exit_code = f"opencode timed out after {_TIMEOUT_SECONDS} s", -1
    except Exception as exc:  # noqa: BLE001
        error, exit_code = str(exc), -1
    finally:
        stats = _snapshot(proxy.stats)
        proxy.stop()

    error = error or _no_calls_error("opencode", stats["calls"])
    analysis = _analyze_after(repo_dir, smell_type, smells_before, baseline, error)
    return ToolResult(
        tool="opencode",
        model=model,
        api_calls=stats["calls"],
        failed_api_calls=stats["failed"],
        input_tokens=stats["in"],
        output_tokens=stats["out"],
        smells_before=smells_before,
        smells_after=analysis.smells_after,
        **_complexity_kwargs(analysis.complexity),
        syntax_broken_files=analysis.syntax_broken_files,
        tests_passed=analysis.tests_passed,
        exit_code=exit_code,
        error=analysis.error,
    )


def _run_gemini(
    repo_dir: Path,
    smell_type: SmellType,
    model: str,
    api_key: str,
    smells_before: int,
    targets: str,
    target_count: int,
    baseline: Baseline,
    verbose: bool,
) -> ToolResult:
    """Gemini CLI (https://github.com/google-gemini/gemini-cli) in YOLO mode.

    Routes through the counting proxy using the Gemini key: GOOGLE_GEMINI_BASE_URL
    points the CLI's generativelanguage requests at the proxy, which forwards to
    Google and counts usageMetadata.

    gemini-cli resolves its settings/credentials directory from $HOME
    (Storage.getGlobalGeminiDir() -> os.homedir()), and a persisted
    `selectedAuthType` there takes priority over GEMINI_API_KEY entirely --
    on a machine with an existing "oauth-personal" login, the CLI silently
    uses that cached OAuth session against a different backend and never
    touches GEMINI_API_KEY, GOOGLE_GEMINI_BASE_URL, or this proxy at all
    (confirmed: api_calls stayed 0 across several runs that clearly did real
    work). Pointing HOME at an empty scratch dir for just this subprocess
    avoids that without touching the real ~/.gemini/settings.json, so the
    CLI falls through to GEMINI_API_KEY as intended.
    """
    gemini_bin = _resolve_binary("REFRACT_GEMINI_BIN", "gemini")
    if not gemini_bin:
        return _missing_binary(
            "gemini", model, smells_before, baseline, "gemini binary not found on PATH"
        )
    if not api_key:
        return _missing_binary(
            "gemini", model, smells_before, baseline, "GEMINI_API_KEY is not set"
        )

    proxy = CountingProxy("https://generativelanguage.googleapis.com")
    proxy.start()
    isolated_home = tempfile.mkdtemp(prefix="refract_gemini_home_")

    prompt = _agentic_prompt(
        smell_type, targets, target_count, smells_before, _checker_cmd(repo_dir, smell_type)
    )
    error = ""
    exit_code = 0
    try:
        cmd = [gemini_bin, "-y", "-m", model, "-p", prompt]
        proc = subprocess.run(
            cmd,
            cwd=str(repo_dir),
            env={
                **os.environ,
                "HOME": isolated_home,
                "GEMINI_API_KEY": api_key,
                "GOOGLE_GEMINI_BASE_URL": proxy.base_url,
            },
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_SECONDS,
        )
        exit_code = proc.returncode
        if exit_code != 0:
            error = _clean_subprocess_stderr(proc.stderr)
        if verbose:
            print(proc.stdout)
    except subprocess.TimeoutExpired:
        error, exit_code = f"gemini timed out after {_TIMEOUT_SECONDS} s", -1
    except Exception as exc:  # noqa: BLE001
        error, exit_code = str(exc), -1
    finally:
        stats = _snapshot(proxy.stats)
        proxy.stop()
        shutil.rmtree(isolated_home, ignore_errors=True)

    error = error or _no_calls_error("gemini", stats["calls"])
    analysis = _analyze_after(repo_dir, smell_type, smells_before, baseline, error)
    return ToolResult(
        tool="gemini",
        model=model,
        api_calls=stats["calls"],
        failed_api_calls=stats["failed"],
        input_tokens=stats["in"],
        output_tokens=stats["out"],
        smells_before=smells_before,
        smells_after=analysis.smells_after,
        **_complexity_kwargs(analysis.complexity),
        syntax_broken_files=analysis.syntax_broken_files,
        tests_passed=analysis.tests_passed,
        exit_code=exit_code,
        error=analysis.error,
    )


def _clean_subprocess_stderr(stderr: str) -> str:
    """Drop Node.js deprecation/experimental warnings that the node-based CLIs
    (gemini-cli, opencode) print to stderr on every run. They're not errors, and
    left in they mask an otherwise-successful run as failed (e.g. the punycode
    DEP0040 warning). Whatever remains is treated as a genuine error message."""
    keep = [
        line
        for line in (stderr or "").splitlines()
        if not line.startswith("(node:") and not line.lstrip().startswith("(Use `node")
    ]
    return "\n".join(keep).strip()


def _no_calls_error(tool: str, api_calls: int) -> str:
    """A proxied tool that made zero inference calls did nothing -- flag it
    loudly rather than letting an empty error field read as success (this is
    exactly how the opencode `--dir` misinvocation hid: usage went to stdout,
    stderr was empty, so it looked OK while doing nothing)."""
    if api_calls == 0:
        return (
            f"{tool} made no API calls -- it may have failed to start, authenticate, or parse args"
        )
    return ""


def _missing_binary(
    tool: str, model: str, smells_before: int, baseline: Baseline, message: str
) -> ToolResult:
    # the repo is untouched, so complexity is pristine: file scope unchanged,
    # every target still present (0 unmatched), before == after distribution.
    file_before = sum(baseline.file_complexity_before.values())
    before_ccs = [t.complexity_before for t in baseline.target_methods]
    med = _median(before_ccs)
    mx = max(before_ccs, default=0)
    pristine = ComplexityResult(file_before, file_before, 0, med, med, mx, mx)
    return ToolResult(
        tool=tool,
        model=model,
        api_calls=0,
        input_tokens=0,
        output_tokens=0,
        smells_before=smells_before,
        smells_after=smells_before,
        **_complexity_kwargs(pristine),
        syntax_broken_files=0,
        exit_code=-1,
        error=message,
    )


@dataclass
class AfterAnalysis:
    smells_after: int
    complexity: ComplexityResult
    syntax_broken_files: int
    tests_passed: bool | None
    error: str


def _complexity_kwargs(c: ComplexityResult) -> dict[str, float]:
    """The ComplexityResult mapped onto ToolResult's flat complexity fields, so
    every tool's result is built from one source of truth."""
    return {
        "complexity_before": c.file_before,
        "complexity_after": c.file_after,
        "complexity_unmatched": c.unmatched,
        "target_cc_before_median": c.target_before_median,
        "target_cc_after_median": c.target_after_median,
        "target_cc_before_max": c.target_before_max,
        "target_cc_after_max": c.target_after_max,
    }


def _unchanged_complexity(baseline: Baseline) -> ComplexityResult:
    """Best-effort complexity when the patched repo can't be re-indexed at all
    (a tool broke syntax so badly the parse is unusable): report file scope as
    unchanged and every target as unmatched, rather than inventing a reduction."""
    file_before = sum(baseline.file_complexity_before.values())
    before_ccs = [t.complexity_before for t in baseline.target_methods]
    med = _median(before_ccs)
    mx = max(before_ccs, default=0)
    return ComplexityResult(
        file_before, file_before, len(baseline.target_methods), med, med, mx, mx
    )


def _analyze_after(
    repo_dir: Path,
    smell_type: SmellType,
    smells_before: int,
    baseline: Baseline,
    error: str,
) -> AfterAnalysis:
    """Re-verify the patched repo: re-run its real test suite (reusing the
    resulting index instead of a second separate re-index), recount smells,
    recompute file-scoped + target-method complexity, and diff syntax errors
    against the pristine baseline.
    """
    try:
        test_result = verify(repo_dir, baseline.test_command)
        after_index = test_result.index
    except Exception as exc:  # noqa: BLE001
        fallback_error = error or f"re-index failed (tool may have broken syntax): {exc}"
        return AfterAnalysis(
            smells_before, _unchanged_complexity(baseline), 0, None, fallback_error
        )

    tests_passed = test_result.passed if test_result.command is not None else None
    smells_after = len(after_index.smells_by_type(smell_type))
    complexity = _complexity_after(after_index, repo_dir, baseline)

    newly_broken = _syntax_error_files(repo_dir) - baseline.broken_files
    if newly_broken and not error:
        broken_list = ", ".join(str(p) for p in sorted(newly_broken))
        error = f"syntax error introduced in: {broken_list}"

    return AfterAnalysis(smells_after, complexity, len(newly_broken), tests_passed, error)


def _resolve_binary(env_var: str, tool_name: str) -> str | None:
    """An env var override takes priority over PATH.

    Lets the benchmark pin each competitor to a specific build -- e.g. one
    compiled from references/<tool> at the commit cited in the thesis --
    instead of silently picking up whatever version happens to be installed
    globally on a given machine. See docs/reference-tool-binaries.md.
    """
    return os.getenv(env_var) or shutil.which(tool_name)


def _snapshot(stats: ProxyStats) -> dict[str, int]:
    return {
        "calls": stats.api_calls,
        "failed": stats.failed_calls,
        "in": stats.input_tokens,
        "out": stats.output_tokens,
    }


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
