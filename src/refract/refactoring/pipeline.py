from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from refract.core.models import RepositoryIndex, SmellLocation, SmellType
from refract.indexing.repository import index_repository
from refract.planning.context import RefactorContext, build_context
from refract.refactoring.extract_constant import build_constant_extraction
from refract.refactoring.patcher import apply_snippet_replacement
from refract.refactoring.prompt import build_repair_prompt, build_system_prompt, build_user_prompt
from refract.refactoring.proposal import RefactorProposal, RefactorProvider

# One initial attempt plus up to two feedback-driven retries. Enough to recover
# the common recoverable failures (a snippet copied imperfectly, an ambiguous
# match, an edit that doesn't parse) without letting a persistently-broken target
# burn an unbounded number of calls. Set to 1 to disable retries entirely.
_DEFAULT_MAX_ATTEMPTS = 3


@dataclass(frozen=True)
class RefactorResult:
    context: RefactorContext
    proposal: RefactorProposal
    applied: bool
    attempts: int  # inference calls it took to land (1 = first try, >1 = repaired)


def run_refactor(
    index: RepositoryIndex,
    repo_root: Path,
    smell_type: SmellType,
    limit: int,
    provider: RefactorProvider,
    apply: bool,
    max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
    verify_after: Callable[[], bool] | None = None,
) -> list[RefactorResult]:
    """Refactor up to ``limit`` smells of ``smell_type``, each via a stateless
    model call spliced in through the decline-only patcher guards.

    ``verify_after`` is the optional behavioural test-gate: a callable returning
    True iff the repo's suite still passes. When given (and ``apply``), an edit
    that fails it is rolled back and the target skipped -- the only guard that
    catches a valid-but-behaviour-changing edit. Opt-in, and only meaningful
    against a baseline that already passes.
    """
    smells = sorted(
        index.smells_by_type(smell_type),
        key=lambda s: (str(s.file), s.line, s.identifier),
    )[: max(limit, 0)]

    live_index = index
    results: list[RefactorResult] = []

    for position, smell in enumerate(smells):
        try:
            context = build_context(live_index, smell)
            system_prompt = build_system_prompt(context.fence or "source")
            base_user_prompt = build_user_prompt(context)
            file_path = smell.file if smell.file.is_absolute() else repo_root / smell.file

            pre_source = file_path.read_text(encoding="utf-8") if apply else None

            proposal, attempts = _propose_and_apply(
                provider, system_prompt, base_user_prompt, file_path, smell, apply, max_attempts
            )

            if apply and verify_after is not None and not verify_after():
                # Passed every static guard but regressed the suite -- a silent
                # behaviour change (severed def-use chain, altered branch). Restore
                # the pre-edit source and skip: do no harm.
                assert pre_source is not None
                file_path.write_text(pre_source, encoding="utf-8")
                live_index = index_repository(repo_root)
                print(
                    f"Warning: reverting {smell.file}:{smell.line}: edit regressed the test suite",
                    file=sys.stderr,
                )
                continue

            results.append(
                RefactorResult(context=context, proposal=proposal, applied=apply, attempts=attempts)
            )

            if apply:
                # A landed edit can add or remove lines (a hoisted constant, an
                # extracted helper), shifting every later smell in the same file off
                # its snapshotted line -- after which line-keyed logic (the
                # deterministic magic-number extractor) can't find the literal, and
                # only the first smell per file ever gets fixed.
                assert pre_source is not None
                _realign_pending_lines(smells[position + 1 :], file_path, repo_root, pre_source, proposal)
                live_index = index_repository(repo_root)
        except Exception as exc:
            hint = " (re-run `refract index` to refresh)" if "not found" in str(exc) else ""
            print(f"Warning: skipping {smell.file}:{smell.line}: {exc}{hint}", file=sys.stderr)

    return results


def _realign_pending_lines(
    pending: list[SmellLocation],
    edited_file: Path,
    repo_root: Path,
    pre_source: str,
    proposal: RefactorProposal,
) -> None:
    """Shift the ``line`` of each pending smell in ``edited_file`` to account for
    lines the just-applied ``proposal`` inserted or removed above it.

    ``pre_source`` is the file just before the edit: each edit's ``old_snippet``
    locates where it landed, and ``new_snippet``'s newline delta says how far rows
    below moved. Deltas are cumulative, in the file's current coordinates, matching
    the smells' already-realigned lines. A no-net-line-change edit shifts nothing."""
    shifts = []
    for edit in proposal.edits:
        idx = pre_source.find(edit.old_snippet)
        if idx < 0:
            continue
        delta = edit.new_snippet.count("\n") - edit.old_snippet.count("\n")
        if delta:
            shifts.append((pre_source.count("\n", 0, idx) + 1, delta))
    if not shifts:
        return
    for smell in pending:
        smell_path = smell.file if smell.file.is_absolute() else repo_root / smell.file
        if smell_path != edited_file:
            continue
        for at_line, delta in shifts:
            if smell.line > at_line:
                smell.line += delta


def _propose_and_apply(
    provider: RefactorProvider,
    system_prompt: str,
    base_user_prompt: str,
    file_path: Path,
    smell: SmellLocation,
    apply: bool,
    max_attempts: int,
) -> tuple[RefactorProposal, int]:
    """Ask the provider, apply, and on a recoverable rejection re-ask with the
    error fed back -- up to max_attempts inference calls.

    A rejected edit raises before anything is written (validation and the syntax
    guard both run before the disk write), so each retry re-reads the pristine
    file: no rollback is needed, and after the last attempt the caller's handler
    just skips the target, leaving it untouched. Only ValueErrors are retried
    (bad snippet / ambiguous match / unparseable patch, or -- for a magic number
    -- a missing name / unlocatable literal raised by ``_effective_proposal``); a
    provider or network error propagates straight out to be skipped.

    For a magic-number fix the model only names the constant; ``_effective_proposal``
    swaps in a deterministically-built extraction (see extract_constant.py) that is
    guaranteed to match, so those targets never depend on the model reproducing the
    source verbatim. No usable name raises a recoverable error and the retry re-asks
    for one -- the model's raw edits are never applied to a magic number.
    """
    user_prompt = base_user_prompt
    last_error: ValueError | None = None
    attempts = max(1, max_attempts)

    rename_of = smell.identifier if smell.smell is SmellType.LONG_IDENTIFIER else None

    for attempt in range(1, attempts + 1):
        model_proposal = provider.propose(system_prompt, user_prompt)
        try:
            proposal = _effective_proposal(smell, model_proposal, file_path)
            apply_snippet_replacement(file_path, proposal, apply, rename_of=rename_of)
            _debug_dump(file_path, attempt, proposal, None)
            return proposal, attempt
        except ValueError as exc:
            _debug_dump(file_path, attempt, model_proposal, str(exc))
            last_error = exc
            print(
                f"Warning: {file_path}: attempt {attempt}/{attempts} rejected: {exc}",
                file=sys.stderr,
            )
            if attempt < attempts:
                user_prompt = build_repair_prompt(base_user_prompt, model_proposal, str(exc))

    assert last_error is not None  # the loop ran at least once and never returned
    raise last_error


def _effective_proposal(
    smell: SmellLocation, proposal: RefactorProposal, file_path: Path
) -> RefactorProposal:
    """The proposal refract will actually apply.

    For every smell except magic_number this is the model's own proposal. A
    magic_number is always fixed by refract's deterministic extractor
    (extract_constant.py); the model's only job is to name the constant. Its raw
    ``edits`` are never applied, because a model rewriting the use site can mutate
    the host expression -- injecting a ``/60`` that is valid, smell-reducing, and
    silently wrong. So no usable name, or a literal that can't be located for a
    clean swap, raises a recoverable ValueError: the caller re-asks for a name and
    skips the target only if the model never complies. A withheld fix beats a broken
    one."""
    if smell.smell is not SmellType.MAGIC_NUMBER:
        return proposal

    name = proposal.constant_name.strip()
    deterministic = build_constant_extraction(file_path, smell, name) if name else None
    if deterministic is None:
        reason = (
            "the flagged literal could not be located for automatic extraction"
            if name
            else "constant_name was empty"
        )
        raise ValueError(
            "A magic-number fix must be performed by Refract's own extractor, not by "
            "your edits. Put a valid UPPER_SNAKE_CASE name for the flagged literal in "
            "the `constant_name` field and do NOT alter the surrounding expression -- "
            f"Refract adds the definition and rewrites the use site itself ({reason})."
        )
    return deterministic


def _debug_dump(
    file_path: Path, attempt: int, proposal: RefactorProposal, error: str | None
) -> None:
    """Append each proposal + outcome to REFRACT_DEBUG_DUMP (JSONL) when set. Diagnostics only."""
    path = os.getenv("REFRACT_DEBUG_DUMP")
    if not path:
        return
    record = {
        "file": str(file_path),
        "attempt": attempt,
        "error": error,
        "edits": [{"old": e.old_snippet, "new": e.new_snippet} for e in proposal.edits],
    }
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
