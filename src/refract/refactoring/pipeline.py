from __future__ import annotations

import json
import os
import sys
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
) -> list[RefactorResult]:
    smells = sorted(
        index.smells_by_type(smell_type),
        key=lambda s: (str(s.file), s.line, s.identifier),
    )[: max(limit, 0)]

    live_index = index
    results: list[RefactorResult] = []

    for smell in smells:
        try:
            context = build_context(live_index, smell)
            system_prompt = build_system_prompt(context.fence or "source")
            base_user_prompt = build_user_prompt(context)
            file_path = smell.file if smell.file.is_absolute() else repo_root / smell.file

            proposal, attempts = _propose_and_apply(
                provider, system_prompt, base_user_prompt, file_path, smell, apply, max_attempts
            )
            results.append(
                RefactorResult(context=context, proposal=proposal, applied=apply, attempts=attempts)
            )

            if apply:
                live_index = index_repository(repo_root)
        except Exception as exc:
            hint = " (re-run `refract index` to refresh)" if "not found" in str(exc) else ""
            print(f"Warning: skipping {smell.file}:{smell.line}: {exc}{hint}", file=sys.stderr)

    return results


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
    just skips the target, leaving it untouched. Only ValueErrors from applying
    are retried (bad snippet / ambiguous match / unparseable patch); a provider
    or network error propagates straight out to be skipped.

    For a magic-number fix the model only names the constant; ``_effective_proposal``
    then swaps in a deterministically-built extraction (see extract_constant.py)
    that is guaranteed to match, so those targets don't depend on the model
    reproducing the source verbatim and never hit the retry loop.
    """
    user_prompt = base_user_prompt
    last_error: ValueError | None = None
    attempts = max(1, max_attempts)

    for attempt in range(1, attempts + 1):
        proposal = _effective_proposal(smell, provider.propose(system_prompt, user_prompt), file_path)
        try:
            apply_snippet_replacement(file_path, proposal, apply)
            _debug_dump(file_path, attempt, proposal, None)
            return proposal, attempt
        except ValueError as exc:
            _debug_dump(file_path, attempt, proposal, str(exc))
            last_error = exc
            print(
                f"Warning: {file_path}: attempt {attempt}/{attempts} rejected: {exc}",
                file=sys.stderr,
            )
            if attempt < attempts:
                user_prompt = build_repair_prompt(base_user_prompt, proposal, str(exc))

    assert last_error is not None  # the loop ran at least once and never returned
    raise last_error


def _effective_proposal(
    smell: SmellLocation, proposal: RefactorProposal, file_path: Path
) -> RefactorProposal:
    """Prefer a deterministic extraction for a named magic-number constant, else
    keep the model's own edits. Falls back to the model whenever the literal
    can't be located unambiguously, so nothing is lost by trying."""
    if smell.smell is SmellType.MAGIC_NUMBER and proposal.constant_name.strip():
        deterministic = build_constant_extraction(file_path, smell, proposal.constant_name.strip())
        if deterministic is not None:
            return deterministic
    return proposal


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
