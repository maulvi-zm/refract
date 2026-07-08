from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from refract.core.models import RepositoryIndex, SmellType
from refract.indexing.repository import index_repository
from refract.planning.context import RefactorContext, build_context
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
                provider, system_prompt, base_user_prompt, file_path, apply, max_attempts
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
    """
    user_prompt = base_user_prompt
    last_error: ValueError | None = None
    attempts = max(1, max_attempts)

    for attempt in range(1, attempts + 1):
        proposal = provider.propose(system_prompt, user_prompt)
        try:
            apply_snippet_replacement(file_path, proposal, apply)
            return proposal, attempt
        except ValueError as exc:
            last_error = exc
            print(
                f"Warning: {file_path}: attempt {attempt}/{attempts} rejected: {exc}",
                file=sys.stderr,
            )
            if attempt < attempts:
                user_prompt = build_repair_prompt(base_user_prompt, proposal, str(exc))

    assert last_error is not None  # the loop ran at least once and never returned
    raise last_error
