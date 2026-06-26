from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from refract.core.models import RepositoryIndex, SmellType
from refract.indexing.repository import index_repository
from refract.planning.context import RefactorContext, build_context
from refract.refactoring.patcher import apply_snippet_replacement
from refract.refactoring.prompt import build_system_prompt, build_user_prompt
from refract.refactoring.proposal import RefactorProposal, RefactorProvider


@dataclass(frozen=True)
class RefactorResult:
    context: RefactorContext
    proposal: RefactorProposal
    applied: bool


def run_refactor(
    index: RepositoryIndex,
    repo_root: Path,
    smell_type: SmellType,
    limit: int,
    provider: RefactorProvider,
    apply: bool,
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
            proposal = provider.propose(
                build_system_prompt(context.fence or "source"),
                build_user_prompt(context),
            )

            file_path = smell.file if smell.file.is_absolute() else repo_root / smell.file
            apply_snippet_replacement(file_path, proposal, apply)
            results.append(RefactorResult(context=context, proposal=proposal, applied=apply))

            if apply:
                live_index = index_repository(repo_root)
        except Exception as exc:
            hint = " (re-run `refract index` to refresh)" if "not found" in str(exc) else ""
            print(f"Warning: skipping {smell.file}:{smell.line}: {exc}{hint}", file=sys.stderr)

    return results
