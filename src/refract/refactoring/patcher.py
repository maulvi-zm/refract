from __future__ import annotations

from pathlib import Path

from refract.refactoring.proposal import RefactorProposal, validate_replacement


def apply_snippet_replacement(file_path: Path, proposal: RefactorProposal, apply: bool) -> str:
    """Return the patched text, writing to disk only when apply is True."""
    source = file_path.read_text(encoding="utf-8")
    validate_replacement(source, proposal)

    updated = source.replace(proposal.old_snippet, proposal.new_snippet, 1)
    if apply:
        file_path.write_text(updated, encoding="utf-8")

    return updated
