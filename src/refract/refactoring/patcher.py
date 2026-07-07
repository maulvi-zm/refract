from __future__ import annotations

from pathlib import Path

from refract.indexing.parser import parse
from refract.languages.registry import spec_for_path
from refract.refactoring.proposal import RefactorProposal, validate_replacement


def apply_snippet_replacement(file_path: Path, proposal: RefactorProposal, apply: bool) -> str:
    """Return the patched text, writing to disk only when apply is True.

    Applies the proposal's edits in order, each against the running text (so an
    earlier edit can change what a later one matches), then refuses the whole
    patch if it introduces a syntax error the pristine file didn't have. Do no
    harm: a broken edit raises and nothing is written, so the caller skips the
    target and leaves the file exactly as it was (see the improvement plan's
    guiding principle -- a non-fix beats a broken fix).
    """
    source = file_path.read_text(encoding="utf-8")
    validate_replacement(source, proposal)

    updated = source
    for edit in proposal.edits:
        occurrences = updated.count(edit.old_snippet)
        if occurrences == 0:
            raise ValueError("Provider old_snippet was not found after preceding edits")
        if occurrences > 1:
            raise ValueError("Provider old_snippet is ambiguous after preceding edits")
        updated = updated.replace(edit.old_snippet, edit.new_snippet, 1)

    _reject_if_syntax_broken(file_path, source, updated)

    if apply:
        file_path.write_text(updated, encoding="utf-8")

    return updated


def _reject_if_syntax_broken(file_path: Path, source: str, updated: str) -> None:
    """Raise if the patched text parses with an error the original didn't.

    tree-sitter is error-tolerant -- it always yields a tree, flagging broken
    regions with ERROR nodes rather than raising -- so ``has_error`` is the only
    thing that notices a patch that no longer parses. Checked against the
    pristine text so a file that already parsed oddly isn't blamed on this edit.
    """
    spec = spec_for_path(file_path)
    if spec is None:
        return  # unknown language: nothing to check against

    before_broken = parse(source.encode("utf-8"), spec.language).has_error
    after_broken = parse(updated.encode("utf-8"), spec.language).has_error
    if after_broken and not before_broken:
        raise ValueError("edit introduces a syntax error; skipping to avoid breaking the file")
