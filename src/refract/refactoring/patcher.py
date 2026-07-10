from __future__ import annotations

from pathlib import Path

from refract.indexing.parser import parse
from refract.languages.registry import spec_for_path
from refract.refactoring.proposal import RefactorProposal, validate_replacement


def apply_snippet_replacement(
    file_path: Path,
    proposal: RefactorProposal,
    apply: bool,
    rename_of: str | None = None,
) -> str:
    """Return the patched text, writing to disk only when apply is True.

    Applies the proposal's edits in order, each against the running text (so an
    earlier edit can change what a later one matches), then refuses the whole
    patch if it introduces a syntax error the pristine file didn't have. Do no
    harm: a broken edit raises and nothing is written, so the caller skips the
    target and leaves the file exactly as it was (see the improvement plan's
    guiding principle -- a non-fix beats a broken fix).

    ``rename_of`` names the identifier a long-identifier fix is renaming away; the
    patch is rejected if any occurrence of it survives (an incomplete rename that
    parses but leaves a dangling reference).
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
    _reject_if_misplaced_root_member(file_path, source, updated)
    _reject_if_new_dead_code(file_path, source, updated)
    if rename_of is not None:
        _reject_if_identifier_remains(file_path, updated, rename_of)

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


def _reject_if_misplaced_root_member(file_path: Path, source: str, updated: str) -> None:
    """Raise if the patch introduces a declaration illegal at compilation-unit
    scope (see ``LanguageSpec.invalid_root_child_types``).

    tree-sitter parses e.g. a Java ``static final`` field hoisted to file scope
    as a valid ``local_variable_declaration`` under ``program`` -- no ERROR node,
    so ``_reject_if_syntax_broken`` waves it through even though it won't compile.
    Compare the count of such direct root children before and after so a genuine
    do-no-harm violation is caught while any pre-existing oddity isn't blamed on
    this edit.
    """
    spec = spec_for_path(file_path)
    if spec is None or not spec.invalid_root_child_types:
        return

    def _misplaced(text: str) -> int:
        root = parse(text.encode("utf-8"), spec.language)
        return sum(1 for c in root.children if c.type in spec.invalid_root_child_types)

    if _misplaced(updated) > _misplaced(source):
        raise ValueError(
            "edit places a declaration outside any type body; skipping to avoid breaking the file"
        )


def _reject_if_new_dead_code(file_path: Path, source: str, updated: str) -> None:
    """Raise if the patch adds unreachable code after a control-flow terminator
    (see ``LanguageSpec.dead_code_terminators``).

    A botched extract-method splits a function by dropping the extracted ``def``
    mid-body: the original function is truncated (silently returns None) and its
    tail is orphaned as statements sitting after the helper's ``return``. That
    parses cleanly and the target method looks shorter, so neither the syntax nor
    the smell check notices -- but the behaviour is broken. We flag a terminator
    (return / raise / throw) that is a direct block child with a later non-comment
    sibling, and reject only when the patch introduces more of them than the
    pristine file had (so pre-existing oddities aren't blamed on this edit).
    """
    spec = spec_for_path(file_path)
    if spec is None or not spec.dead_code_terminators:
        return

    def _dead_code_blocks(text: str) -> int:
        count = 0
        stack = [parse(text.encode("utf-8"), spec.language)]
        while stack:
            node = stack.pop()
            named = [c for c in node.children if c.is_named]
            for i, child in enumerate(named):
                if child.type in spec.dead_code_terminators and any(
                    "comment" not in sib.type for sib in named[i + 1 :]
                ):
                    count += 1
            stack.extend(node.children)
        return count

    if _dead_code_blocks(updated) > _dead_code_blocks(source):
        raise ValueError(
            "edit leaves unreachable code after a return/raise; skipping to avoid breaking the file"
        )


def _reject_if_identifier_remains(file_path: Path, updated: str, old_identifier: str) -> None:
    """Raise if ``old_identifier`` still occurs as an identifier token in the
    patched text -- an incomplete rename.

    A long-identifier fix must rename EVERY occurrence; when the model changes the
    declaration but misses a use (e.g. it only saw part of the file), the leftover
    reference still parses but is now dangling -- a NameError / AttributeError at
    runtime, invisible to the syntax check. Comparing identifier tokens (not raw
    text) ignores the name inside strings and comments, which a rename leaves
    alone on purpose. Conservative: if the same spelling names an unrelated
    binding elsewhere this over-rejects, but a safe skip beats a broken rename.
    """
    spec = spec_for_path(file_path)
    if spec is None:
        return

    data = updated.encode("utf-8")
    target = old_identifier.encode("utf-8")
    stack = [parse(data, spec.language)]
    while stack:
        node = stack.pop()
        if node.type == "identifier" and data[node.start_byte : node.end_byte] == target:
            raise ValueError(
                f"rename left an occurrence of '{old_identifier}' behind; "
                "skipping incomplete rename to avoid a dangling reference"
            )
        stack.extend(node.children)
