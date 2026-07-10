"""Deterministic extract-constant for magic numbers.

The model only picks a name; refract does the mechanical part. Because a magic
number smell already carries the file, the line, and the literal text, refract
can re-find the exact tree-sitter node itself, ask the language spec for a valid
insertion point (module scope in Python, inside the enclosing class in Java),
and build the two edits (define the constant, rewrite the use site) straight
from the real source -- so the snippets match by construction.

This removes the one failure the snippet approach can't tune away: asking the
model to reproduce a multi-line, decorator-dense span character-for-character.
When the literal can't be located unambiguously the builder returns None and the
caller falls back to the model's own edits, so do-no-harm is preserved.
"""

from __future__ import annotations

import keyword
import re
from pathlib import Path

from tree_sitter import Node

from refract.core.models import SmellLocation
from refract.indexing.parser import compile_query, matches, node_text, parse
from refract.languages.base import ConstantPlan
from refract.languages.registry import spec_for_path
from refract.refactoring.patcher import apply_snippet_replacement
from refract.refactoring.proposal import RefactorProposal, SnippetEdit

_MAX_WINDOW_LINES = 8  # how far a snippet window may grow reaching for uniqueness


def build_constant_extraction(
    file_path: Path, smell: SmellLocation, const_name: str
) -> RefactorProposal | None:
    """Build the edits to hoist ``smell``'s literal into a named constant.

    Returns a proposal proven to apply cleanly (validated by a dry run through
    the normal patcher, syntax guard included), or None when the fix can't be
    made deterministically -- an unknown language, a bad name, a literal that
    isn't uniquely locatable, or one the language spec can't place a constant
    for (e.g. a Java literal outside any plain class). The placement is
    per-language via ``spec.plan_constant``; the edits are language-agnostic.
    """
    spec = spec_for_path(file_path)
    if spec is None or not _is_valid_name(const_name):
        return None

    source = file_path.read_text(encoding="utf-8")
    data = source.encode("utf-8")
    root = parse(data, spec.language)

    literal = _locate_literal(root, data, spec, smell)
    if literal is None:
        return None

    plan = spec.plan_constant(root, literal, data, const_name)
    if plan is None:
        return None

    # An earlier target in the same file may already have hoisted this literal
    # under this name (the model names constants independently, so two equal
    # literals often collide on one name). Re-adding the definition would create
    # a duplicate declaration -- a compile error in Java that tree-sitter still
    # parses without an ERROR, so the syntax guard misses it. If the identical
    # definition is already present, reuse it (rewrite the use site only); if the
    # name is taken by a *different* definition, decline to the model rather than
    # bind the use site to the wrong value.
    existing = _existing_definition(source, plan.text, const_name)
    if existing == "conflict":
        return None

    use_site = _use_site_edit(source, data, literal, const_name)
    if use_site is None:
        return None

    if existing == "present":
        edits: tuple[SnippetEdit, ...] = (use_site,)
    else:
        definition = _definition_edit(source, data, plan)
        if definition is None:
            return None
        edits = (definition, use_site)

    value = node_text(literal, data)
    proposal = RefactorProposal(
        explanation=f"Extract magic number {value} into constant {const_name}.",
        edits=edits,
        confidence=1.0,
        constant_name=const_name,
    )
    try:
        apply_snippet_replacement(file_path, proposal, apply=False)
    except ValueError:
        return None
    return proposal


def _is_valid_name(name: str) -> bool:
    return name.isidentifier() and not keyword.iskeyword(name)


def _existing_definition(source: str, definition_text: str, name: str) -> str:
    """Whether ``name`` is already defined in ``source``.

    Returns ``"present"`` when the exact ``definition_text`` line is already
    there (an earlier target hoisted the same literal under the same name --
    reuse it), ``"conflict"`` when ``name`` is assigned by some *other*
    definition (extracting under it would bind the use site to a different
    value -- decline), and ``"absent"`` otherwise. The assignment probe matches
    a definition of ``name`` (``NAME =``, ``NAME: T =``, or ``... NAME =``) while
    excluding ``==`` and uses where ``name`` is on the right-hand side.
    """
    if definition_text.strip() in source:
        return "present"
    if re.search(rf"\b{re.escape(name)}\b\s*(?::[^=\n]+)?=(?!=)", source):
        return "conflict"
    return "absent"


def _locate_literal(root: Node, data: bytes, spec, smell: SmellLocation) -> Node | None:
    """The single numeric-literal node matching the smell's line and text.

    Mirrors the detector's own filtering (ignored numbers, existing constant
    definitions) so we re-find exactly the node that was flagged. Returns None
    unless precisely one candidate matches, so an ambiguous line falls back to
    the model rather than risk touching the wrong literal.
    """
    number_query = compile_query(spec.language, spec.number_query)
    hits = [
        node
        for caps in matches(number_query, root)
        for node in caps.get("number", [])
        if node.start_point[0] + 1 == smell.line
        and node_text(node, data) == smell.identifier
        and not spec.is_constant_definition(node, data)
    ]
    return hits[0] if len(hits) == 1 else None


def _definition_edit(source: str, data: bytes, plan: ConstantPlan) -> SnippetEdit | None:
    """Append the planned definition right after the anchor line, keying the edit
    on a unique window ending at that line so it lands where the spec intends."""
    start, end = _row_bounds(data, plan.anchor_row)
    window = _unique_window(source, data, start, end)
    if window is None:
        return None
    return SnippetEdit(old_snippet=window, new_snippet=f"{window}\n{plan.indent}{plan.text}")


def _use_site_edit(source: str, data: bytes, literal: Node, name: str) -> SnippetEdit | None:
    """Replace just the flagged literal with ``name``, keying the edit on the
    smallest whole-line window around it that is unique in the file."""
    window = _unique_window(source, data, literal.start_byte, literal.end_byte)
    if window is None:
        return None
    win_start = _find_window_start(source, data, literal.start_byte, literal.end_byte, window)
    if win_start is None:
        return None
    rel = literal.start_byte - win_start
    lit_len = literal.end_byte - literal.start_byte
    old_bytes = data[win_start : win_start + len(window.encode("utf-8"))]
    new = (old_bytes[:rel] + name.encode("utf-8") + old_bytes[rel + lit_len :]).decode("utf-8")
    return SnippetEdit(old_snippet=window, new_snippet=new)


def _unique_window(source: str, data: bytes, start: int, end: int) -> str | None:
    """Grow a whole-line byte window around [start, end) until its text occurs
    exactly once in the source, or give up after a few lines."""
    win_start, win_end = _line_bounds(data, start, end)
    for _ in range(_MAX_WINDOW_LINES):
        text = data[win_start:win_end].decode("utf-8")
        if source.count(text) == 1:
            return text
        grown_start, grown_end = _grow(data, win_start, win_end)
        if (grown_start, grown_end) == (win_start, win_end):
            break  # hit both file edges, can't grow further
        win_start, win_end = grown_start, grown_end
    return None


def _find_window_start(source: str, data: bytes, start: int, end: int, window: str) -> int | None:
    """Byte offset where ``window`` begins -- recompute the same line growth used
    to build it so the literal's relative offset can be sliced out exactly."""
    win_start, win_end = _line_bounds(data, start, end)
    for _ in range(_MAX_WINDOW_LINES):
        if data[win_start:win_end].decode("utf-8") == window:
            return win_start
        grown_start, grown_end = _grow(data, win_start, win_end)
        if (grown_start, grown_end) == (win_start, win_end):
            break
        win_start, win_end = grown_start, grown_end
    return None


def _row_bounds(data: bytes, row: int) -> tuple[int, int]:
    lines = data.split(b"\n")
    start = sum(len(line) + 1 for line in lines[:row])
    return start, start + len(lines[row])


def _line_bounds(data: bytes, start: int, end: int) -> tuple[int, int]:
    line_start = data.rfind(b"\n", 0, start) + 1
    newline = data.find(b"\n", end)
    line_end = newline if newline != -1 else len(data)
    return line_start, line_end


def _grow(data: bytes, start: int, end: int) -> tuple[int, int]:
    new_start = data.rfind(b"\n", 0, start - 1) + 1 if start > 0 else start
    newline = data.find(b"\n", end + 1)
    new_end = newline if newline != -1 else len(data)
    return new_start, new_end
