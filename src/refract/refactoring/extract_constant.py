"""Deterministic extract-constant for magic numbers.

The model only picks a name; refract does the mechanical part. Because a magic
number smell already carries the file, the line, and the literal text, refract
can re-find the exact tree-sitter node itself, compute a valid module-level
insertion point, and build the two edits (define the constant, rewrite the use
site) straight from the real source -- so the snippets match by construction.

This removes the one failure the snippet approach can't tune away: asking the
model to reproduce a multi-line, decorator-dense span character-for-character.
When the literal can't be located unambiguously the builder returns None and the
caller falls back to the model's own edits, so do-no-harm is preserved.
"""

from __future__ import annotations

import keyword
from pathlib import Path

from tree_sitter import Node

from refract.core.models import SmellLocation
from refract.indexing.parser import compile_query, matches, node_text, parse
from refract.languages.registry import spec_for_path
from refract.refactoring.patcher import apply_snippet_replacement
from refract.refactoring.proposal import RefactorProposal, SnippetEdit

# Top-level nodes a new constant may safely be placed after: the module docstring
# and comments/imports. We anchor on the last of these and stop at the first real
# statement, so the definition always lands at module scope.
_IMPORT_TYPES = frozenset(
    {"import_statement", "import_from_statement", "future_import_statement"}
)
_MAX_WINDOW_LINES = 8  # how far a snippet window may grow reaching for uniqueness


def build_constant_extraction(
    file_path: Path, smell: SmellLocation, const_name: str
) -> RefactorProposal | None:
    """Build the edits to hoist ``smell``'s literal into a module constant.

    Returns a proposal proven to apply cleanly (validated by a dry run through
    the normal patcher, syntax guard included), or None when the fix can't be
    made deterministically -- an unknown language, a bad name, a literal that
    isn't uniquely locatable, or a literal that already lives in the header.
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

    anchor_row = _header_anchor_row(root)
    if anchor_row is None or literal.start_point[0] <= anchor_row:
        # no safe module-level anchor, or the literal sits inside the header
        return None

    value = node_text(literal, data)
    definition = _definition_edit(source, data, anchor_row, const_name, value)
    use_site = _use_site_edit(source, data, literal, const_name)
    if definition is None or use_site is None:
        return None

    proposal = RefactorProposal(
        explanation=f"Extract magic number {value} into constant {const_name}.",
        edits=(definition, use_site),
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


def _locate_literal(
    root: Node, data: bytes, spec, smell: SmellLocation
) -> Node | None:
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


def _header_anchor_row(root: Node) -> int | None:
    """0-based row to insert the constant after: the last leading comment,
    module docstring, or import. None when the file opens with real code."""
    anchor: int | None = None
    for index, child in enumerate(root.children):
        if child.type in _IMPORT_TYPES or child.type == "comment":
            anchor = child.end_point[0]
        elif index == 0 and _is_docstring(child):
            anchor = child.end_point[0]
        else:
            break
    return anchor


def _is_docstring(node: Node) -> bool:
    return (
        node.type == "expression_statement"
        and node.child_count > 0
        and node.children[0].type == "string"
    )


def _definition_edit(
    source: str, data: bytes, anchor_row: int, name: str, value: str
) -> SnippetEdit | None:
    """Append ``name = value`` right after the anchor line, keying the edit on a
    unique window ending at that line so it lands at module scope."""
    start, end = _row_bounds(data, anchor_row)
    window = _unique_window(source, data, start, end)
    if window is None:
        return None
    old = window
    return SnippetEdit(old_snippet=old, new_snippet=f"{old}\n{name} = {value}")


def _use_site_edit(
    source: str, data: bytes, literal: Node, name: str
) -> SnippetEdit | None:
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
