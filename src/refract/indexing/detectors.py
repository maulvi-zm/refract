from __future__ import annotations

from pathlib import Path

from tree_sitter import Node

from refract.core.models import SmellLocation, SmellType
from refract.indexing.parser import compile_query, matches, node_text, parse
from refract.languages.base import LanguageSpec

# type suffixes on literals like 10L or 1.5f
_NUMBER_SUFFIXES = "lfd"


def detect_smells(path: Path, spec: LanguageSpec) -> list[SmellLocation]:
    source = path.read_bytes()
    root = parse(source, spec.language)

    return [
        *_long_methods(root, source, path, spec),
        *_long_identifiers(root, source, path, spec),
        *_magic_numbers(root, source, path, spec),
    ]


def _long_methods(root: Node, source: bytes, path: Path, spec: LanguageSpec) -> list[SmellLocation]:
    function_query = compile_query(spec.language, spec.function_query)
    statement_query = compile_query(spec.language, spec.statement_query)

    smells: list[SmellLocation] = []

    for match in matches(function_query, root):
        name_node = match["function.name"][0]
        body_node = match["function.body"][0]

        statements = len(matches(statement_query, body_node))
        if statements <= spec.long_method_threshold:
            continue

        name = node_text(name_node, source)
        smells.append(
            SmellLocation(
                smell=SmellType.LONG_METHOD,
                file=path,
                line=name_node.start_point[0] + 1,
                identifier=name,
                detail=(
                    f"Method '{name}' has {statements} statements "
                    f"(threshold {spec.long_method_threshold})."
                ),
            )
        )

    return smells


def _long_identifiers(
    root: Node, source: bytes, path: Path, spec: LanguageSpec
) -> list[SmellLocation]:
    declaration_query = compile_query(spec.language, spec.declaration_query)

    smells: list[SmellLocation] = []
    seen: set[tuple[int, str]] = set()

    for node in _captured_nodes(declaration_query, root, "decl.name"):
        name = node_text(node, source)
        line = node.start_point[0] + 1

        if len(name) <= spec.long_identifier_threshold or (line, name) in seen:
            continue
        seen.add((line, name))

        smells.append(
            SmellLocation(
                smell=SmellType.LONG_IDENTIFIER,
                file=path,
                line=line,
                identifier=name,
                detail=(
                    f"Identifier '{name}' is {len(name)} characters "
                    f"(threshold {spec.long_identifier_threshold})."
                ),
            )
        )

    return smells


def _magic_numbers(
    root: Node, source: bytes, path: Path, spec: LanguageSpec
) -> list[SmellLocation]:
    number_query = compile_query(spec.language, spec.number_query)

    smells: list[SmellLocation] = []

    for node in _captured_nodes(number_query, root, "number"):
        literal = node_text(node, source)

        if _normalize_number(literal) in spec.ignored_numbers:
            continue

        smells.append(
            SmellLocation(
                smell=SmellType.MAGIC_NUMBER,
                file=path,
                line=node.start_point[0] + 1,
                identifier=literal,
                detail=f"Magic number '{literal}' should be a named constant.",
            )
        )

    return smells


def _captured_nodes(query_pattern, root: Node, capture: str) -> list[Node]:
    return [n for caps in matches(query_pattern, root) for n in caps.get(capture, [])]


def _normalize_number(literal: str) -> str:
    return literal.replace("_", "").rstrip(_NUMBER_SUFFIXES + _NUMBER_SUFFIXES.upper())
