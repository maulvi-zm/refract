from __future__ import annotations

from pathlib import Path

from tree_sitter import Node

from refract.core.models import MethodInfo
from refract.indexing.parser import compile_query, matches, node_text, parse
from refract.languages.base import LanguageSpec


def extract_methods(path: Path, spec: LanguageSpec) -> list[MethodInfo]:
    source = path.read_bytes()
    root = parse(source, spec.language)

    function_query = compile_query(spec.language, spec.function_query)
    parameter_query = compile_query(spec.language, spec.parameter_query)
    call_query = compile_query(spec.language, spec.call_query)

    methods: list[MethodInfo] = []

    for match in matches(function_query, root):
        definition = match["function.def"][0]
        name_node = match["function.name"][0]
        body_node = match["function.body"][0]

        parameters = [
            node_text(n, source)
            for caps in matches(parameter_query, definition)
            for n in caps.get("param.name", [])
        ]
        calls = [
            node_text(n, source)
            for caps in matches(call_query, body_node)
            for n in caps.get("call.name", [])
        ]

        methods.append(
            MethodInfo(
                name=node_text(name_node, source),
                class_name=_enclosing_class(definition, source, spec),
                file=path,
                start_line=name_node.start_point[0] + 1,  # rows are 0-based
                end_line=body_node.end_point[0] + 1,
                cyclomatic_complexity=_cyclomatic_complexity(body_node, source, spec),
                parameters=parameters,
                calls=calls,
            )
        )

    return methods


def _enclosing_class(node: Node, source: bytes, spec: LanguageSpec) -> str:
    current = node.parent
    while current is not None:
        if current.type in spec.enclosing_class_types:
            name_node = current.child_by_field_name(spec.class_name_field)
            if name_node is not None:
                return node_text(name_node, source)
        current = current.parent

    return "<unknown>"


def _cyclomatic_complexity(body: Node, source: bytes, spec: LanguageSpec) -> int:
    # McCabe: 1 plus one per decision point
    count = 1
    stack = [body]

    while stack:
        current = stack.pop()

        if current.type in spec.decision_node_types:
            count += 1
        elif current.type in spec.logical_binary_types:
            operator = current.child_by_field_name("operator")
            if operator and node_text(operator, source) in spec.logical_operators:
                count += 1

        stack.extend(current.children)

    return count
