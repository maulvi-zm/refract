from __future__ import annotations

from functools import lru_cache

from tree_sitter import Language, Node, Parser, Query, QueryCursor


def parse(source: bytes, language: Language) -> Node:
    return Parser(language).parse(source).root_node


@lru_cache(maxsize=None)
def compile_query(language: Language, pattern: str) -> Query:
    return Query(language, pattern)


def matches(query: Query, node: Node) -> list[dict[str, list[Node]]]:
    return [captures for _, captures in QueryCursor(query).matches(node)]


def node_text(node: Node, source: bytes) -> str:
    return source[node.start_byte : node.end_byte].decode("utf-8")
