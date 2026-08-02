from __future__ import annotations

import re

import tree_sitter_java
from tree_sitter import Language, Node

from refract.indexing.parser import node_text
from refract.languages.base import LanguageSpec

_LANGUAGE = Language(tree_sitter_java.language())


def _is_constant_definition(number: Node, source: bytes) -> bool:
    # true when the number sits inside a `static final` field
    current = number.parent

    while current is not None:
        if current.type == "field_declaration":
            modifiers = next((c for c in current.children if c.type == "modifiers"), None)
            text = node_text(modifiers, source) if modifiers else ""
            return "static" in text and "final" in text
        current = current.parent

    return False


JAVA = LanguageSpec(
    name="java",
    extensions=(".java",),
    language=_LANGUAGE,
    fence="java",
    function_query="""
        (method_declaration
            name: (identifier) @function.name
            body: (block) @function.body) @function.def
    """,
    enclosing_class_types=frozenset(
        {"class_declaration", "interface_declaration", "enum_declaration", "record_declaration"}
    ),
    class_name_field="name",
    parameter_query="(formal_parameter name: (identifier) @param.name)",
    call_query="(method_invocation name: (identifier) @call.name)",
    decision_node_types=frozenset(
        {
            "if_statement",
            "for_statement",
            "enhanced_for_statement",
            "while_statement",
            "do_statement",
            "catch_clause",
            "switch_label",
            "ternary_expression",
        }
    ),
    logical_binary_types=frozenset({"binary_expression"}),
    logical_operators=frozenset({"&&", "||"}),
    statement_query="""
        [
            (expression_statement)
            (local_variable_declaration)
            (if_statement)
            (for_statement)
            (enhanced_for_statement)
            (while_statement)
            (do_statement)
            (return_statement)
            (switch_expression)
            (try_statement)
            (throw_statement)
            (synchronized_statement)
            (yield_statement)
            (break_statement)
            (continue_statement)
            (assert_statement)
            (labeled_statement)
        ] @statement
    """,
    long_method_threshold=20,
    declaration_query="""
        (variable_declarator name: (identifier) @decl.name)
        (formal_parameter name: (identifier) @decl.name)
    """,
    long_identifier_threshold=17,
    number_query="""
        [
            (decimal_integer_literal)
            (hex_integer_literal)
            (octal_integer_literal)
            (binary_integer_literal)
            (decimal_floating_point_literal)
            (hex_floating_point_literal)
        ] @number
    """,
    ignored_numbers=frozenset({"0", "1", "2"}),
    is_constant_definition=_is_constant_definition,
    constant_pattern=re.compile(
        r"\b(?:public|private|protected)?\s*static\s+final\s+[^=;]+=[^;]+;"
    ),
)
