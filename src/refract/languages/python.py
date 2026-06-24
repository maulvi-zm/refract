from __future__ import annotations

import re

import tree_sitter_python
from tree_sitter import Language, Node

from refract.indexing.parser import node_text
from refract.languages.base import LanguageSpec

_LANGUAGE = Language(tree_sitter_python.language())


def _is_constant_definition(number: Node, source: bytes) -> bool:
    # true if the number is assigned to an ALL_CAPS name
    current = number.parent

    while current is not None:
        if current.type == "assignment":
            left = current.child_by_field_name("left")
            if left is not None and left.type == "identifier":
                name = node_text(left, source)
                return name.isupper() and len(name) > 1
            return False
        current = current.parent

    return False


PYTHON = LanguageSpec(
    name="python",
    extensions=(".py",),
    language=_LANGUAGE,
    fence="python",
    function_query="""
        (function_definition
            name: (identifier) @function.name
            body: (block) @function.body) @function.def
    """,
    enclosing_class_types=frozenset({"class_definition"}),
    class_name_field="name",
    parameter_query="""
        (parameters (identifier) @param.name)
        (default_parameter name: (identifier) @param.name)
        (typed_parameter (identifier) @param.name)
        (typed_default_parameter name: (identifier) @param.name)
    """,
    call_query="""
        (call function: (identifier) @call.name)
        (call function: (attribute attribute: (identifier) @call.name))
    """,
    decision_node_types=frozenset(
        {
            "if_statement",
            "elif_clause",
            "for_statement",
            "while_statement",
            "except_clause",
            "conditional_expression",
            "boolean_operator",
            "case_clause",
            "if_clause",  # comprehension filter
        }
    ),
    logical_binary_types=frozenset(),  # Python uses boolean_operator nodes
    logical_operators=frozenset(),
    statement_query="""
        [
            (expression_statement)
            (if_statement)
            (for_statement)
            (while_statement)
            (with_statement)
            (try_statement)
            (return_statement)
            (raise_statement)
            (assert_statement)
            (delete_statement)
            (global_statement)
            (nonlocal_statement)
            (import_statement)
            (import_from_statement)
            (pass_statement)
            (break_statement)
            (continue_statement)
            (match_statement)
        ] @statement
    """,
    long_method_threshold=20,
    declaration_query="""
        (assignment left: (identifier) @decl.name)
        (parameters (identifier) @decl.name)
        (default_parameter name: (identifier) @decl.name)
        (typed_parameter (identifier) @decl.name)
        (typed_default_parameter name: (identifier) @decl.name)
    """,
    long_identifier_threshold=20,
    number_query="[(integer) (float)] @number",
    ignored_numbers=frozenset({"0", "1", "2"}),
    is_constant_definition=_is_constant_definition,
    constant_pattern=re.compile(r"^[A-Z_][A-Z0-9_]*\s*(?::[^=]+)?=\s*.+$", re.MULTILINE),
)
