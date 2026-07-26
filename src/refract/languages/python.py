from __future__ import annotations

import re

import tree_sitter_python
from tree_sitter import Language, Node

from refract.indexing.parser import node_text
from refract.languages.base import ConstantPlan, LanguageSpec

_LANGUAGE = Language(tree_sitter_python.language())

# Top-level nodes a new constant may safely be placed after: the module
# docstring, comments, and imports. We anchor on the last of these and stop at
# the first real statement, so the definition always lands at module scope.
_IMPORT_TYPES = frozenset({"import_statement", "import_from_statement", "future_import_statement"})


def _plan_constant(root: Node, literal: Node, data: bytes, name: str) -> ConstantPlan | None:
    """Define ``name`` at module scope, right after the header (docstring /
    comments / imports). None when the file opens with real code or the literal
    itself already lives in the header."""
    row = _header_anchor_row(root)
    if row is None or literal.start_point[0] <= row:
        return None
    value = node_text(literal, data)
    return ConstantPlan(anchor_row=row, indent="", text=f"{name} = {value}")


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
    # Thresholds anchored to the Designite defaults (the oracle): longMethod=20,
    # longIdentifier=17. We count statements (NCSS) where Designite counts LOC --
    # same threshold, different unit; a documented detector/oracle divergence.
    long_method_threshold=20,
    declaration_query="""
        (assignment left: (identifier) @decl.name)
        (parameters (identifier) @decl.name)
        (default_parameter name: (identifier) @decl.name)
        (typed_parameter (identifier) @decl.name)
        (typed_default_parameter name: (identifier) @decl.name)
    """,
    long_identifier_threshold=17,  # Designite default (was 20); matches Java + DPy
    number_query="[(integer) (float)] @number",
    ignored_numbers=frozenset({"0", "1", "2"}),
    is_constant_definition=_is_constant_definition,
    plan_constant=_plan_constant,
    constant_pattern=re.compile(r"^[A-Z_][A-Z0-9_]*\s*(?::[^=]+)?=\s*.+$", re.MULTILINE),
    dead_code_terminators=frozenset({"return_statement", "raise_statement"}),
)
