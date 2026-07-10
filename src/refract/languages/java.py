from __future__ import annotations

import re

import tree_sitter_java
from tree_sitter import Language, Node

from refract.indexing.parser import node_text
from refract.languages.base import ConstantPlan, LanguageSpec

_LANGUAGE = Language(tree_sitter_java.language())

_INT_MAX = 2**31 - 1  # largest signed 32-bit int; a decimal literal above needs `long`
_UINT_MAX = 2**32 - 1  # hex/binary/octal fit int as a 32-bit pattern up to here


def _plan_constant(root: Node, literal: Node, data: bytes, name: str) -> ConstantPlan | None:
    """Define ``name`` as a ``private static final`` field inside the class that
    encloses ``literal``, placed as the first member (right after the opening
    ``{``). None when there's no plain enclosing class -- a literal in an
    annotation, or inside an interface / enum / record body (different field
    semantics) -- so the caller falls back to the model's own edits.
    """
    node = literal.parent
    cls: Node | None = None
    while node is not None:
        if node.type == "annotation":
            return None  # annotation arguments: leave to the model
        if node.type == "class_declaration":
            cls = node
            break
        if node.type in ("interface_declaration", "enum_declaration", "record_declaration"):
            return None
        node = node.parent
    if cls is None:
        return None

    body = cls.child_by_field_name("body")
    if body is None:
        return None

    type_ = _infer_type(literal, data)
    value = node_text(literal, data)
    return ConstantPlan(
        anchor_row=body.start_point[0],  # the line carrying the opening `{`
        indent=_member_indent(cls, body, data),
        text=f"private static final {type_} {name} = {value};",
    )


def _member_indent(cls: Node, body: Node, data: bytes) -> str:
    """Indentation for a new class member: copy the first existing member's
    leading whitespace, or fall back to the class's own indent plus one level."""
    for child in body.children:
        if child.type not in ("{", "}", "line_comment", "block_comment", "comment"):
            return _line_indent(data, child.start_byte)
    return _line_indent(data, cls.start_byte) + "    "


def _line_indent(data: bytes, offset: int) -> str:
    line_start = data.rfind(b"\n", 0, offset) + 1
    i = line_start
    while i < len(data) and data[i : i + 1] in (b" ", b"\t"):
        i += 1
    return data[line_start:i].decode("utf-8")


def _infer_type(literal: Node, data: bytes) -> str:
    """The declared type for a field initialised by ``literal`` (its text is kept
    verbatim as the initialiser; only the type is inferred)."""
    text = node_text(literal, data)
    if literal.type in ("decimal_floating_point_literal", "hex_floating_point_literal"):
        return "float" if text.endswith(("f", "F")) else "double"
    if text.endswith(("l", "L")):
        return "long"
    return "long" if _int_overflows(literal, data) else "int"


def _int_overflows(literal: Node, data: bytes) -> bool:
    text = node_text(literal, data).replace("_", "").rstrip("lL")
    try:
        if literal.type == "hex_integer_literal":
            value, ceiling = int(text, 16), _UINT_MAX
        elif literal.type == "binary_integer_literal":
            value, ceiling = int(text, 2), _UINT_MAX
        elif literal.type == "octal_integer_literal":
            value, ceiling = int(text.lstrip("0") or "0", 8), _UINT_MAX
        else:  # decimal_integer_literal
            value, ceiling = int(text, 10), _INT_MAX
    except ValueError:
        return False
    return value > ceiling


def _is_constant_definition(number: Node, source: bytes) -> bool:
    # true if the number sits in a `static final` field
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
    plan_constant=_plan_constant,
    constant_pattern=re.compile(
        r"\b(?:public|private|protected)?\s*static\s+final\s+[^=;]+=[^;]+;"
    ),
    # A field / variable declaration directly under `program` is not legal Java
    # (fields live in a type body), but tree-sitter parses it without an ERROR --
    # so the syntax guard misses a constant the model hoisted to file scope
    # instead of into the class. Reject it structurally.
    invalid_root_child_types=frozenset({"local_variable_declaration", "field_declaration"}),
)
