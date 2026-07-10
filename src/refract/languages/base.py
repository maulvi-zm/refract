from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from tree_sitter import Language, Node


@dataclass(frozen=True)
class ConstantPlan:
    """Where and how to insert a hoisted magic-number constant.

    Language-agnostic instructions the deterministic extract-constant builder
    turns into a ``SnippetEdit``: append ``indent + text`` on the line after
    ``anchor_row``. Python renders a module-scope ``NAME = value`` after the
    header; Java renders a ``private static final <type> NAME = value;`` field
    just inside the enclosing class. See languages' ``plan_constant``.
    """

    anchor_row: int  # 0-based source row to insert the definition *after*
    indent: str  # leading whitespace for the definition line
    text: str  # the rendered definition (no leading indent, no trailing newline)


@dataclass(frozen=True)
class LanguageSpec:
    name: str
    extensions: tuple[str, ...]
    language: Language
    fence: str  # code-fence label for prompts (e.g. "java")

    # --- method / function extraction ---
    # captures @function.name, @function.body and @function.def (the whole node)
    function_query: str
    enclosing_class_types: frozenset[str]
    class_name_field: str
    parameter_query: str  # @param.name, run on a function node
    call_query: str  # @call.name, run on a function body

    # --- cyclomatic complexity ---
    decision_node_types: frozenset[str]
    logical_binary_types: frozenset[str]  # binary nodes that might be && / ||
    logical_operators: frozenset[str]

    # --- long method ---
    statement_query: str  # @statement, run on a body; one capture = one statement
    long_method_threshold: int

    # --- long identifier ---
    declaration_query: str  # @decl.name for every declared name
    long_identifier_threshold: int

    # --- magic number ---
    number_query: str  # @number for numeric literals
    ignored_numbers: frozenset[str]
    is_constant_definition: Callable[[Node, bytes], bool]
    # (root, literal, data, const_name) -> where/how to define the constant, or
    # None to fall back to the model when it can't be done deterministically.
    plan_constant: Callable[[Node, Node, bytes, str], "ConstantPlan | None"]

    # --- planning context ---
    # lines matching this are shown to the LLM as relevant constants
    constant_pattern: re.Pattern[str] | None = None

    # --- structural guard ---
    # node types that are illegal as a *direct child* of the parse root (the
    # compilation unit / module). tree-sitter is error-tolerant and parses some
    # misplaced declarations without an ERROR node -- e.g. a Java
    # `static final` field hoisted to file scope reads as a valid
    # `local_variable_declaration` under `program`, so `has_error` stays False
    # even though javac rejects it. The patcher rejects any edit that introduces
    # one of these at root, catching the do-no-harm hole the syntax check misses.
    # Empty for languages (e.g. Python) where top-level statements are valid.
    invalid_root_child_types: frozenset[str] = frozenset()
