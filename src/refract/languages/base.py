from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from tree_sitter import Language, Node


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

    # --- planning context ---
    # lines matching this are shown to the LLM as relevant constants
    constant_pattern: re.Pattern[str] | None = None
