"""LanguageSpecs for the ECMAScript family: JavaScript, TypeScript, TSX.

The three grammars share node-type names almost everywhere (tree-sitter's
TypeScript grammar is the JavaScript grammar plus type syntax), so the specs are
built by one factory and differ only where the parse trees genuinely diverge:

  * parameters -- JavaScript puts bare patterns inside ``formal_parameters``,
    TypeScript wraps each one in ``required_parameter`` / ``optional_parameter``
    to carry the type annotation.
  * declarations -- TypeScript adds class fields (``public_field_definition``),
    which is where a NestJS provider keeps most of its state.

Everything else (statements, decision points, literals, constant placement) is
shared. TSX reuses the TypeScript node types wholesale; only the grammar object
and the extensions change.
"""

from __future__ import annotations

import re

import tree_sitter_javascript
import tree_sitter_typescript
from tree_sitter import Language, Node

from refract.indexing.parser import node_text
from refract.languages.base import ConstantPlan, LanguageSpec

# Top-level nodes a new constant may safely be placed after: leading comments,
# imports, and a "use strict" prologue. We anchor on the last of these and stop
# at the first real statement, so the definition always lands at module scope.
_IMPORT_TYPES = frozenset({"import_statement"})


def _plan_constant(root: Node, literal: Node, data: bytes, name: str) -> ConstantPlan | None:
    """Define ``name`` as a module-scope ``const``, right after the header
    (comments / imports / prologue).

    Module scope rather than a class field: a hoisted ``const`` is visible to
    every method in the file and is legal at the top level of both scripts and
    ES modules, so it needs none of the enclosing-class analysis Java requires.
    None when the file opens with real code (no safe anchor -- inserting after
    line 0 could land inside a class body) or the literal itself already lives
    in the header, so the caller falls back to the model's own edits.
    """
    row = _header_anchor_row(root)
    if row is None or literal.start_point[0] <= row:
        return None
    value = node_text(literal, data)
    return ConstantPlan(anchor_row=row, indent="", text=f"const {name} = {value};")


def _header_anchor_row(root: Node) -> int | None:
    """0-based row to insert the constant after: the last leading comment,
    import, or "use strict" prologue. None when the file opens with real code."""
    anchor: int | None = None
    for index, child in enumerate(root.children):
        if child.type in _IMPORT_TYPES or child.type == "comment":
            anchor = child.end_point[0]
        elif index == 0 and _is_prologue(child):
            anchor = child.end_point[0]
        else:
            break
    return anchor


def _is_prologue(node: Node) -> bool:
    """A bare string expression statement opening the file -- "use strict"."""
    return (
        node.type == "expression_statement"
        and node.child_count > 0
        and node.children[0].type == "string"
    )


def _is_constant_definition(number: Node, source: bytes) -> bool:
    """True when the number initialises an ALL_CAPS ``const`` or ``readonly``
    class field -- i.e. it is already a named constant, not a magic number."""
    current = number.parent

    while current is not None:
        if current.type == "variable_declarator":
            # only `const` counts; a reassignable `let`/`var` is not a constant
            declaration = current.parent
            if declaration is None or not _is_const_declaration(declaration, source):
                return False
            name = current.child_by_field_name("name")
            return name is not None and _is_screaming_case(name, source)
        if current.type == "public_field_definition":
            if "readonly" not in node_text(current, source).split("=")[0]:
                return False
            name = current.child_by_field_name("name")
            return name is not None and _is_screaming_case(name, source)
        current = current.parent

    return False


def _is_const_declaration(declaration: Node, source: bytes) -> bool:
    kind = declaration.child_by_field_name("kind")
    if kind is not None:  # newer grammars expose the keyword as a field
        return node_text(kind, source) == "const"
    # older ones leave it as the first anonymous child of the lexical_declaration
    return (
        declaration.type == "lexical_declaration"
        and declaration.child_count > 0
        and node_text(declaration.children[0], source) == "const"
    )


def _is_screaming_case(name: Node, source: bytes) -> bool:
    text = node_text(name, source)
    return text.isupper() and len(text) > 1


# --- shared query / node-type vocabulary --------------------------------------

# Four ways to spell a function. The first three are the classic declarations;
# the fourth catches `const handler = (req) => {...}`, which is how a large share
# of modern JS/TS code declares functions at all -- the name lives on the
# variable, not the function, so the declarator is the @function.def node.
_FUNCTION_QUERY = """
    (function_declaration
        name: (identifier) @function.name
        body: (statement_block) @function.body) @function.def
    (generator_function_declaration
        name: (identifier) @function.name
        body: (statement_block) @function.body) @function.def
    (method_definition
        name: (property_identifier) @function.name
        body: (statement_block) @function.body) @function.def
    (variable_declarator
        name: (identifier) @function.name
        value: [
            (arrow_function body: (statement_block) @function.body)
            (function_expression body: (statement_block) @function.body)
        ]) @function.def
"""

# Expression-bodied arrows (`const f = (n) => n * 2`) are deliberately absent:
# they have no statement_block, so they cannot be a long method and have no body
# to scan for calls.

_CALL_QUERY = """
    (call_expression function: (identifier) @call.name)
    (call_expression function: (member_expression property: (property_identifier) @call.name))
"""

_DECISION_NODE_TYPES = frozenset(
    {
        "if_statement",
        "for_statement",
        "for_in_statement",  # covers both `for...in` and `for...of`
        "while_statement",
        "do_statement",
        "catch_clause",
        "switch_case",  # `default` is not a decision point, so switch_default is out
        "ternary_expression",
    }
)

# `??` branches on nullishness exactly as `&&` / `||` branch on truthiness, so it
# counts toward McCabe alongside them.
_LOGICAL_OPERATORS = frozenset({"&&", "||", "??"})

_STATEMENT_QUERY = """
    [
        (expression_statement)
        (lexical_declaration)
        (variable_declaration)
        (if_statement)
        (for_statement)
        (for_in_statement)
        (while_statement)
        (do_statement)
        (try_statement)
        (return_statement)
        (throw_statement)
        (switch_statement)
        (break_statement)
        (continue_statement)
        (labeled_statement)
        (with_statement)
        (import_statement)
    ] @statement
"""

# Every numeric literal is a single `number` node in both grammars (hex, binary,
# octal, float and exponent forms all included).
_NUMBER_QUERY = "(number) @number"

# `export const MAX = 5`, `private static readonly TIMEOUT_MS = 5000`
_CONSTANT_PATTERN = re.compile(
    r"^\s*(?:export\s+)?"
    r"(?:const|(?:(?:public|private|protected)\s+)?(?:static\s+)?readonly)\s+"
    r"[A-Z_][A-Z0-9_]*\b.*$",
    re.MULTILINE,
)


def _build(
    *,
    name: str,
    extensions: tuple[str, ...],
    language: Language,
    fence: str,
    parameter_query: str,
    declaration_query: str,
    enclosing_class_types: frozenset[str],
) -> LanguageSpec:
    return LanguageSpec(
        name=name,
        extensions=extensions,
        language=language,
        fence=fence,
        function_query=_FUNCTION_QUERY,
        enclosing_class_types=enclosing_class_types,
        class_name_field="name",
        parameter_query=parameter_query,
        call_query=_CALL_QUERY,
        decision_node_types=_DECISION_NODE_TYPES,
        logical_binary_types=frozenset({"binary_expression"}),
        logical_operators=_LOGICAL_OPERATORS,
        statement_query=_STATEMENT_QUERY,
        # Same Designite-derived thresholds the Java and Python specs use
        # (longMethod=20, longIdentifier=17), kept identical so a cross-language
        # comparison measures the language, not a retuned detector.
        long_method_threshold=20,
        declaration_query=declaration_query,
        long_identifier_threshold=17,
        number_query=_NUMBER_QUERY,
        ignored_numbers=frozenset({"0", "1", "2"}),
        is_constant_definition=_is_constant_definition,
        plan_constant=_plan_constant,
        constant_pattern=_CONSTANT_PATTERN,
        # Unlike Java, a declaration at the root of a module is perfectly legal
        # here (that is where module-scope consts live), so there is nothing for
        # the misplaced-member guard to reject.
        invalid_root_child_types=frozenset(),
        dead_code_terminators=frozenset({"return_statement", "throw_statement"}),
    )


# --- JavaScript ---------------------------------------------------------------

_JS_PARAMETER_QUERY = """
    (formal_parameters (identifier) @param.name)
    (formal_parameters (assignment_pattern left: (identifier) @param.name))
    (formal_parameters (rest_pattern (identifier) @param.name))
    (formal_parameters (object_pattern (shorthand_property_identifier_pattern) @param.name))
    (arrow_function parameter: (identifier) @param.name)
"""

_JS_DECLARATION_QUERY = """
    (variable_declarator name: (identifier) @decl.name)
    (formal_parameters (identifier) @decl.name)
    (formal_parameters (assignment_pattern left: (identifier) @decl.name))
    (field_definition property: (property_identifier) @decl.name)
"""

JAVASCRIPT = _build(
    name="javascript",
    # .jsx is handled by this grammar too -- tree-sitter-javascript parses JSX.
    extensions=(".js", ".jsx", ".mjs", ".cjs"),
    language=Language(tree_sitter_javascript.language()),
    fence="javascript",
    parameter_query=_JS_PARAMETER_QUERY,
    declaration_query=_JS_DECLARATION_QUERY,
    enclosing_class_types=frozenset({"class_declaration", "class"}),
)


# --- TypeScript ---------------------------------------------------------------

# TypeScript wraps each parameter to carry its type annotation, so the bare
# `(formal_parameters (identifier))` pattern that works for JavaScript matches
# nothing here.
_TS_PARAMETER_QUERY = """
    (required_parameter pattern: (identifier) @param.name)
    (optional_parameter pattern: (identifier) @param.name)
    (required_parameter pattern: (rest_pattern (identifier) @param.name))
    (required_parameter pattern:
        (object_pattern (shorthand_property_identifier_pattern) @param.name))
    (arrow_function parameter: (identifier) @param.name)
"""

_TS_DECLARATION_QUERY = """
    (variable_declarator name: (identifier) @decl.name)
    (required_parameter pattern: (identifier) @decl.name)
    (optional_parameter pattern: (identifier) @decl.name)
    (public_field_definition name: (property_identifier) @decl.name)
"""

# An interface / type alias body declares names too, but they are types rather
# than variables; the Java spec likewise skips type members, so they stay out.
_TS_CLASS_TYPES = frozenset({"class_declaration", "class", "interface_declaration"})

TYPESCRIPT = _build(
    name="typescript",
    extensions=(".ts", ".mts", ".cts"),
    language=Language(tree_sitter_typescript.language_typescript()),
    fence="typescript",
    parameter_query=_TS_PARAMETER_QUERY,
    declaration_query=_TS_DECLARATION_QUERY,
    enclosing_class_types=_TS_CLASS_TYPES,
)

# A separate grammar, not just a separate extension: the `<T>` of a type
# assertion and the `<T>` of a JSX tag are genuinely ambiguous, and the two
# grammars resolve it in opposite directions. Node types are otherwise identical,
# so the whole vocabulary above is reused unchanged.
TSX = _build(
    name="tsx",
    extensions=(".tsx",),
    language=Language(tree_sitter_typescript.language_tsx()),
    fence="tsx",
    parameter_query=_TS_PARAMETER_QUERY,
    declaration_query=_TS_DECLARATION_QUERY,
    enclosing_class_types=_TS_CLASS_TYPES,
)
