from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from refract.core.models import MethodInfo, RepositoryIndex, SmellLocation, SmellType
from refract.indexing.parser import parse
from refract.languages.registry import spec_for_path


@dataclass(frozen=True)
class RefactorContext:
    smell: SmellLocation
    target_method: MethodInfo | None
    snippet: str
    snippet_start_line: int
    file_header: str = ""
    same_file_methods: list[MethodInfo] = field(default_factory=list)
    callers: list[MethodInfo] = field(default_factory=list)
    callees: list[MethodInfo] = field(default_factory=list)
    constants: list[str] = field(default_factory=list)

    @property
    def fence(self) -> str:
        spec = spec_for_path(self.smell.file)
        return spec.fence if spec else ""

    def relative_file(self, repo_root: Path) -> str:
        try:
            return str(self.smell.file.relative_to(repo_root))
        except ValueError:
            return str(self.smell.file)


def build_contexts(
    index: RepositoryIndex,
    smell_type: SmellType,
    limit: int = 1,
) -> list[RefactorContext]:
    smells = sorted(
        index.smells_by_type(smell_type),
        key=lambda smell: (str(smell.file), smell.line, smell.identifier),
    )
    return [build_context(index, smell) for smell in smells[: max(limit, 0)]]


def build_context(index: RepositoryIndex, smell: SmellLocation) -> RefactorContext:
    methods = sorted(index.methods_by_file(smell.file), key=lambda m: m.start_line)
    target = _containing_method(methods, smell)

    # A rename must cover EVERY occurrence, so the snippet has to show them all --
    # see _snippet_for.
    must_cover = (
        _identifier_lines(smell.file, smell.identifier)
        if smell.smell is SmellType.LONG_IDENTIFIER
        else ()
    )

    snippet, start_line = _snippet_for(smell.file, smell.line, target, must_cover=must_cover)
    header = _file_header(smell.file, start_line)

    return RefactorContext(
        smell=smell,
        target_method=target,
        snippet=snippet,
        snippet_start_line=start_line,
        file_header=header,
        same_file_methods=methods,
        callers=_callers(index, target),
        callees=_callees(index, target),
        constants=_constants(smell.file),
    )


def _containing_method(methods: list[MethodInfo], smell: SmellLocation) -> MethodInfo | None:
    containing = [m for m in methods if m.start_line <= smell.line <= m.end_line]
    if containing:
        # innermost method for nested definitions
        return min(containing, key=lambda m: m.end_line - m.start_line)

    # for long method
    by_identifier = [m for m in methods if m.name == smell.identifier]
    return by_identifier[0] if by_identifier else None


def _snippet_for(
    file_path: Path,
    smell_line: int,
    target: MethodInfo | None,
    radius: int = 20,
    must_cover: tuple[int, ...] = (),
    padding: int = 3,
) -> tuple[str, int]:
    """The source region to show the model, and the file line it starts at.

    Base region: the containing method's body, or ``smell_line +/- radius`` when the
    smell sits outside any method (a class-level field).

    ``must_cover`` widens that region until it spans those lines. A long-identifier
    fix has to rename every occurrence -- and the patcher rejects the patch if one
    survives -- so a use site the model was never shown is an unfixable target, not
    a bad answer. That is exactly what a hoisted constant hits: the definition
    stacks at the top of the class while its uses stay put in a method further
    down, easily more than ``radius`` lines away.
    """
    lines = file_path.read_text(encoding="utf-8").splitlines()

    if target:
        start = max(target.start_line, 1)
        end = min(target.end_line, len(lines))
    else:
        start = max(smell_line - radius, 1)
        end = min(smell_line + radius, len(lines))

    if must_cover:
        start = min(start, min(must_cover) - padding)
        end = max(end, max(must_cover) + padding)

    start = max(start, 1)
    end = min(end, len(lines))

    return "\n".join(lines[start - 1 : end]), start


def _identifier_lines(file_path: Path, identifier: str) -> tuple[int, ...]:
    """1-based lines carrying ``identifier`` as an identifier token.

    Matches on tokens rather than raw text so occurrences inside strings and
    comments are ignored -- the same rule ``_reject_if_identifier_remains`` applies
    when it decides a rename was complete. Empty when the language is unknown, in
    which case the caller keeps its default window.
    """
    spec = spec_for_path(file_path)
    if spec is None:
        return ()

    data = file_path.read_bytes()
    target = identifier.encode("utf-8")

    lines: set[int] = set()
    stack = [parse(data, spec.language)]
    while stack:
        node = stack.pop()
        if node.type == "identifier" and data[node.start_byte : node.end_byte] == target:
            lines.add(node.start_point[0] + 1)
        stack.extend(node.children)

    return tuple(sorted(lines))


def _file_header(file_path: Path, snippet_start_line: int, max_lines: int = 30) -> str:
    """Top-of-file region (imports, module docstring, existing module constants).

    Extract-constant needs to place the new definition at module scope, but the
    per-method snippet hides the top of the file -- so the model would guess the
    placement anchor and land the constant between a decorator and its def (a
    syntax error) or against an import line it misremembers (not found). This
    hands it the real header text to anchor against. Cut at the first top-level
    def/class/decorator (constants belong above it) and capped; skipped when the
    snippet already starts at the top so it isn't shown twice.
    """
    lines = file_path.read_text(encoding="utf-8").splitlines()
    end = 0
    for i, line in enumerate(lines[:max_lines]):
        stripped = line.lstrip()
        if (
            line
            and not line[0].isspace()
            and stripped.startswith(("def ", "class ", "@", "async def "))
        ):
            break
        end = i + 1

    # nothing useful, or the snippet already covers the top of the file
    if end == 0 or snippet_start_line <= end:
        return ""
    return "\n".join(lines[:end])


def _callers(index: RepositoryIndex, target: MethodInfo | None) -> list[MethodInfo]:
    if not target:
        return []

    return sorted(
        (m for m in index.methods if target.name in m.calls and m != target),
        key=lambda m: (str(m.file), m.start_line),
    )


def _callees(index: RepositoryIndex, target: MethodInfo | None) -> list[MethodInfo]:
    if not target:
        return []

    by_name: dict[str, list[MethodInfo]] = {}
    for method in index.methods:
        by_name.setdefault(method.name, []).append(method)

    result: list[MethodInfo] = []
    for name in sorted(set(target.calls)):
        candidates = by_name.get(name, [])
        if not candidates:
            continue
        # disambiguate by class when the name is not unique
        same_class = [m for m in candidates if m.class_name == target.class_name]
        result.append(same_class[0] if same_class else candidates[0])

    return result


def _constants(file_path: Path) -> list[str]:
    spec = spec_for_path(file_path)
    if spec is None or spec.constant_pattern is None:
        return []

    text = file_path.read_text(encoding="utf-8")
    return [match.group(0).strip() for match in spec.constant_pattern.finditer(text)]
