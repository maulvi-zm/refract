from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class SmellType(str, Enum):
    LONG_METHOD = "long_method"
    LONG_IDENTIFIER = "long_identifier"
    MAGIC_NUMBER = "magic_number"


@dataclass
class MethodInfo:
    """A parsed method or function. Lines are 1-based and inclusive."""

    name: str
    class_name: str
    file: Path
    start_line: int
    end_line: int

    cyclomatic_complexity: int = 1

    parameters: list[str] = field(default_factory=list)
    calls: list[str] = field(default_factory=list)  # callee names, unresolved

    @property
    def line_count(self) -> int:
        return self.end_line - self.start_line + 1


@dataclass
class SmellLocation:
    """A detected smell.

    `identifier` is the method name (long_method), the name (long_identifier),
    or the literal such as "42" (magic_number). `detail` is the LLM message.
    """

    smell: SmellType
    file: Path
    line: int
    identifier: str
    detail: str


@dataclass
class RepositoryIndex:
    methods: list[MethodInfo] = field(default_factory=list)
    smells: list[SmellLocation] = field(default_factory=list)

    def methods_by_file(self, file: Path) -> list[MethodInfo]:
        resolved = file.resolve()
        return [m for m in self.methods if m.file.resolve() == resolved]

    def smells_by_type(self, smell: SmellType) -> list[SmellLocation]:
        return [s for s in self.smells if s.smell == smell]
