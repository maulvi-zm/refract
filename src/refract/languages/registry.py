from __future__ import annotations

from pathlib import Path

from refract.languages.base import LanguageSpec
from refract.languages.ecmascript import JAVASCRIPT, TSX, TYPESCRIPT
from refract.languages.java import JAVA
from refract.languages.python import PYTHON

_SPECS: tuple[LanguageSpec, ...] = (JAVA, PYTHON, TYPESCRIPT, TSX, JAVASCRIPT)

_BY_EXTENSION: dict[str, LanguageSpec] = {ext: spec for spec in _SPECS for ext in spec.extensions}


def spec_for_path(path: Path) -> LanguageSpec | None:
    return _BY_EXTENSION.get(path.suffix)


def all_specs() -> tuple[LanguageSpec, ...]:
    return _SPECS
