from __future__ import annotations

import re
from pathlib import Path

from refract.core.models import RepositoryIndex
from refract.indexing.detectors import detect_smells
from refract.indexing.methods import extract_methods
from refract.languages.registry import spec_for_path

# test/build/tooling trees, matched against posix paths
_EXCLUDE = re.compile(
    "|".join(
        (
            r"/xdocs-examples/",
            r"/resources/",
            r"noncompilable",
            r"/module-info\.java$",
            r"/(?:\.venv|venv|site-packages|node_modules|__pycache__)/",
            r"/\.[^/]+/",  # hidden dirs
        )
    )
)


def in_scope(path: Path) -> bool:
    return _EXCLUDE.search(path.as_posix()) is None


def source_files(root: Path) -> list[Path]:
    # absolute + sorted: stable order, and no path guessing downstream
    return [
        path
        for path in sorted(root.resolve().rglob("*"))
        if path.is_file() and spec_for_path(path) is not None and in_scope(path)
    ]


def index_repository(root: Path) -> RepositoryIndex:
    index = RepositoryIndex()

    for path in source_files(root):
        spec = spec_for_path(path)
        assert spec is not None

        index.methods.extend(extract_methods(path, spec))
        index.smells.extend(detect_smells(path, spec))

    return index
