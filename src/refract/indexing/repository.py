from __future__ import annotations

import re
from pathlib import Path

from refract.core.models import RepositoryIndex
from refract.indexing.detectors import detect_smells
from refract.indexing.methods import extract_methods
from refract.languages.registry import spec_for_path

# test/build/tooling trees, matched against the repo-RELATIVE posix path
# (with a leading "/" prepended by in_scope) so an ancestor directory that
# happens to contain e.g. "test" (a home dir like /Users/tester/...) can't
# silently exclude the entire repo.
_EXCLUDE = re.compile(
    "|".join(
        (
            # Any path component starting with "test" -- mirrors the dataset's
            # ground-truth rule (tools/scripts/gen_markdown.py: Java "*/test*",
            # Python "*test*") so refract's refactor targeting matches the
            # "application code only" counts and never edits a test file. Covers
            # Java src/test, Python tests/ & testsuite/, pytest test_*.py /
            # conftest.py, and the public testing.py helpers (click.testing,
            # pint.testing) that the dataset also treats as test code. Renaming
            # inside test files is pure risk with no study value -- a pytest
            # `@parametrize("name", ...)` binds by string literal, which an
            # AST-level rename can't see, so it breaks collection (see the
            # click long_identifier regression, refract-run3-behavior-failures).
            r"/test",
            r"/conftest\.py$",  # pytest config file -- doesn't start with "test"
            r"/src/it/",
            r"/xdocs-examples/",
            r"/resources/",
            r"noncompilable",
            r"/module-info\.java$",
            r"/(?:\.venv|venv|site-packages|node_modules|__pycache__)/",
            r"/\.[^/]+/",  # hidden dirs
        )
    )
)


def in_scope(rel_path: Path) -> bool:
    # rel_path is relative to the repo root; the leading "/" anchors patterns
    # like /src/test/ and keeps ancestor directories out of the match.
    return _EXCLUDE.search("/" + rel_path.as_posix()) is None


def source_files(root: Path) -> list[Path]:
    # absolute + sorted: stable order, and no path guessing downstream
    root = root.resolve()
    return [
        path
        for path in sorted(root.rglob("*"))
        if path.is_file() and spec_for_path(path) is not None and in_scope(path.relative_to(root))
    ]


def index_repository(root: Path) -> RepositoryIndex:
    index = RepositoryIndex()

    for path in source_files(root):
        spec = spec_for_path(path)
        assert spec is not None

        index.methods.extend(extract_methods(path, spec))
        index.smells.extend(detect_smells(path, spec))

    return index
