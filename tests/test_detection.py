from pathlib import Path

from refract.core.models import SmellType
from refract.indexing.detectors import detect_smells
from refract.indexing.methods import extract_methods
from refract.indexing.repository import index_repository
from refract.languages.registry import spec_for_path

JAVA_SOURCE = """\
public class Example {
    private static final int MAX_RETRIES = 3;

    int compute(int quantity) {
        int total = 0;
        for (int i = 0; i < quantity; i++) {
            total += i * 7;
        }
        return total + 42;
    }
}
"""

PYTHON_SOURCE = """\
DEFAULT_TAX_RATE = 7


def compute(quantity):
    running_accumulated_order_total = 0
    for index in range(quantity):
        if index and index % 2 == 0:
            running_accumulated_order_total += 42
    return running_accumulated_order_total
"""


def _write(tmp_path: Path, name: str, source: str) -> Path:
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")
    return path


def _spec(path: Path):
    spec = spec_for_path(path)
    assert spec is not None
    return spec


def _magic_values(smells) -> set[str]:
    return {s.identifier for s in smells if s.smell == SmellType.MAGIC_NUMBER}


def test_java_magic_numbers_with_constant_waiver(tmp_path: Path) -> None:
    path = _write(tmp_path, "Example.java", JAVA_SOURCE)

    values = _magic_values(detect_smells(path, _spec(path)))

    assert "7" in values and "42" in values
    # the static final 3 is waived, 0 and 1 are ignored
    assert "3" not in values and "0" not in values and "1" not in values


def test_python_detects_all_three_smells(tmp_path: Path) -> None:
    path = _write(tmp_path, "example.py", PYTHON_SOURCE)

    smells = detect_smells(path, _spec(path))

    by_type = {s.smell for s in smells}
    assert SmellType.MAGIC_NUMBER in by_type
    assert SmellType.LONG_IDENTIFIER in by_type

    assert _magic_values(smells) == {"42"}  # 7 belongs to an ALL_CAPS constant

    long_ids = {s.identifier for s in smells if s.smell == SmellType.LONG_IDENTIFIER}
    assert "running_accumulated_order_total" in long_ids


def test_method_extraction_populates_parameters_and_calls(tmp_path: Path) -> None:
    path = _write(tmp_path, "example.py", PYTHON_SOURCE)

    methods = extract_methods(path, _spec(path))

    compute = next(m for m in methods if m.name == "compute")
    assert compute.parameters == ["quantity"]
    assert "range" in compute.calls


def test_unsupported_files_are_skipped(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("not source", encoding="utf-8")
    _write(tmp_path, "example.py", PYTHON_SOURCE)

    index = index_repository(tmp_path)

    assert all(m.file.suffix == ".py" for m in index.methods)


def _write_nested(tmp_path: Path, rel: str, source: str) -> Path:
    path = tmp_path / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return path


def test_test_files_are_excluded_from_indexing(tmp_path: Path) -> None:
    # Application code is indexed; test code is not -- refract must never edit a
    # test file (renaming a pytest parametrize/fixture binds by string literal
    # and breaks collection). Mirrors the dataset's application-code-only scope.
    _write_nested(tmp_path, "src/pkg/app.py", PYTHON_SOURCE)
    _write_nested(tmp_path, "tests/test_app.py", PYTHON_SOURCE)  # tests/ dir
    _write_nested(tmp_path, "pkg/testsuite/test_x.py", PYTHON_SOURCE)  # testsuite/ (pint)
    _write_nested(tmp_path, "pkg/conftest.py", PYTHON_SOURCE)  # pytest conftest
    _write_nested(tmp_path, "pkg/testing.py", PYTHON_SOURCE)  # public testing helper
    _write_nested(tmp_path, "src/main/java/App.java", JAVA_SOURCE)
    _write_nested(tmp_path, "src/test/java/AppTest.java", JAVA_SOURCE)  # Java test tree

    indexed = {m.file for m in index_repository(tmp_path).methods}

    assert (tmp_path / "src/pkg/app.py").resolve() in indexed
    assert (tmp_path / "src/main/java/App.java").resolve() in indexed
    for excluded in (
        "tests/test_app.py",
        "pkg/testsuite/test_x.py",
        "pkg/conftest.py",
        "pkg/testing.py",
        "src/test/java/AppTest.java",
    ):
        assert (tmp_path / excluded).resolve() not in indexed, excluded


def test_ancestor_dir_named_test_does_not_exclude_repo(tmp_path: Path) -> None:
    # The match is anchored to the repo-relative path, so a repo living under an
    # ancestor like ".../tester/" must still index its application code.
    repo = tmp_path / "tester" / "myrepo"
    _write_nested(repo, "src/app.py", PYTHON_SOURCE)

    indexed = {m.file for m in index_repository(repo).methods}

    assert (repo / "src/app.py").resolve() in indexed


TYPESCRIPT_SOURCE = """\
import { Injectable } from '@nestjs/common';

const MAX_RETRIES = 3;

@Injectable()
export class UsersService {
  private readonly DEFAULT_TIMEOUT_MS = 5000;

  constructor(private readonly repo: Repo) {}

  async findOne(id: number, opts?: FindOptions): Promise<User | null> {
    const runningAccumulatedOrderTotal = 42;
    if (id > 99) {
      return null;
    }
    for (const key of Object.keys(opts ?? {})) {
      this.repo.evict(key);
    }
    return this.repo.find(id, runningAccumulatedOrderTotal);
  }
}

export const buildHandler = (retries: number) => {
  return retries * 750;
};
"""

JAVASCRIPT_SOURCE = """\
'use strict';
const RETRY_LIMIT = 5;

class Widget {
  render(depth = 3, ...rest) {
    if (depth > 8 && rest.length) {
      return null;
    }
    return rest.map((r) => r * 7);
  }
}
"""


def test_typescript_detects_all_three_smell_inputs(tmp_path: Path) -> None:
    path = _write(tmp_path, "users.service.ts", TYPESCRIPT_SOURCE)
    assert _spec(path).name == "typescript"

    smells = detect_smells(path, _spec(path))

    # MAX_RETRIES (const) and DEFAULT_TIMEOUT_MS (readonly field) are already
    # named constants, so their literals are waived; 99/42/750 are not.
    assert _magic_values(smells) == {"42", "99", "750"}

    long_ids = {s.identifier for s in smells if s.smell == SmellType.LONG_IDENTIFIER}
    assert "runningAccumulatedOrderTotal" in long_ids
    assert "DEFAULT_TIMEOUT_MS" in long_ids  # 18 chars, over the 17 threshold


def test_typescript_method_extraction_covers_classes_and_arrows(tmp_path: Path) -> None:
    path = _write(tmp_path, "users.service.ts", TYPESCRIPT_SOURCE)

    methods = {m.name: m for m in extract_methods(path, _spec(path))}

    find_one = methods["findOne"]
    assert find_one.class_name == "UsersService"
    assert find_one.parameters == ["id", "opts"]
    assert "find" in find_one.calls
    # 1 + if + for-of + ?? -- the nullish coalescing branches like && / ||
    assert find_one.cyclomatic_complexity == 4
    # a `const x = () => {}` is a function even though the name is on the binding
    assert "buildHandler" in methods


def test_javascript_and_tsx_use_their_own_grammars(tmp_path: Path) -> None:
    js = _write(tmp_path, "widget.js", JAVASCRIPT_SOURCE)
    assert _spec(js).name == "javascript"

    render = next(m for m in extract_methods(js, _spec(js)) if m.name == "render")
    assert render.class_name == "Widget"
    assert _magic_values(detect_smells(js, _spec(js))) == {"3", "8", "7"}

    # JSX would be a parse error under the plain TypeScript grammar
    tsx = _write(tmp_path, "App.tsx", "export const App = () => {\n  return <div>{9}</div>;\n};\n")
    assert _spec(tsx).name == "tsx"
    assert _magic_values(detect_smells(tsx, _spec(tsx))) == {"9"}


def test_colocated_js_test_files_are_excluded(tmp_path: Path) -> None:
    # JS/TS put tests beside the source, so no path component starts with "test"
    # and the directory-based rule alone would happily refactor them.
    _write_nested(tmp_path, "src/users/users.service.ts", TYPESCRIPT_SOURCE)
    _write_nested(tmp_path, "src/users/users.service.spec.ts", TYPESCRIPT_SOURCE)
    _write_nested(tmp_path, "src/users/__tests__/users.ts", TYPESCRIPT_SOURCE)
    _write_nested(tmp_path, "dist/users.service.js", JAVASCRIPT_SOURCE)
    _write_nested(tmp_path, "src/types.d.ts", "export declare const x: number;\n")

    indexed = {m.file for m in index_repository(tmp_path).methods}

    assert (tmp_path / "src/users/users.service.ts").resolve() in indexed
    for excluded in (
        "src/users/users.service.spec.ts",
        "src/users/__tests__/users.ts",
        "dist/users.service.js",
        "src/types.d.ts",
    ):
        assert (tmp_path / excluded).resolve() not in indexed, excluded
