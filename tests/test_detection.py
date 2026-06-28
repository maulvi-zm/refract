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
