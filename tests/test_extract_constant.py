from pathlib import Path

from refract.core.models import SmellLocation, SmellType
from refract.indexing.parser import parse
from refract.languages.registry import spec_for_path
from refract.refactoring.extract_constant import build_constant_extraction
from refract.refactoring.patcher import apply_snippet_replacement


def _write(tmp_path: Path, name: str, source: str) -> Path:
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")
    return path


def _smell(path: Path, literal: str, source: str) -> SmellLocation:
    line = next(i for i, text in enumerate(source.splitlines(), 1) if literal in text)
    return SmellLocation(SmellType.MAGIC_NUMBER, path, line, literal, "magic")


def _parses(path: Path, text: str) -> bool:
    spec = spec_for_path(path)
    assert spec is not None
    return not parse(text.encode("utf-8"), spec.language).has_error


def test_extracts_literal_inside_stacked_decorator(tmp_path: Path) -> None:
    # the case single-shot verbatim matching keeps missing: a magic number buried
    # in a multi-line, decorator-dense span above the def.
    source = (
        "import click\n"
        "\n"
        "\n"
        "@click.command()\n"
        "@click.option(\n"
        '    "--timeout",\n'
        "    default=2.5,\n"
        "    help=\"seconds\",\n"
        ")\n"
        "def run(timeout):\n"
        "    return timeout\n"
    )
    path = _write(tmp_path, "cli.py", source)

    proposal = build_constant_extraction(path, _smell(path, "2.5", source), "DEFAULT_TIMEOUT")

    assert proposal is not None
    updated = apply_snippet_replacement(path, proposal, apply=True)
    # constant defined at module scope, right after the import and above the decorator
    assert "DEFAULT_TIMEOUT = 2.5" in updated
    assert updated.index("DEFAULT_TIMEOUT = 2.5") < updated.index("@click.command()")
    # use site rewritten, literal gone
    assert "default=DEFAULT_TIMEOUT," in updated
    assert "default=2.5" not in updated
    assert _parses(path, updated)


def test_extracts_simple_literal_after_imports(tmp_path: Path) -> None:
    source = "import os\nimport sys\n\n\ndef read():\n    return os.read(0, 1024)\n"
    path = _write(tmp_path, "io.py", source)

    proposal = build_constant_extraction(path, _smell(path, "1024", source), "BUFFER_SIZE")

    assert proposal is not None
    updated = apply_snippet_replacement(path, proposal, apply=True)
    assert "BUFFER_SIZE = 1024" in updated
    assert updated.index("BUFFER_SIZE = 1024") < updated.index("def read()")
    assert "os.read(0, BUFFER_SIZE)" in updated
    assert _parses(path, updated)


def test_bails_when_literal_is_ambiguous_on_its_line(tmp_path: Path) -> None:
    # two identical literals on the flagged line: no column info to disambiguate,
    # so the deterministic path must decline and let the model try.
    source = "import os\n\n\ndef f():\n    return 1234 + 1234\n"
    path = _write(tmp_path, "amb.py", source)

    assert build_constant_extraction(path, _smell(path, "1234", source), "VALUE") is None


def test_rejects_invalid_constant_name(tmp_path: Path) -> None:
    source = "import os\n\n\ndef f():\n    return 1024\n"
    path = _write(tmp_path, "bad.py", source)

    assert build_constant_extraction(path, _smell(path, "1024", source), "1024") is None
    assert build_constant_extraction(path, _smell(path, "1024", source), "class") is None


def test_bails_when_literal_is_already_a_named_constant(tmp_path: Path) -> None:
    # a literal already assigned to an ALL_CAPS name isn't flagged by the detector,
    # and re-finding it here filters it out the same way -- so we decline.
    source = "import os\n\nSETTINGS = dict(retries=1500)\n\n\ndef f():\n    return SETTINGS\n"
    path = _write(tmp_path, "hdr.py", source)

    assert build_constant_extraction(path, _smell(path, "1500", source), "MAX_RETRIES") is None
