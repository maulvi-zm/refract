from pathlib import Path

from refract.core.models import MethodInfo, RepositoryIndex, SmellLocation, SmellType
from refract.indexing.database import load, save


def _index(name: str, source: Path) -> RepositoryIndex:
    return RepositoryIndex(
        methods=[
            MethodInfo(
                name=name,
                class_name="Example",
                file=source,
                start_line=1,
                end_line=5,
                cyclomatic_complexity=2,
                parameters=["a", "b"],
                calls=["helper"],
            )
        ],
        smells=[SmellLocation(SmellType.MAGIC_NUMBER, source, 3, "42", "magic")],
    )


def test_save_load_roundtrip(tmp_path: Path) -> None:
    db = tmp_path / "out.db"
    source = tmp_path / "Example.java"
    save(_index("compute", source), db)

    loaded = load(db)
    assert [m.name for m in loaded.methods] == ["compute"]
    assert loaded.methods[0].parameters == ["a", "b"]
    assert loaded.methods[0].calls == ["helper"]
    assert loaded.smells[0].identifier == "42"


def test_save_replaces_previous_snapshot(tmp_path: Path) -> None:
    db = tmp_path / "out.db"
    source = tmp_path / "Example.java"
    save(_index("first", source), db)
    save(_index("second", source), db)

    assert [m.name for m in load(db).methods] == ["second"]
