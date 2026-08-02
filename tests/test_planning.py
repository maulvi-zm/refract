from pathlib import Path

from refract.core.models import MethodInfo, RepositoryIndex, SmellLocation, SmellType
from refract.planning.context import build_context

SOURCE = """\
public class Example {
  private static final int LIMIT = 10;
  void helper() {}
  void target() {
    int value = 42;
    helper();
  }
  void caller() { target(); }
}
"""


def test_build_context_finds_target_callers_callees_constants(tmp_path: Path) -> None:
    source = tmp_path / "Example.java"
    source.write_text(SOURCE, encoding="utf-8")

    index = RepositoryIndex(
        methods=[
            MethodInfo("helper", "Example", source, 3, 3, 1),
            MethodInfo("target", "Example", source, 4, 7, 2, calls=["helper"]),
            MethodInfo("caller", "Example", source, 8, 8, 1, calls=["target"]),
        ],
        smells=[SmellLocation(SmellType.MAGIC_NUMBER, source, 5, "42", "magic")],
    )

    context = build_context(index, index.smells[0])

    assert context.target_method is not None
    assert context.target_method.name == "target"
    assert [m.name for m in context.callers] == ["caller"]
    assert [m.name for m in context.callees] == ["helper"]
    assert context.constants == ["private static final int LIMIT = 10;"]
    assert context.fence == "java"
