from __future__ import annotations

from refract.benchmark.runner import ToolResult

_HEADERS = ["Tool", "Model", "API Calls", "Input Tok", "Output Tok", "Before", "After", "Fixed"]


def print_report(results: list[ToolResult]) -> None:
    if not results:
        print("No results to display.")
        return

    rows = [
        [
            r.tool,
            r.model,
            str(r.api_calls),
            str(r.input_tokens),
            str(r.output_tokens),
            str(r.smells_before),
            str(r.smells_after),
            str(r.fixed),
        ]
        for r in results
    ]
    # each column is as wide as its header or its widest value
    widths = [max(len(h), *(len(row[i]) for row in rows)) for i, h in enumerate(_HEADERS)]

    def render(cells: list[str]) -> str:
        return "  ".join(cell.ljust(w) for cell, w in zip(cells, widths))

    print()
    print(render(_HEADERS))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print(render(row))

    for result in results:
        if result.error:
            print(f"\n[{result.tool}] error: {result.error}")
