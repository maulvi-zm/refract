from __future__ import annotations

from refract.benchmark.runner import ToolResult

_HEADERS = [
    "Tool",
    "Model",
    "API Calls",
    "Failed",
    "Input Tok",
    "Output Tok",
    "Before",
    "After",
    "Fixed",
    "FileCC Before",
    "FileCC After",
    "Unmatched",
    "Tgt CC med",
    "Tgt CC max",
    "Tgt LOC med",
    "Tgt LOC max",
    "Syntax Broken",
    "Tests",
    "Oracle Before",
    "Oracle After",
]


def _tests_cell(tests_passed: bool | None) -> str:
    if tests_passed is None:
        return "n/a"
    return "PASS" if tests_passed else "FAIL"


def _oracle_cell(count: int | None) -> str:
    return "n/a" if count is None else str(count)


def _num(value: float) -> str:
    """Whole numbers print without a trailing .0; medians keep their half."""
    return str(int(value)) if float(value).is_integer() else f"{value:g}"


def _arrow(before: float, after: float) -> str:
    return f"{_num(before)}->{_num(after)}"


def print_report(results: list[ToolResult]) -> None:
    if not results:
        print("No results to display.")
        return

    rows = [
        [
            r.tool,
            r.model,
            str(r.api_calls),
            str(r.failed_api_calls),
            str(r.input_tokens),
            str(r.output_tokens),
            str(r.smells_before),
            str(r.smells_after),
            str(r.fixed),
            str(r.complexity_before),
            str(r.complexity_after),
            str(r.complexity_unmatched),
            _arrow(r.target_cc_before_median, r.target_cc_after_median),
            _arrow(r.target_cc_before_max, r.target_cc_after_max),
            _arrow(r.target_loc_before_median, r.target_loc_after_median),
            _arrow(r.target_loc_before_max, r.target_loc_after_max),
            str(r.syntax_broken_files),
            _tests_cell(r.tests_passed),
            _oracle_cell(r.oracle_smells_before),
            _oracle_cell(r.oracle_smells_after),
        ]
        for r in results
    ]
    # widen each column to fit its header and its values
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
