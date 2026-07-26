from pathlib import Path
from unittest.mock import patch

from refract.benchmark.report import print_report
from refract.benchmark.runner import (
    Baseline,
    TargetMethod,
    ToolResult,
    _agentic_prompt,
    _analyze_after,
    _cap_flagged,
    _complexity_after,
    _clean_subprocess_stderr,
    _enclosing_method,
    _no_calls_error,
    _syntax_error_files,
    _target_methods,
    run_benchmark,
)
from refract.core.models import MethodInfo, RepositoryIndex, SmellLocation, SmellType
from refract.verification.runner import VerificationResult


def _method(
    name: str, class_name: str, start: int, end: int, complexity: int, file: Path
) -> MethodInfo:
    return MethodInfo(
        name=name,
        class_name=class_name,
        file=file,
        start_line=start,
        end_line=end,
        cyclomatic_complexity=complexity,
    )


def _smell(
    file: Path,
    line: int,
    identifier: str = "x",
    smell: SmellType = SmellType.LONG_METHOD,
) -> SmellLocation:
    return SmellLocation(smell=smell, file=file, line=line, identifier=identifier, detail="")


def test_enclosing_method_picks_innermost(tmp_path: Path) -> None:
    f = tmp_path / "mod.py"
    outer = _method("outer", "<unknown>", 1, 20, 5, f)
    inner = _method("inner", "<unknown>", 5, 10, 2, f)
    index = RepositoryIndex(methods=[outer, inner])

    found = _enclosing_method(index, f, 7)

    assert found is inner  # smallest enclosing range wins, not whichever matched first


def test_target_methods_resolves_and_dedupes_same_method(tmp_path: Path) -> None:
    f = tmp_path / "mod.py"
    method = _method("progress", "<unknown>", 10, 50, 9, f)
    index = RepositoryIndex(methods=[method])
    # two smells inside the same method (e.g. it's also flagged for a magic number)
    flagged = [_smell(f, 10, identifier="progress"), _smell(f, 25, identifier="progress")]

    targets = _target_methods(index, flagged, tmp_path)

    assert targets == [
        TargetMethod(
            file=Path("mod.py"),
            class_name="<unknown>",
            name="progress",
            start_line=10,
            complexity_before=9,
            loc_before=41,  # 50 - 10 + 1
        )
    ]


def test_target_methods_skips_smells_outside_any_method(tmp_path: Path) -> None:
    f = tmp_path / "mod.py"
    index = RepositoryIndex(methods=[])
    flagged = [_smell(f, 1, identifier="42", smell=SmellType.MAGIC_NUMBER)]

    assert _target_methods(index, flagged, tmp_path) == []


def _one_target_baseline(target: TargetMethod) -> Baseline:
    return Baseline(
        target_methods=[target],
        file_complexity_before={target.file: target.complexity_before},
        broken_files=frozenset(),
        tests_passed=None,
        test_command="auto",
    )


def test_complexity_after_disambiguates_same_name_different_class(tmp_path: Path) -> None:
    f = tmp_path / "mod.py"
    target = TargetMethod(
        file=Path("mod.py"),
        class_name="A",
        name="__init__",
        start_line=1,
        complexity_before=8,
        loc_before=12,
    )
    after_index = RepositoryIndex(
        methods=[
            _method("__init__", "A", 1, 5, 3, f),  # the one that actually got refactored
            _method("__init__", "B", 40, 42, 1, f),  # unrelated method, same name + file
        ]
    )

    result = _complexity_after(after_index, tmp_path, _one_target_baseline(target))

    # must measure A.__init__ (CC 3), not B.__init__ (CC 1)
    assert result.target_cc_before_max == 8
    assert result.target_cc_after_max == 3
    assert result.unmatched == 0


def test_complexity_after_breaks_ties_by_line_proximity(tmp_path: Path) -> None:
    f = tmp_path / "mod.py"
    # two free functions both literally named "decorator" (e.g. two decorator-factory
    # closures), at very different lines -- class_name alone can't disambiguate these
    target = TargetMethod(
        file=Path("mod.py"),
        class_name="<unknown>",
        name="decorator",
        start_line=100,
        complexity_before=9,
        loc_before=20,
    )
    after_index = RepositoryIndex(
        methods=[
            _method("decorator", "<unknown>", 5, 8, 1, f),  # unrelated closure, far away
            _method(
                "decorator", "<unknown>", 98, 103, 4, f
            ),  # the real target, roughly where it was
        ]
    )

    result = _complexity_after(after_index, tmp_path, _one_target_baseline(target))

    assert result.target_cc_before_max == 9
    assert result.target_cc_after_max == 4  # the closest match, not the far-away closure
    assert result.unmatched == 0


def test_complexity_after_counts_unmatched_when_method_disappears(tmp_path: Path) -> None:
    target = TargetMethod(
        file=Path("mod.py"),
        class_name="<unknown>",
        name="renamed_away",
        start_line=1,
        complexity_before=9,
        loc_before=20,
    )
    after_index = RepositoryIndex(methods=[])

    result = _complexity_after(after_index, tmp_path, _one_target_baseline(target))

    assert result.unmatched == 1
    assert result.target_cc_after_max == 0  # no matched target -> empty after distribution


def test_complexity_after_counts_extracted_helpers_in_file_scope(tmp_path: Path) -> None:
    f = tmp_path / "mod.py"
    # a CC-9 method that got split into a CC-4 remainder plus two CC-3 helpers:
    # the target sum reads -5, but the file scope is unchanged (a pure move).
    target = TargetMethod(
        file=Path("mod.py"),
        class_name="<unknown>",
        name="big",
        start_line=1,
        complexity_before=9,
        loc_before=30,
    )
    baseline = Baseline(
        target_methods=[target],
        file_complexity_before={Path("mod.py"): 9},
        broken_files=frozenset(),
        tests_passed=None,
        test_command="auto",
    )
    after_index = RepositoryIndex(
        methods=[
            _method("big", "<unknown>", 1, 5, 4, f),
            _method("_helper_a", "<unknown>", 7, 9, 3, f),
            _method("_helper_b", "<unknown>", 11, 13, 2, f),
        ]
    )

    result = _complexity_after(after_index, tmp_path, baseline)

    assert result.file_before == 9
    assert result.file_after == 9  # 4 + 3 + 2: helpers counted, so the move nets zero
    assert result.target_cc_after_max == 4  # the target method itself did drop
    # LOC tracked alongside CC: big shrank 30 -> 5 lines, and a per-method record exists
    assert result.target_loc_before_max == 30
    assert result.target_loc_after_max == 5
    assert len(result.records) == 1
    rec = result.records[0]
    assert (rec.method, rec.matched, rec.cc_after, rec.loc_after) == ("big", True, 4, 5)


def test_syntax_error_files_flags_broken_python(tmp_path: Path) -> None:
    (tmp_path / "good.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    (tmp_path / "bad.py").write_text("def f(:\n    return 1\n", encoding="utf-8")  # unclosed paren

    broken = _syntax_error_files(tmp_path)

    assert broken == frozenset({Path("bad.py")})


def test_analyze_after_reports_newly_introduced_syntax_error(tmp_path: Path) -> None:
    f = tmp_path / "mod.py"
    f.write_text("def target():\n    return 1\n", encoding="utf-8")
    baseline = Baseline(
        target_methods=[],
        file_complexity_before={},
        broken_files=frozenset(),
        tests_passed=None,
        test_command="auto",
    )

    # simulate a tool leaving the file broken
    f.write_text("def target(:\n    return 1\n", encoding="utf-8")

    analysis = _analyze_after(tmp_path, SmellType.LONG_METHOD, 0, baseline, "")

    assert analysis.syntax_broken_files == 1
    assert "mod.py" in analysis.error


def test_analyze_after_ignores_preexisting_syntax_error(tmp_path: Path) -> None:
    f = tmp_path / "mod.py"
    f.write_text("def already_broken(:\n    return 1\n", encoding="utf-8")
    baseline = Baseline(
        target_methods=[],
        file_complexity_before={},
        broken_files=frozenset({Path("mod.py")}),
        tests_passed=None,
        test_command="auto",
    )

    analysis = _analyze_after(tmp_path, SmellType.LONG_METHOD, 0, baseline, "")

    assert analysis.syntax_broken_files == 0
    assert analysis.error == ""


def _long_method_src(name: str) -> str:
    body = "\n".join(f"    x{i} = {i}" for i in range(25))  # 25 statements > threshold 20
    return f"def {name}():\n{body}\n"


def test_analyze_after_scopes_smells_to_baseline_source_files(tmp_path: Path) -> None:
    # A pristine file whose long method the tool left unfixed, plus a scratch
    # file the tool created that ALSO has a long method. Only the pristine one
    # should count -- the scratch file is the tool's own litter.
    (tmp_path / "mod.py").write_text(_long_method_src("real"), encoding="utf-8")
    (tmp_path / "scratch_copy.py").write_text(_long_method_src("junk"), encoding="utf-8")
    baseline = Baseline(
        target_methods=[],
        file_complexity_before={},
        broken_files=frozenset(),
        tests_passed=None,
        test_command="auto",
        source_files=frozenset({Path("mod.py")}),
    )

    analysis = _analyze_after(tmp_path, SmellType.LONG_METHOD, 1, baseline, "")

    assert analysis.smells_after == 1  # scratch_copy.py's long method is excluded

    # Sanity: with scoping disabled (None), the scratch file inflates the count.
    baseline_unscoped = Baseline(
        target_methods=[],
        file_complexity_before={},
        broken_files=frozenset(),
        tests_passed=None,
        test_command="auto",
    )
    analysis_unscoped = _analyze_after(tmp_path, SmellType.LONG_METHOD, 1, baseline_unscoped, "")
    assert analysis_unscoped.smells_after == 2


def test_analyze_after_ignores_syntax_error_in_scratch_file(tmp_path: Path) -> None:
    (tmp_path / "mod.py").write_text("def target():\n    return 1\n", encoding="utf-8")
    # tool leaves a broken scratch file it created, but never touched mod.py
    (tmp_path / "patch_helper.py").write_text("def broken(:\n    return 1\n", encoding="utf-8")
    baseline = Baseline(
        target_methods=[],
        file_complexity_before={},
        broken_files=frozenset(),
        tests_passed=None,
        test_command="auto",
        source_files=frozenset({Path("mod.py")}),
    )

    analysis = _analyze_after(tmp_path, SmellType.LONG_METHOD, 0, baseline, "")

    assert analysis.syntax_broken_files == 0  # scratch file's breakage isn't the repo's
    assert analysis.error == ""


def _fake_verification(passed: bool | None, index: RepositoryIndex) -> VerificationResult:
    if passed is None:
        return VerificationResult(command=None, returncode=0, stdout="", stderr="", index=index)
    return VerificationResult(
        command=["pytest"], returncode=0 if passed else 1, stdout="", stderr="", index=index
    )


@patch("refract.benchmark.runner.verify")
def test_analyze_after_reports_test_failure(mock_verify, tmp_path: Path) -> None:
    mock_verify.return_value = _fake_verification(False, RepositoryIndex())
    baseline = Baseline(
        target_methods=[],
        file_complexity_before={},
        broken_files=frozenset(),
        tests_passed=True,
        test_command="auto",
    )

    analysis = _analyze_after(tmp_path, SmellType.LONG_METHOD, 0, baseline, "")

    assert analysis.tests_passed is False


@patch("refract.benchmark.runner.verify")
def test_analyze_after_tests_passed_none_when_no_command_detected(
    mock_verify, tmp_path: Path
) -> None:
    mock_verify.return_value = _fake_verification(None, RepositoryIndex())
    baseline = Baseline(
        target_methods=[],
        file_complexity_before={},
        broken_files=frozenset(),
        tests_passed=None,
        test_command="auto",
    )

    analysis = _analyze_after(tmp_path, SmellType.LONG_METHOD, 0, baseline, "")

    assert analysis.tests_passed is None


@patch("refract.benchmark.runner.verify")
def test_analyze_after_reports_test_success(mock_verify, tmp_path: Path) -> None:
    mock_verify.return_value = _fake_verification(True, RepositoryIndex())
    baseline = Baseline(
        target_methods=[],
        file_complexity_before={},
        broken_files=frozenset(),
        tests_passed=True,
        test_command="auto",
    )

    analysis = _analyze_after(tmp_path, SmellType.LONG_METHOD, 0, baseline, "")

    assert analysis.tests_passed is True


def test_cap_flagged_matches_run_refactors_own_sort_and_limit() -> None:
    f = Path("mod.py")
    # deliberately out of order, so this also proves the sort, not just the cap
    flagged = [
        _smell(f, 30, identifier="c"),
        _smell(f, 10, identifier="a"),
        _smell(f, 20, identifier="b"),
    ]

    capped = _cap_flagged(flagged, limit=2)

    assert [s.identifier for s in capped] == ["a", "b"]


def test_cap_flagged_limit_zero_or_negative_yields_nothing() -> None:
    flagged = [_smell(Path("mod.py"), 1, identifier="a")]

    assert _cap_flagged(flagged, limit=0) == []
    assert _cap_flagged(flagged, limit=-5) == []


def test_agentic_prompt_gives_reachable_goal_when_capped() -> None:
    prompt = _agentic_prompt(
        SmellType.LONG_METHOD,
        "- mod.py:10 - ...",
        target_count=10,
        smells_before=458,
        checker_cmd="check",
    )

    assert "448" in prompt  # 458 - 10: the reachable stopping count
    assert "not 0" in prompt


def test_agentic_prompt_allows_zero_when_targets_cover_everything() -> None:
    prompt = _agentic_prompt(
        SmellType.LONG_METHOD,
        "- mod.py:10 - ...",
        target_count=51,
        smells_before=51,
        checker_cmd="check",
    )

    assert "0 smells" in prompt
    assert "reachable" in prompt


def test_print_report_renders_tests_column(capsys) -> None:
    results = [
        ToolResult(
            tool="refract",
            model="m",
            api_calls=1,
            input_tokens=1,
            output_tokens=1,
            smells_before=5,
            smells_after=3,
            tests_passed=True,
        ),
        ToolResult(
            tool="codex",
            model="m",
            api_calls=2,
            input_tokens=2,
            output_tokens=2,
            smells_before=5,
            smells_after=1,
            tests_passed=None,
        ),
    ]

    print_report(results)
    output = capsys.readouterr().out

    assert "PASS" in output
    assert "n/a" in output


def test_print_report_renders_oracle_columns(capsys) -> None:
    results = [
        ToolResult(
            tool="refract",
            model="m",
            api_calls=1,
            input_tokens=1,
            output_tokens=1,
            smells_before=5,
            smells_after=3,
            oracle_smells_before=6,
            oracle_smells_after=4,
        ),
        ToolResult(
            tool="opencode",
            model="m",
            api_calls=1,
            input_tokens=1,
            output_tokens=1,
            smells_before=5,
            smells_after=4,
            oracle_smells_before=None,
            oracle_smells_after=None,
        ),
    ]

    print_report(results)
    output = capsys.readouterr().out

    assert "Oracle Before" in output
    assert "Oracle After" in output
    # unvalidated tool shows n/a rather than 0
    assert "n/a" in output


def test_oracle_fixed_property() -> None:
    validated = ToolResult(
        tool="x",
        model="m",
        api_calls=0,
        input_tokens=0,
        output_tokens=0,
        smells_before=5,
        smells_after=2,
        oracle_smells_before=6,
        oracle_smells_after=4,
    )
    unvalidated = ToolResult(
        tool="y",
        model="m",
        api_calls=0,
        input_tokens=0,
        output_tokens=0,
        smells_before=5,
        smells_after=2,
    )

    assert validated.oracle_fixed == 2
    assert unvalidated.oracle_fixed is None


def _fake_tool_runner(tool: str):
    def _run(*_args, **_kwargs):
        return ToolResult(
            tool=tool,
            model="m",
            api_calls=0,
            input_tokens=0,
            output_tokens=0,
            smells_before=0,
            smells_after=0,
        )

    return _run


def test_run_benchmark_skips_done_tools_and_streams_each_result(tmp_path, monkeypatch) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "m.py").write_text("def f():\n    return 42\n")

    # avoid any real tool/LLM/oracle work; just record which tools ran
    monkeypatch.setattr("refract.benchmark.runner.oracle_count_smells", lambda *a, **k: None)
    monkeypatch.setattr("refract.benchmark.runner._run_refract", _fake_tool_runner("refract"))
    monkeypatch.setattr("refract.benchmark.runner._run_codex", _fake_tool_runner("codex"))
    monkeypatch.setattr("refract.benchmark.runner._run_opencode", _fake_tool_runner("opencode"))
    monkeypatch.setattr("refract.benchmark.runner._run_gemini", _fake_tool_runner("gemini"))

    streamed: list[str] = []
    results = run_benchmark(
        repo=repo,
        smell_type=SmellType.MAGIC_NUMBER,
        model="m",
        api_key="k",
        tools=["codex", "opencode", "gemini"],
        gemini_api_key="k",
        refract_provider="gemini",
        done_tools={"refract", "codex"},
        on_result=lambda r: streamed.append(r.tool),
    )

    ran = [r.tool for r in results]
    assert ran == ["opencode", "gemini"]  # refract + codex already done -> skipped
    assert streamed == ["opencode", "gemini"]  # each streamed the moment it finished


def test_run_benchmark_keeps_workdir_when_requested(tmp_path, monkeypatch) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "m.py").write_text("def f():\n    return 42\n")

    monkeypatch.setattr("refract.benchmark.runner.oracle_count_smells", lambda *a, **k: None)
    monkeypatch.setattr("refract.benchmark.runner._run_refract", _fake_tool_runner("refract"))

    workdir = tmp_path / "kept"
    run_benchmark(
        repo=repo,
        smell_type=SmellType.MAGIC_NUMBER,
        model="m",
        api_key="k",
        tools=[],
        refract_provider="gemini",
        gemini_api_key="k",
        workdir=workdir,
    )

    # the patched copy survives the run instead of being cleaned up
    assert (workdir / "refract_copy" / "m.py").exists()


def test_rerun_for_other_tool_preserves_done_refract_copy(tmp_path, monkeypatch) -> None:
    """Regression: resuming a cell for another tool (refract already done) must
    not clobber the saved refract_copy. The old code re-cloned refract_copy
    unconditionally before the done check, wiping refract's patched output."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "m.py").write_text("def f():\n    return 42\n")

    monkeypatch.setattr("refract.benchmark.runner.oracle_count_smells", lambda *a, **k: None)
    monkeypatch.setattr("refract.benchmark.runner._run_codex", _fake_tool_runner("codex"))

    # simulate a prior run: refract already produced a patched copy
    workdir = tmp_path / "kept"
    refract_copy = workdir / "refract_copy"
    refract_copy.mkdir(parents=True)
    (refract_copy / "m.py").write_text("def f():\n    return SHIFT  # refract edit\n")

    run_benchmark(
        repo=repo,
        smell_type=SmellType.MAGIC_NUMBER,
        model="m",
        api_key="k",
        tools=["codex"],
        gemini_api_key="k",
        refract_provider="gemini",
        done_tools={"refract"},
        workdir=workdir,
    )

    # refract's patched output is untouched, and the throwaway baseline copy is
    # not left behind
    assert (refract_copy / "m.py").read_text() == "def f():\n    return SHIFT  # refract edit\n"
    assert not (workdir / "_baseline_copy").exists()


def test_clean_subprocess_stderr_drops_node_noise_keeps_real_errors() -> None:
    noise = (
        "(node:22484) [DEP0040] DeprecationWarning: The `punycode` module is deprecated.\n"
        "(Use `node --trace-deprecation ...` to show where the warning was created)"
    )
    assert _clean_subprocess_stderr(noise) == ""
    mixed = "(node:1) DeprecationWarning: whatever\nError: real failure here"
    assert _clean_subprocess_stderr(mixed) == "Error: real failure here"
    assert _clean_subprocess_stderr("") == ""


def test_no_calls_error_flags_silent_tool() -> None:
    assert "no API calls" in _no_calls_error("opencode", 0)
    assert _no_calls_error("gemini", 12) == ""


def test_tool_result_derived_properties() -> None:
    result = ToolResult(
        tool="x",
        model="m",
        api_calls=1,
        input_tokens=1,
        output_tokens=1,
        smells_before=10,
        smells_after=4,
        complexity_before=20,
        complexity_after=12,
    )

    assert result.fixed == 6
    assert result.complexity_delta == -8
