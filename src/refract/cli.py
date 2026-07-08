from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

from refract.core.config import load_dotenv
from refract.core.models import RepositoryIndex, SmellType
from refract.indexing.database import load, save
from refract.indexing.repository import index_repository
from refract.languages.registry import all_specs
from refract.refactoring.pipeline import run_refactor
from refract.refactoring.proposal import ProviderName
from refract.refactoring.providers import (
    DEFAULT_MODELS,
    config_from_env,
    provider_from_config,
)
from refract.verification.runner import verify

_SMELL_CHOICES = [smell.value for smell in SmellType]
_PROVIDER_CHOICES = [provider.value for provider in ProviderName]


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    parser = _build_parser()
    args = parser.parse_args(argv)
    args.func(args)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="refract")
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="Index a source tree")
    index_parser.add_argument("repo", type=Path)
    index_parser.add_argument("--db", type=Path, required=True)
    index_parser.set_defaults(func=_cmd_index)

    refactor_parser = subparsers.add_parser("refactor", help="Plan or apply one refactoring")
    refactor_parser.add_argument("repo", type=Path)
    refactor_parser.add_argument("--db", type=Path, required=True)
    refactor_parser.add_argument("--provider", choices=_PROVIDER_CHOICES)
    refactor_parser.add_argument("--model")
    refactor_parser.add_argument("--smell", choices=_SMELL_CHOICES, required=True)
    refactor_parser.add_argument("--limit", type=int, default=10)
    refactor_parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Inference calls per target: the first proposal plus feedback-driven "
        "retries when an edit is rejected (bad snippet, ambiguous match, "
        "unparseable patch). 1 disables retries. Default: 3.",
    )
    action = refactor_parser.add_mutually_exclusive_group()
    action.add_argument("--dry-run", action="store_true", default=True)
    action.add_argument("--apply", action="store_true")
    refactor_parser.add_argument("--allow-dirty", action="store_true")
    refactor_parser.set_defaults(func=_cmd_refactor)

    verify_parser = subparsers.add_parser("verify", help="Run the repo's tests and re-check smells")
    verify_parser.add_argument("repo", type=Path)
    verify_parser.add_argument("--db", type=Path, required=True)
    verify_parser.add_argument("--test-command", default="auto")
    verify_parser.set_defaults(func=_cmd_verify)

    doctor_parser = subparsers.add_parser("doctor", help="Report local setup readiness")
    doctor_parser.set_defaults(func=_cmd_doctor)

    check_parser = subparsers.add_parser(
        "check", help="Report remaining smells of one type (exit 1 if any remain)"
    )
    check_parser.add_argument("repo", type=Path)
    check_parser.add_argument("--smell", choices=_SMELL_CHOICES, required=True)
    check_parser.set_defaults(func=_cmd_check)

    bench_parser = subparsers.add_parser(
        "benchmark", help="Compare refract vs agentic CLIs (codex, opencode, gemini)"
    )
    bench_parser.add_argument("repo", type=Path)
    bench_parser.add_argument("--smell", choices=_SMELL_CHOICES, required=True)
    bench_parser.add_argument("--model", default="gpt-4o-mini")
    bench_parser.add_argument("--limit", type=int, default=10)
    bench_parser.add_argument("--verbose", action="store_true")
    bench_parser.add_argument(
        "--tools",
        default="codex",
        help="Comma-separated agentic tools to run against refract "
        "(codex, opencode, gemini). Default: codex.",
    )
    bench_parser.add_argument(
        "--gemini-model",
        default="gemini-2.5-flash",
        help="Model for the gemini tool (Gemini talks to Google, not OpenAI).",
    )
    bench_parser.add_argument(
        "--codex-api-key-mode",
        action="store_true",
        help="Route codex through the counting proxy using OPENAI_API_KEY "
        "(requires a verified OpenAI organization).",
    )
    bench_parser.add_argument(
        "--refract-provider",
        choices=["openai", "gemini"],
        default="openai",
        help="Provider backing refract's own baseline run. 'gemini' interprets "
        "--model/the provider key as Gemini instead of OpenAI, so the whole "
        "run can go through a single Gemini key (see "
        "references/gemini-provider-setup.md for the full same-model setup "
        "across all four tools). Default: openai.",
    )
    bench_parser.add_argument(
        "--test-command",
        default="auto",
        help="Override how each tool's after-state is test-verified, same syntax "
        "as `refract verify --test-command`. 'auto' (default) detects "
        "mvn/gradle/pytest; pass an explicit command for repos that need one "
        "(e.g. a multi-module Maven project where the generic `mvn test` also "
        "builds unrelated submodules).",
    )
    bench_parser.add_argument(
        "--keep-workdir",
        type=Path,
        default=None,
        help="Persist each tool's patched repo copy under this directory instead "
        "of a self-deleting temp dir, so the actual edits and diffs are auditable "
        "after the run (refract_copy/, <tool>_copy/).",
    )
    bench_parser.add_argument(
        "--refract-max-attempts",
        type=int,
        default=3,
        help="Inference calls refract may spend per target: the first proposal "
        "plus feedback-driven retries when an edit is rejected. 1 disables retries "
        "(single-shot arm). Default: 3.",
    )
    bench_parser.set_defaults(func=_cmd_benchmark)

    return parser


def _cmd_index(args: argparse.Namespace) -> None:
    index = index_repository(args.repo)
    save(index, args.db)

    print(f"Indexed {len(index.methods)} methods into {args.db}")
    _print_smells(index)


def _cmd_refactor(args: argparse.Namespace) -> None:
    repo = args.repo.resolve()
    if args.apply and not args.allow_dirty:
        _require_clean_worktree(repo)

    index = load(args.db)
    config = config_from_env(args.provider, args.model)
    try:
        provider = provider_from_config(config)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    results = run_refactor(
        index=index,
        repo_root=repo,
        smell_type=SmellType(args.smell),
        limit=args.limit,
        provider=provider,
        apply=args.apply,
        max_attempts=args.max_attempts,
    )

    mode = "applied" if args.apply else "planned"
    print(f"{len(results)} refactor(s) {mode} with {config.provider.value}/{config.model}.")

    for result in results:
        smell = result.context.smell
        print(f"\n{smell.file}:{smell.line} {smell.smell.value}")
        print(result.proposal.explanation)
        if not args.apply:
            for i, edit in enumerate(result.proposal.edits, start=1):
                label = (
                    f" ({i}/{len(result.proposal.edits)})" if len(result.proposal.edits) > 1 else ""
                )
                print(f"\n--- old snippet{label} ---")
                print(edit.old_snippet)
                print(f"--- new snippet{label} ---")
                print(edit.new_snippet)

    if args.apply and results:
        updated = index_repository(repo)
        save(updated, args.db)
        print(f"\nIndex refreshed: {len(updated.smells)} smell(s) remaining.")


def _cmd_verify(args: argparse.Namespace) -> None:
    result = verify(args.repo.resolve(), args.test_command)
    save(result.index, args.db)

    print(f"Command: {' '.join(result.command) if result.command else 'none detected'}")
    print(f"Command status: {result.returncode}")
    print(f"Smells after verification: {len(result.smells)}")

    if result.stdout:
        print("\nstdout:\n" + result.stdout)
    if result.stderr:
        print("\nstderr:\n" + result.stderr)

    if not result.passed:
        raise SystemExit(result.returncode)


def _cmd_doctor(_: argparse.Namespace) -> None:
    print("refract doctor")

    print("\nLanguages:")
    for spec in all_specs():
        print(f"- {spec.name}: {', '.join(spec.extensions)}")

    print("\nBuild/test tools:")
    for tool in ("mvn", "gradle", "pytest", "git"):
        print(f"- {tool}: {shutil.which(tool) or 'missing'}")

    print("\nProviders:")
    env_keys = {
        ProviderName.OPENAI: "OPENAI_API_KEY",
        ProviderName.GEMINI: "GEMINI_API_KEY",
    }
    for provider in ProviderName:
        configured = bool(os.getenv(env_keys[provider]))
        state = "configured" if configured else "missing key (live calls disabled)"
        print(f"- {provider.value}: {state} (default model {DEFAULT_MODELS[provider]})")


def _cmd_check(args: argparse.Namespace) -> None:
    """Fresh re-index + count for one smell type. Exits 1 while any remain.

    This is the ground-truth oracle the benchmark hands to the agentic tools:
    they run it in their own loop and iterate until it reports 0.
    """
    repo = args.repo.resolve()
    index = index_repository(repo)
    smells = index.smells_by_type(SmellType(args.smell))

    print(f"{len(smells)} '{args.smell}' smell(s) remaining.")
    for smell in smells:
        try:
            location = smell.file.resolve().relative_to(repo)
        except ValueError:
            location = smell.file
        print(f"  {location}:{smell.line} - {smell.detail}")

    if smells:
        raise SystemExit(1)


def _cmd_benchmark(args: argparse.Namespace) -> None:
    # refract's own baseline run needs whichever key matches --refract-provider;
    # OpenAI stays the default so existing invocations behave unchanged.
    baseline_key_env = "GEMINI_API_KEY" if args.refract_provider == "gemini" else "OPENAI_API_KEY"
    api_key = os.getenv(baseline_key_env)
    if not api_key:
        raise SystemExit(f"{baseline_key_env} is not set.")

    # import here so the cli still loads without the benchmark bits
    from refract.benchmark.report import print_report
    from refract.benchmark.runner import AGENTIC_TOOLS, run_benchmark

    tools = [t.strip() for t in args.tools.split(",") if t.strip()]
    unknown = [t for t in tools if t not in AGENTIC_TOOLS]
    if unknown:
        raise SystemExit(
            f"Unknown --tools value(s): {', '.join(unknown)}. Choose from {', '.join(AGENTIC_TOOLS)}."
        )

    print(
        f"Benchmarking refract ({args.refract_provider}) vs {', '.join(tools)} on {args.repo} "
        f"({args.smell}, model={args.model})"
    )
    results = run_benchmark(
        repo=args.repo.resolve(),
        smell_type=SmellType(args.smell),
        model=args.model,
        api_key=api_key,
        limit=args.limit,
        tools=tools,
        codex_api_key_mode=args.codex_api_key_mode,
        gemini_model=args.gemini_model,
        gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
        refract_provider=args.refract_provider,
        test_command=args.test_command,
        verbose=args.verbose,
        workdir=args.keep_workdir.resolve() if args.keep_workdir else None,
        refract_max_attempts=args.refract_max_attempts,
    )
    print_report(results)
    if args.keep_workdir:
        print(f"\nPatched repos kept under {args.keep_workdir.resolve()}")


def _print_smells(index: RepositoryIndex) -> None:
    for smell_type in SmellType:
        smells = index.smells_by_type(smell_type)
        print(f"\n=== {smell_type.value} ({len(smells)}) ===")
        for smell in smells:
            print(f"  {smell.file.name}:{smell.line} - {smell.detail}")


def _require_clean_worktree(repo: Path) -> None:
    root = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "--show-toplevel"],
        check=False,
        capture_output=True,
        text=True,
    )
    if root.returncode != 0:
        return  # not a git repo, nothing to check

    status = subprocess.run(
        ["git", "-C", root.stdout.strip(), "status", "--porcelain"],
        check=False,
        capture_output=True,
        text=True,
    )
    if status.stdout.strip():
        raise SystemExit(
            "Target git worktree is dirty. Commit or stash changes, or pass --allow-dirty."
        )


if __name__ == "__main__":
    main()
