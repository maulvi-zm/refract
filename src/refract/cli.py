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

    bench_parser = subparsers.add_parser("benchmark", help="Compare refract vs Codex CLI")
    bench_parser.add_argument("repo", type=Path)
    bench_parser.add_argument("--smell", choices=_SMELL_CHOICES, required=True)
    bench_parser.add_argument("--model", default="gpt-4o-mini")
    bench_parser.add_argument("--limit", type=int, default=10)
    bench_parser.add_argument("--verbose", action="store_true")
    bench_parser.add_argument(
        "--codex-api-key-mode",
        action="store_true",
        help="Route codex through the counting proxy using OPENAI_API_KEY "
        "(requires a verified OpenAI organization).",
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
    )

    mode = "applied" if args.apply else "planned"
    print(f"{len(results)} refactor(s) {mode} with {config.provider.value}/{config.model}.")

    for result in results:
        smell = result.context.smell
        print(f"\n{smell.file}:{smell.line} {smell.smell.value}")
        print(result.proposal.explanation)
        if not args.apply:
            print("\n--- old snippet ---")
            print(result.proposal.old_snippet)
            print("--- new snippet ---")
            print(result.proposal.new_snippet)

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


def _cmd_benchmark(args: argparse.Namespace) -> None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is not set.")

    # import here so the cli still loads without the benchmark bits
    from refract.benchmark.report import print_report
    from refract.benchmark.runner import run_benchmark

    print(f"Benchmarking refract vs codex on {args.repo} ({args.smell}, model={args.model})")
    results = run_benchmark(
        repo=args.repo.resolve(),
        smell_type=SmellType(args.smell),
        model=args.model,
        api_key=api_key,
        limit=args.limit,
        codex_api_key_mode=args.codex_api_key_mode,
        verbose=args.verbose,
    )
    print_report(results)


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
