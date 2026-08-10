<p align="center">
  <img src="./docs/images/refraction.jpg" alt="refraction-image" width="800">
</p>

# refract

Multi-language CLI that detects code smells with [tree-sitter](https://tree-sitter.github.io/)
and proposes LLM-based refactors, then verifies them by re-running the project's tests.

- **Languages:** Java and Python
- **Smells:** long method, long identifier, magic number
- **Providers:** OpenAI, Google Gemini

Detection is done entirely from the AST (no external linters), so every language
goes through one code path. Adding a language means writing a single
`LanguageSpec` (see `src/refract/languages/`).

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) for dependency management
- For `refract verify`: the target project's build/test tool (`mvn`/`gradle` for
  Java, `pytest` for Python)

## Setup

```sh
uv sync
```

Copy `.env.example` to `.env` and set an API key for the provider you'll use.
Every supported variable is documented in [.env.example](.env.example); refract
loads `.env` automatically (exported variables take precedence).

```sh
cp .env.example .env
$EDITOR .env
```

Check that keys, languages, and build tools are reachable:

```sh
uv run refract doctor
```

## Usage

All commands share one `--db` SQLite file: `index` creates it; `refactor` and
`verify` read and update it. Reuse the same path for a given repo.

Index a source tree:

```sh
uv run refract index path/to/repo --db out.db
```

Plan a refactor for a detected smell (`long_method`, `long_identifier`, or
`magic_number`) without touching files:

```sh
uv run refract refactor path/to/repo --db out.db --smell magic_number --dry-run
```

Apply it (requires a clean git working tree in the target repo unless
`--allow-dirty`):

```sh
uv run refract refactor path/to/repo --db out.db --smell magic_number --apply
```

Scope a run to one file with `--file`, so only smells detected in that file are
refactored:

```sh
uv run refract refactor path/to/repo --db out.db --smell magic_number --file src/main/java/com/demo/CartService.java --dry-run
```

The path may be absolute or relative to the repo root (not to your shell's
working directory), and must point at an existing file — refract exits with
`No such file: ...` otherwise. Filtering happens before `--limit`, so
`--file X --limit 3` means the first three smells *in X*. The file still has to
be indexed: `--file` narrows what gets refactored, it does not index anything
new, so run `refract index` first.

Select the provider/model per command with `--provider`/`--model`, or set
`REFRACT_PROVIDER`/`REFRACT_MODEL` for the session.

Verify by re-running the repo's tests and re-indexing (`auto` detects the tool):

```sh
uv run refract verify path/to/repo --db out.db --test-command auto
```

The evaluation harness used for the thesis (benchmarking refract against
agentic CLIs, the counting proxy, and the DesigniteJava/DPy oracles) lives on
the [`benchmark`](../../tree/benchmark) branch, not on `main`.

## How it works

Each `refract refactor` call runs this pipeline:

1. **Index** (`indexing/`): walk the tree, parse each file with tree-sitter,
   extract per-method info (complexity, parameters, calls), and detect smells
   from the AST. Results are stored in the `--db` SQLite file.
2. **Plan** (`planning/`): for a chosen smell, gather the target method plus
   same-file methods, callers, callees, and relevant constants, bounded by
   method boundaries.
3. **Propose** (`refactoring/`): send that context to the LLM provider and parse
   the response into an explanation, an old/new snippet pair, and a confidence.
4. **Patch**: with `--apply`, replace the old snippet with the new one on disk.
5. **Verify** (`verification/`): re-run the project's tests and re-index.

## Adding a language

1. Add `src/refract/languages/<lang>.py` defining a `LanguageSpec` (tree-sitter
   queries, node-type sets, thresholds).
2. Register it in `src/refract/languages/registry.py`.

No other module changes.

## Development

```sh
uv run python -m compileall src     # syntax check
uv run ruff format src tests        # format
uv run ruff check src tests         # lint
uv run pytest                       # tests
```
