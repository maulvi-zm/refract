# refract — thesis defense demo script

Target repo: `demo/cart-service` — a single-class Maven project (`CartService.java`)
with a JUnit 5 suite of 6 tests. It is its own git repo, seeded at commit
`3f8db01` with exactly three smells:

| smell | where | detail |
|---|---|---|
| `long_method` | `checkout` | 27 statements (threshold 20) |
| `long_identifier` | `taxAmountForDiscountedOrder` | 27 chars (threshold 17) |
| `magic_number` | throughout `checkout` | 14 literals (0.25, 0.05, 500.0, 10, 0.30, 0.08, 15.0, 100.0, 25.0) |

Run everything from the repo root: `/Users/maulvizm/Documents/TA/refract-next`.

## One command for the whole thing

There is no single refract subcommand that chains index → refactor → verify
(`benchmark` does, but on throwaway copies and against competitor tools). Use the
driver script, which runs the steps below in order and pauses before each one so
you can talk over it:

```sh
./demo/run_demo.sh          # pauses before each step
./demo/run_demo.sh --auto   # no pauses, straight through
```

It resets the repo to the seed commit itself, so it is safe to re-run between takes.
The manual sequence below is the same thing, step by step, if you'd rather type.

## 0. Reset before each take

```sh
git -C demo/cart-service reset --hard 3f8db01
rm -f demo/demo.db
```

## 1. Show the repo and that it is green

```sh
bat demo/cart-service/src/main/java/com/demo/CartService.java   # or `cat`
mvn -f demo/cart-service/pom.xml -o test
```

Expect: `Tests run: 6, Failures: 0, Errors: 0`.

## 2. Index — detection only, no LLM

```sh
uv run refract index demo/cart-service --db demo/demo.db
```

Expect: `Indexed 2 methods`, then the three smell sections (1 / 1 / 14).

## 3. Propose without touching the disk

```sh
uv run refract refactor demo/cart-service --db demo/demo.db --smell magic_number --limit 3 --dry-run
```

Shows explanation plus the old/new snippet pairs (constant hoisted into the class,
literal replaced at the use site). Nothing written.

## 4. Apply, verify — magic_number

```sh
refract refactor demo/cart-service --db demo/demo.db --smell magic_number --limit 3 --apply
refract verify demo/cart-service --db demo/demo.db --test-command auto
git -c demo/cart-service diff
```

`verify` prints `Command: mvn test`, `Command status: 0`, `Smells after verification: 16`.
(Full Maven output follows — `| head -3` keeps the frame clean. Drop the pipe if you
want the test run visible on camera.)

Commit so the next step starts from a clean worktree (refract refuses a dirty tree
unless `--allow-dirty`):

```sh
git -C demo/cart-service commit -am "refactor: name discount constants"
```

## 5. Apply, verify — long_identifier

Re-index first: a refactor plans against the last index, so after the previous
step's edits the snippets sent to the provider would otherwise be stale (symptom:
`Provider old_snippet was not found in the target file`).

```sh
refract index demo/cart-service --db demo/demo.db
refract refactor demo/cart-service --db demo/demo.db --smell long_identifier --apply
refract verify demo/cart-service --db demo/demo.db --test-command auto
git -C demo/cart-service commit -am "refactor: shorten identifier"
```

`taxAmountForDiscountedOrder` becomes `taxAmount`, renamed at every occurrence.

## 6. Apply, verify — long_method

```sh
refract index demo/cart-service --db demo/demo.db
refract refactor demo/cart-service --db demo/demo.db --smell long_method --apply
refract verify demo/cart-service --db demo/demo.db --test-command auto
git -C demo/cart-service diff
```

`checkout` shrinks to a few lines; helpers (`calculateDiscountRate`,
`calculateShipping`, …) appear at class scope. Tests still pass.

## 7. Close on the scoreboard

```sh
refract check demo/cart-service --smell long_method
refract check demo/cart-service --smell long_identifier
refract check demo/cart-service --smell magic_number
```

Rehearsal result: `long_method` 0 remaining, `magic_number` 14 → 11 (only 3 were
requested via `--limit 3`), `long_identifier` 1 remaining.

## Notes for the narration

- Output is LLM-generated, so names and counts shift between takes. The invariants
  are: tests stay green, `long_method` reaches 0, and every `--limit N` run fixes N.
- In rehearsal the last `long_identifier` came from a constant refract itself named
  (`HIGH_SUBTOTAL_THRESHOLD`, 23 chars). Good beat if it reappears: the detector
  holds the tool's own output to the same threshold, and the next pass would fix it.
- Guard rails worth mentioning if a warning shows on camera — refract rejects an
  edit that breaks the parse, orphans code, or leaves a rename half-done, and
  retries up to 3 times instead of writing it.
- `uv run refract doctor` is a good 10-second opener: it lists languages, build
  tools, and configured providers.
