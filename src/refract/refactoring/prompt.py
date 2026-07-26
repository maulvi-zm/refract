from __future__ import annotations

from refract.planning.context import RefactorContext
from refract.refactoring.proposal import RefactorProposal


def build_system_prompt(language: str) -> str:
    return (
        f"You are Refract, a {language} refactoring assistant.\n"
        "Return only JSON with keys: explanation, edits, confidence. Keep "
        "`explanation` to one short sentence. `edits` is a "
        "list of {old_snippet, new_snippet} objects, applied in order to the SAME "
        "file.\n"
        "Use several edits when the fix needs changes in more than one place -- "
        "extract-constant is one edit that adds the constant definition and another "
        "that rewrites the use site; extract-method adds the helper and replaces the "
        "block with a call; renaming an identifier is one edit per occurrence. Do NOT "
        "cram multiple locations into a single snippet.\n"
        "For a magic-number fix, you MUST put your chosen UPPER_SNAKE_CASE name in the "
        "`constant_name` field: Refract does the extraction itself -- it adds the "
        "definition at module scope and rewrites ONLY the flagged literal -- so you do "
        "not touch the surrounding line at all. Your `edits` are IGNORED for a magic "
        "number (naming is your only job); a magic-number fix with an empty "
        "`constant_name` is rejected. Never change the surrounding expression. "
        'Leave `constant_name` empty ("") for every other kind of fix.\n'
        "Place a new module-level constant on its own line at the top of the file. "
        "Anchor the edit that adds it on a line copied verbatim from the `File header` "
        "section below -- typically the last import line -- so it lands at module "
        "scope. NEVER anchor it on a decorator (a line starting with @) or a def/class "
        "line, and never insert a statement between a decorator and the function or "
        "class it decorates, inside a call's parentheses, or in the middle of the "
        "imports -- each of those does not parse. So a magic number inside a decorator "
        "like @option(default=2.0) is fixed by adding the constant after the imports "
        "and changing the argument to default=THE_CONSTANT.\n"
        "Each old_snippet must be copied verbatim from the supplied source context, "
        "character for character, including whitespace and indentation, and must match "
        "exactly one location in the file.\n"
        "Make each edit the SMALLEST change that fixes the smell. old_snippet must be "
        "the shortest span that still matches exactly one location -- ideally the single "
        "line or expression that changes, plus only enough surrounding text to be unique. "
        "new_snippet is that same span with only the necessary change applied. A bare "
        "literal or short token (like 1.0 or 90) usually occurs many times -- include "
        "the surrounding assignment, call, or argument so old_snippet matches exactly "
        "one location. NEVER "
        "restate unchanged lines or copy a whole function body just to alter a few tokens; "
        "the two snippets should differ only by the edited code. When a fix genuinely adds "
        "new code (e.g. an extract-method helper), that helper is its own edit and the "
        "block it replaces becomes just the short call site.\n"
        "Each new_snippet must be complete and runnable for the span it covers -- never a "
        "placeholder comment or stub body.\n"
        "For extract-method, MOVE a contiguous block VERBATIM into the new helper and "
        "replace it with a call -- do NOT rewrite, simplify, merge, reorder, or otherwise "
        "change the logic, conditionals, signs, or return values of the code you move; copy "
        "it character for character. Define the helper as its own top-level function BEFORE "
        "or AFTER the original (never in the middle of a function body) and pass it the "
        "variables it needs; the original keeps its return statement, now calling the helper. "
        "A refactor that alters behavior to look cleaner is WRONG -- prefer leaving the smell "
        "to changing what the code does.\n"
        "The edits together must preserve all existing behavior while addressing the "
        "requested code smell.\n"
        "Do not include markdown, surrounding prose, or unrelated edits."
    )


def build_user_prompt(context: RefactorContext) -> str:
    fence = context.fence or ""

    return "\n".join(
        [
            f"Smell: {context.smell.smell.value}",
            f"Location: {context.smell.file}:{context.smell.line}",
            f"Identifier: {context.smell.identifier}",
            f"Detail: {context.smell.detail}",
            _method_summary(context),
            "",
            "Same-file methods:",
            _method_list(context.same_file_methods, with_file=False),
            "",
            "Caller hints:",
            _method_list(context.callers, with_file=True),
            "",
            "Callee hints:",
            _method_list(context.callees, with_file=True),
            "",
            "Relevant constants:",
            "\n".join(f"- {c}" for c in context.constants) or "- none",
            "",
            _file_header_section(context),
            f"Source context starts at line {context.snippet_start_line}:",
            f"```{fence}",
            context.snippet,
            "```",
        ]
    )


def _file_header_section(context: RefactorContext) -> str:
    """Show the top of the file so the model can anchor a module-level constant on
    real header text (an import line) instead of guessing. Empty when the source
    context already starts at the top of the file."""
    if not context.file_header:
        return ""
    fence = context.fence or ""
    return "\n".join(
        [
            "File header (top of the file -- add any new module-level constant here, "
            "anchored on a line shown below, never on a decorator or def):",
            f"```{fence}",
            context.file_header,
            "```",
            "",
        ]
    )


def _method_summary(context: RefactorContext) -> str:
    method = context.target_method
    if method is None:
        return "No containing method was found."

    return (
        f"Containing method: {method.class_name}.{method.name}, "
        f"lines {method.start_line}-{method.end_line}, "
        f"CC {method.cyclomatic_complexity}, "
        f"parameters {method.parameters}, calls {method.calls}."
    )


def build_repair_prompt(base_user_prompt: str, proposal: RefactorProposal, error: str) -> str:
    """Re-ask the model after its previous proposal was rejected and NOT applied.

    Carries the full original context plus the concrete rejection reason and the
    exact edits that failed, so the model can correct the specific mistake (a
    snippet that isn't verbatim, an ambiguous match, or a patch that doesn't
    parse) rather than guessing. The file is untouched -- a rejected attempt
    never writes -- so the corrected edits still apply against the same source.
    """
    edits_desc = "\n".join(
        f"  edit {i}:\n    old_snippet: {edit.old_snippet!r}\n    new_snippet: {edit.new_snippet!r}"
        for i, edit in enumerate(proposal.edits, start=1)
    )
    return "\n".join(
        [
            base_user_prompt,
            "",
            "Your previous proposal was REJECTED and NOT applied. Correct it and "
            "return a new proposal.",
            f"Rejection reason: {error}",
            "The edits you proposed were:",
            edits_desc,
            "",
            "Likely causes: an old_snippet that is not present verbatim in the source "
            "above (match whitespace and indentation exactly); an old_snippet that "
            "occurs more than once (extend it until it matches exactly one location); "
            "or edits that together leave the file unparseable -- most often a new "
            "definition placed where a statement is illegal (between a decorator and "
            "its function/class, inside a call's parentheses, or splitting the "
            "imports). Put extracted constants at the top of the file after the "
            "imports. Return edits that apply cleanly and preserve behavior.",
        ]
    )


def _method_list(methods, with_file: bool) -> str:
    lines = [
        f"- {m.class_name}.{m.name}"
        + (f" in {m.file}" if with_file else "")
        + f" lines {m.start_line}-{m.end_line}"
        for m in methods
    ]
    return "\n".join(lines) or "- none"
