from __future__ import annotations

from refract.planning.context import RefactorContext
from refract.refactoring.proposal import RefactorProposal


def build_system_prompt(language: str) -> str:
    return (
        f"You are Refract, a {language} refactoring assistant.\n"
        "Return only JSON with keys: explanation, edits, confidence. `edits` is a "
        "list of {old_snippet, new_snippet} objects, applied in order to the SAME "
        "file.\n"
        "Use several edits when the fix needs changes in more than one place -- "
        "extract-constant is one edit that adds the constant definition and another "
        "that rewrites the use site; extract-method adds the helper and replaces the "
        "block with a call; renaming an identifier is one edit per occurrence. Do NOT "
        "cram multiple locations into a single snippet.\n"
        "Each old_snippet must be copied verbatim from the supplied source context, "
        "character for character, including whitespace and indentation, and must match "
        "exactly one location in the file.\n"
        "Each new_snippet must contain the complete, working replacement. Never use "
        "placeholder comments or stub bodies. Copy every line of logic from the original "
        "and restructure it.\n"
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
            f"Source context starts at line {context.snippet_start_line}:",
            f"```{fence}",
            context.snippet,
            "```",
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
            "or edits that together leave the file unparseable. Return edits that "
            "apply cleanly and preserve behavior.",
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
