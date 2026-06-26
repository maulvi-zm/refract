from __future__ import annotations

from refract.planning.context import RefactorContext


def build_system_prompt(language: str) -> str:
    return (
        f"You are Refract, a {language} refactoring assistant.\n"
        "Return only JSON with keys: explanation, old_snippet, new_snippet, confidence.\n"
        "The old_snippet must be copied verbatim from the supplied source context, "
        "character for character, including whitespace and indentation.\n"
        "The new_snippet must contain the complete, working implementation. Never use "
        "placeholder comments or stub bodies. Copy every line of logic from the original "
        "and restructure it.\n"
        "The new_snippet must preserve all existing behavior while addressing the "
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


def _method_list(methods, with_file: bool) -> str:
    lines = [
        f"- {m.class_name}.{m.name}"
        + (f" in {m.file}" if with_file else "")
        + f" lines {m.start_line}-{m.end_line}"
        for m in methods
    ]
    return "\n".join(lines) or "- none"
