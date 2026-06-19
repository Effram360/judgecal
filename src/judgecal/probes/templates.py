"""Judge prompt templates for pairwise comparison.

One default template plus four hand-written, semantically equivalent
paraphrases (ids ``tpl:default``, ``tpl:v1`` .. ``tpl:v4``). All five:

* pose the identical task — pick the better of two AI assistant
  responses to the same user question;
* name the presented-first response "A" and the presented-second "B";
* instruct the judge to weigh helpfulness, accuracy, relevance, depth,
  and clarity, and to ignore presentation order and response length
  (neutral wording, no position hints);
* pin the MT-Bench-style output convention: the reply must end with
  exactly one of ``[[A]]`` (first is better), ``[[B]]`` (second is
  better), or ``[[C]]`` (tie), which the verdict parser maps to
  presented-coordinates verdicts.

``render`` produces an OpenAI chat-completions ``messages`` list
(system + user); model name and sampling parameters are attached later
at manifest-emission time.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class JudgeTemplate:
    """A pairwise judge prompt template.

    Attributes:
        template_id: Stable identifier (``tpl:default``, ``tpl:v1``, ...).
        system: System message content (fixed text).
        user: User message format string with ``{prompt}``,
            ``{first_text}``, and ``{second_text}`` placeholders.
    """

    template_id: str
    system: str
    user: str


_DEFAULT = JudgeTemplate(
    template_id="tpl:default",
    system=(
        "You are an impartial judge tasked with evaluating the quality of "
        "responses provided by two AI assistants."
    ),
    user=(
        "Please act as an impartial judge and evaluate the quality of the responses "
        "provided by two AI assistants to the user question displayed below. Choose "
        "the assistant that answers the question better. Your evaluation should "
        "consider helpfulness, relevance, accuracy, depth, and clarity. Do not allow "
        "the order in which the responses are presented, the length of the responses, "
        "or the names of the assistants to influence your decision. Begin your "
        "evaluation with a brief explanation. After providing your explanation, "
        "output your final verdict on a new line, strictly in this format: "
        '"[[A]]" if Assistant A\'s response is better, "[[B]]" if Assistant B\'s '
        'response is better, or "[[C]]" for a tie.\n\n'
        "[User Question]\n{prompt}\n\n"
        "[Assistant A's Response]\n{first_text}\n\n"
        "[Assistant B's Response]\n{second_text}"
    ),
)

_V1 = JudgeTemplate(
    template_id="tpl:v1",
    system="You are a careful and unbiased evaluator of AI assistant responses.",
    user=(
        "Below is a question from a user, followed by two candidate responses "
        "written by different AI assistants (Assistant A and Assistant B).\n\n"
        "Question:\n{prompt}\n\n"
        "Response from Assistant A:\n{first_text}\n\n"
        "Response from Assistant B:\n{second_text}\n\n"
        "Decide which response answers the question better, judging helpfulness, "
        "factual accuracy, relevance, depth of detail, and clarity. The position in "
        "which a response appears and how long it is must not affect your judgment. "
        "First write a short justification. Then, on a final line, give your verdict "
        "using exactly one of these markers: [[A]] (Assistant A is better), "
        "[[B]] (Assistant B is better), or [[C]] (the responses are equally good)."
    ),
)

_V2 = JudgeTemplate(
    template_id="tpl:v2",
    system="You serve as a neutral referee comparing two AI-generated answers.",
    user=(
        "You will compare two answers to the same user question and determine which "
        "one is of higher quality.\n\n"
        "=== Question ===\n{prompt}\n\n"
        "=== Answer A ===\n{first_text}\n\n"
        "=== Answer B ===\n{second_text}\n\n"
        "Assess each answer for how helpful, accurate, relevant, thorough, and clear "
        "it is. Base your decision only on the substance of the answers: neither the "
        "order in which they appear nor their length should sway you. Explain your "
        "reasoning briefly, then conclude on its own line with exactly [[A]] if "
        "Answer A is better, [[B]] if Answer B is better, or [[C]] if they are tied."
    ),
)

_V3 = JudgeTemplate(
    template_id="tpl:v3",
    system="You are an expert reviewer asked to grade two AI assistant replies against each other.",
    user=(
        "Your task: read the user's question and the two replies below, then judge "
        "which reply is the stronger answer.\n\n"
        "Question posed by the user:\n{prompt}\n\n"
        "Reply A:\n{first_text}\n\n"
        "Reply B:\n{second_text}\n\n"
        "Weigh helpfulness, correctness, relevance to the question, depth, and "
        "clarity of writing. Ignore presentation order and reply length; these must "
        "not influence your verdict. Give a short explanation of your comparison, "
        "and finish with a single line containing only your verdict: [[A]] means "
        "Reply A is better, [[B]] means Reply B is better, and [[C]] means it is a tie."
    ),
)

_V4 = JudgeTemplate(
    template_id="tpl:v4",
    system="You judge pairs of AI assistant responses and pick the better one.",
    user=(
        "Compare the two responses to the question below and decide which is better "
        "overall.\n\n"
        "Question:\n{prompt}\n\n"
        "First response (Assistant A):\n{first_text}\n\n"
        "Second response (Assistant B):\n{second_text}\n\n"
        "Judge on helpfulness, accuracy, relevance, depth, and clarity alone - not "
        "on response order or response length. Briefly explain your decision, then "
        "end your reply with exactly one verdict marker on its own line: [[A]] for "
        "Assistant A, [[B]] for Assistant B, or [[C]] for a tie."
    ),
)

#: All shipped templates, keyed by id.
TEMPLATES: dict[str, JudgeTemplate] = {t.template_id: t for t in (_DEFAULT, _V1, _V2, _V3, _V4)}

#: Template ids in canonical order (default first, then paraphrases).
TEMPLATE_IDS: tuple[str, ...] = ("tpl:default", "tpl:v1", "tpl:v2", "tpl:v3", "tpl:v4")

#: The template used by every probe except the template-sensitivity probe.
DEFAULT_TEMPLATE_ID: str = "tpl:default"


def get_template(template_id: str) -> JudgeTemplate:
    """Look up a shipped template by id.

    Args:
        template_id: One of ``TEMPLATE_IDS``.

    Returns:
        The matching :class:`JudgeTemplate`.

    Raises:
        KeyError: If ``template_id`` is unknown.
    """
    try:
        return TEMPLATES[template_id]
    except KeyError:
        raise KeyError(
            f"unknown template id {template_id!r}; available: {list(TEMPLATE_IDS)}"
        ) from None


def template_ids_for(n_variants: int) -> tuple[str, ...]:
    """First ``n_variants`` template ids (default first).

    Args:
        n_variants: Number of template variants requested (1..5).

    Returns:
        Tuple of template ids of length ``n_variants``.

    Raises:
        ValueError: If ``n_variants`` is outside ``[1, len(TEMPLATE_IDS)]``.
    """
    if not 1 <= n_variants <= len(TEMPLATE_IDS):
        raise ValueError(
            f"n_template_variants must be in [1, {len(TEMPLATE_IDS)}], got {n_variants}"
        )
    return TEMPLATE_IDS[:n_variants]


def render(
    template_id: str, prompt: str, first_text: str, second_text: str
) -> list[dict[str, str]]:
    """Render a template into an OpenAI chat-completions messages list.

    Args:
        template_id: Id of the template to render.
        prompt: The user question both responses answer.
        first_text: The presented-first response (judge sees it as "A").
        second_text: The presented-second response (judge sees it as "B").

    Returns:
        ``[{"role": "system", ...}, {"role": "user", ...}]``. Substituted
        values are inserted verbatim (braces in responses are safe — only
        the fixed template string is interpreted as a format string).

    Raises:
        KeyError: If ``template_id`` is unknown.
    """
    tpl = get_template(template_id)
    user = tpl.user.format(prompt=prompt, first_text=first_text, second_text=second_text)
    return [
        {"role": "system", "content": tpl.system},
        {"role": "user", "content": user},
    ]


__all__ = [
    "DEFAULT_TEMPLATE_ID",
    "TEMPLATES",
    "TEMPLATE_IDS",
    "JudgeTemplate",
    "get_template",
    "render",
    "template_ids_for",
]
