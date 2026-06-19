"""Deterministic, meaning-preserving padding for the verbosity probe.

``pad_text`` lengthens a response to roughly ``target_ratio`` times its
character length without adding new claims. It works in three rule-based
layers, appended after the original text:

1. **Sentence restatements** — each original sentence restated behind a
   fixed connective ("In other words, ...", "To restate the point, ...").
2. **Enumerated recap** — "To recap the main points: (1) ... (2) ..."
   re-listing the original sentences.
3. **Word-level top-up** — a short "Recall:" clause cycling words from
   the original text; the final filler word is character-truncated to
   the remaining gap, so the realized length lands within a couple of
   characters of the target even for terminator-free single-token texts
   (code blobs, base64, long URLs).

Layers 1 and 2 are added greedily, skipping any unit that would
overshoot the target; layer 3 closes the remaining gap exactly.

Limitation (by design, documented per the implementation contract):
this is *rule-based* padding, not an LLM rewrite. The output is
grammatical-ish plain text that preserves meaning by construction
(restatement + recap of the very same sentences), but it is stylistically
repetitive. A length-biased judge should still prefer it; a quality-
sensitive judge should treat it as equivalent. Interpret the pad probe
accordingly.

Determinism: ``pad_text`` is a pure function of ``(text, target_ratio)``
— no RNG, no state.
"""

from __future__ import annotations

import re

#: Fixed connective templates cycled over sentences for restatement.
_CONNECTIVES: tuple[str, ...] = (
    "In other words, {s}",
    "To restate the point, {s}",
    "Put differently, {s}",
    "Said another way, {s}",
)

_RECAP_INTRO = "To recap the main points:"
_TOPUP_INTRO = "Recall:"

_SENTENCE_RE = re.compile(r"[^.!?]+[.!?]*")


def _split_sentences(text: str) -> list[str]:
    """Split text into sentence-ish chunks (terminator kept)."""
    return [s.strip() for s in _SENTENCE_RE.findall(text) if s.strip()]


def _decapitalize(sentence: str) -> str:
    """Lowercase the leading character when it looks safe to do so."""
    if len(sentence) >= 2 and sentence[0].isupper() and sentence[1].islower():
        return sentence[0].lower() + sentence[1:]
    return sentence


def _restatement_units(sentences: list[str]) -> list[str]:
    units = []
    for i, sentence in enumerate(sentences):
        body = _decapitalize(sentence.rstrip(".!?").strip())
        template = _CONNECTIVES[i % len(_CONNECTIVES)]
        units.append(template.format(s=body + "."))
    return units


def _recap_units(sentences: list[str]) -> list[str]:
    units = []
    for i, sentence in enumerate(sentences):
        body = sentence.rstrip(".!?").strip()
        unit = f"({i + 1}) {body}."
        if i == 0:
            unit = f"{_RECAP_INTRO} {unit}"
        units.append(unit)
    return units


def pad_text(text: str, target_ratio: float = 1.6) -> str:
    """Pad ``text`` to about ``target_ratio`` times its character length.

    The original text is preserved verbatim as a prefix; padding is
    appended (restatements, then an enumerated recap, then a word-level
    top-up whose final filler word is truncated to the remaining gap —
    see module docstring). For typical multi-sentence responses the
    realized ratio lands well within ±15% of the target, and the top-up
    truncation keeps even terminator-free single-token texts within a
    few characters of the target; for very short texts (a few dozen
    characters) the discrete unit sizes dominate and the ratio is
    best-effort.

    Args:
        text: The response text to pad. Returned unchanged when empty.
        target_ratio: Desired ``len(padded) / len(text)``; values <= 1.0
            return ``text`` unchanged.

    Returns:
        The padded text (always ``startswith(text)``), deterministic in
        its inputs.
    """
    if not text or target_ratio <= 1.0:
        return text
    target = int(round(len(text) * target_ratio))
    if target <= len(text):
        return text

    sentences = _split_sentences(text) or [text.strip()]
    units = _restatement_units(sentences) + _recap_units(sentences)

    out = text
    for unit in units:
        if len(out) + 1 + len(unit) <= target:
            out = f"{out} {unit}"

    if len(out) < target:
        words = [w.strip(".,;:!?\"'()[]") or "point" for w in text.split()] or ["point"]
        out = f"{out} {_TOPUP_INTRO}"
        i = 0
        while len(out) + 1 < target:  # +1 leaves room for the closing period
            word = words[i % len(words)]
            # Room left for " <word>" while keeping the closing period.
            room = target - len(out) - 2
            if room <= 0:
                break
            if len(word) > room:
                word = word[:room]  # truncate the final filler to the gap
            out = f"{out} {word}"
            i += 1
        out = f"{out}."
    return out


__all__ = ["pad_text"]
