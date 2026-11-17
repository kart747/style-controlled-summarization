"""Style registry and control-prefix construction.

Style control is implemented by prepending a natural-language control prefix
to the source document (the T5 "task prefix" idea). A single seq2seq model is
fine-tuned on a mixture of datasets whose reference summaries differ in
register, so the prefix becomes the only signal telling the model which
register to produce.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Style:
    name: str
    description: str
    instruction: str  # natural-language control prefix fed to the model
    source: str  # dataset the training targets come from


STYLES: dict[str, Style] = {
    "academic": Style(
        name="academic",
        description="Formal scientific abstract with technical terminology.",
        instruction="Write an academic abstract summarizing",
        source="PLOS (paper body -> author abstract)",
    ),
    "simple": Style(
        name="simple",
        description="Plain-language summary for non-experts.",
        instruction="Summarize in simple language for a general audience",
        source="PLOS (paper body -> lay summary)",
    ),
    "formal": Style(
        name="formal",
        description="Concise, neutral, news-style highlights.",
        instruction="Write formal news highlights summarizing",
        source="CNN/DailyMail (article -> highlights)",
    ),
    "casual": Style(
        name="casual",
        description="Informal, conversational one-or-two-line TL;DR.",
        instruction="Give a casual tl;dr of",
        source="Reddit TIFU (post -> tl;dr)",
    ),
}

STYLE_NAMES: list[str] = list(STYLES)


def get_style(name: str) -> Style:
    key = name.strip().lower()
    if key not in STYLES:
        raise ValueError(f"Unknown style {name!r}; choose one of {STYLE_NAMES}")
    return STYLES[key]


def build_input(text: str, style: str) -> str:
    """Return the model input: ``"<instruction>: <document>"``."""
    return f"{get_style(style).instruction}: {text.strip()}"
