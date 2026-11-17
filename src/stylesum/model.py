"""Inference wrapper around a (fine-tuned) seq2seq model with style control."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .extractive import compress_to_budget
from .styles import build_input, get_style

DEFAULT_MODEL = "google/flan-t5-small"
MAX_SOURCE_TOKENS = 512


@dataclass(frozen=True)
class GenConfig:
    min_new_tokens: int
    max_new_tokens: int
    length_penalty: float = 1.0


# Length is part of style: an abstract is long, a tl;dr is one line.
GEN_CONFIG: dict[str, GenConfig] = {
    "academic": GenConfig(80, 256, 1.2),
    "simple": GenConfig(80, 256, 1.2),
    "formal": GenConfig(25, 96, 1.0),
    "casual": GenConfig(8, 64, 0.8),
}


def pick_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


class StyleSummarizer:
    def __init__(self, model_name_or_path: str = DEFAULT_MODEL, device: str | None = None):
        self.device = device or pick_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path).to(self.device)
        self.model.eval()

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=True).input_ids)

    def prepare(self, text: str, style: str) -> str:
        """Build the prefixed input, compressing long documents with TextRank so
        the most central sentences survive T5's 512-token window."""
        prefix_len = self.count_tokens(build_input("", style))
        doc = compress_to_budget(text, self.count_tokens, MAX_SOURCE_TOKENS - prefix_len)
        return build_input(doc, style)

    @torch.inference_mode()
    def summarize(
        self,
        texts: str | Sequence[str],
        style: str,
        num_beams: int = 4,
        batch_size: int = 8,
        compress: bool = True,
    ) -> list[str]:
        if isinstance(texts, str):
            texts = [texts]
        get_style(style)  # validate
        cfg = GEN_CONFIG[style]
        inputs = [self.prepare(t, style) if compress else build_input(t, style) for t in texts]
        out: list[str] = []
        for i in range(0, len(inputs), batch_size):
            enc = self.tokenizer(
                inputs[i : i + batch_size],
                max_length=MAX_SOURCE_TOKENS,
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            ids = self.model.generate(
                **enc,
                num_beams=num_beams,
                min_new_tokens=cfg.min_new_tokens,
                max_new_tokens=cfg.max_new_tokens,
                length_penalty=cfg.length_penalty,
                no_repeat_ngram_size=3,
                early_stopping=True,
            )
            out.extend(s.strip() for s in self.tokenizer.batch_decode(ids, skip_special_tokens=True))
        return out

    def summarize_all_styles(self, text: str, **kw) -> dict[str, str]:
        from .styles import STYLE_NAMES

        return {s: self.summarize(text, s, **kw)[0] for s in STYLE_NAMES}
