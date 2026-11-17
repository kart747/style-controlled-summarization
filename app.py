"""Gradio demo: paste a document, pick a style (or compare all four).

    python app.py --model outputs/flan-t5-small-style
"""

from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path

import gradio as gr

from stylesum.extractive import lead_k, textrank
from stylesum.metrics import style_markers
from stylesum.model import DEFAULT_MODEL, StyleSummarizer
from stylesum.styles import STYLE_NAMES, STYLES

EXAMPLE = (
    "Researchers at the University of Example have developed a new battery chemistry "
    "that stores nearly twice as much energy as conventional lithium-ion cells. The "
    "team replaced the graphite anode with a silicon-carbon composite and added an "
    "electrolyte additive that prevents the silicon from cracking as it swells during "
    "charging. In laboratory tests, prototype cells kept 90 percent of their capacity "
    "after 1,000 charge cycles. The authors say the materials are inexpensive and "
    "compatible with existing factories, but caution that the cells have so far only "
    "been tested at small scale and at room temperature. Electric-vehicle makers have "
    "expressed interest, and a pilot production line is planned for next year."
)


@lru_cache(maxsize=2)
def load(path: str) -> StyleSummarizer:
    return StyleSummarizer(path)


def fmt_markers(text: str) -> str:
    m = style_markers(text)
    return (f"{m['words']} words · grade level {m['fkgl']:.1f} · "
            f"reading ease {m['flesch']:.0f} · contractions {m['contractions_pct']:.1f}%")


def build_ui(model_path: str) -> gr.Blocks:
    def run_one(text, style, beams):
        if not text.strip():
            return "", ""
        out = load(model_path).summarize(text, style, num_beams=int(beams))[0]
        return out, fmt_markers(out)

    def run_all(text, beams):
        if not text.strip():
            return [""] * (len(STYLE_NAMES) + 2)
        m = load(model_path)
        outs = [m.summarize(text, s, num_beams=int(beams))[0] for s in STYLE_NAMES]
        return outs + [lead_k(text), textrank(text)]

    with gr.Blocks(title="Style-Controlled Summarizer") as demo:
        gr.Markdown(
            "# Style-Controlled Summarizer\n"
            f"FLAN-T5 fine-tuned with style control prefixes. Model: `{model_path}`"
        )
        text = gr.Textbox(label="Document", lines=12, value=EXAMPLE)
        beams = gr.Slider(1, 8, value=4, step=1, label="Beam size")
        with gr.Tab("Single style"):
            style = gr.Radio(STYLE_NAMES, value="simple", label="Style",
                             info=" | ".join(f"{k}: {v.description}" for k, v in STYLES.items()))
            btn = gr.Button("Summarize", variant="primary")
            out = gr.Textbox(label="Summary", lines=6)
            stats = gr.Markdown()
            btn.click(run_one, [text, style, beams], [out, stats])
        with gr.Tab("Compare all styles"):
            btn2 = gr.Button("Summarize in every style", variant="primary")
            boxes = [gr.Textbox(label=f"{s} (abstractive)", lines=4) for s in STYLE_NAMES]
            boxes += [gr.Textbox(label="Lead-3 (extractive baseline)", lines=3),
                      gr.Textbox(label="TextRank (extractive baseline)", lines=3)]
            btn2.click(run_all, [text, beams], boxes)
    return demo


def main() -> None:
    ap = argparse.ArgumentParser()
    default = "outputs/flan-t5-small-style"
    ap.add_argument("--model", default=default if Path(default).exists() else DEFAULT_MODEL)
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args()
    build_ui(a.model).launch(share=a.share, server_port=a.port)


if __name__ == "__main__":
    main()
