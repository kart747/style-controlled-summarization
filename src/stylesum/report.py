"""Render ``results/*.json`` (from ``stylesum.evaluate``) into ``results/RESULTS.md``.

    python -m stylesum.report --results results
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .styles import STYLE_NAMES

ORDER = ["reference", "lead3", "textrank", "zeroshot", "finetuned"]
LABEL = {
    "reference": "Reference (human)",
    "lead3": "Lead-3 (extractive)",
    "textrank": "TextRank (extractive)",
    "zeroshot": "FLAN-T5-small zero-shot + prompt",
    "finetuned": "FLAN-T5-small fine-tuned + style prefix",
}


def _pct(m: dict, key: str) -> str:
    return f"{m[key]:.0f}%" if key in m else "-"


def _load(results: Path) -> list[dict]:
    reports = []
    for name in ORDER:
        p = results / f"{name}.json"
        if p.exists():
            reports.append(json.loads(p.read_text()))
    return reports


def quality_table(reports: list[dict]) -> str:
    head = "| System | Style | R-1 | R-2 | R-Lsum | Style acc. (lexical) | Style acc. (markers) | Words | FK grade |\n"
    head += "|---|---|---:|---:|---:|---:|---:|---:|---:|\n"
    rows = []
    for rep in reports:
        for style in STYLE_NAMES:
            m = rep["per_style"].get(style)
            if not m:
                continue
            rows.append(
                f"| {LABEL.get(rep['system'], rep['system'])} | {style} | {m['rouge1']:.1f} | "
                f"{m['rouge2']:.1f} | {m['rougeLsum']:.1f} | {m['style_acc']:.0f}% | {_pct(m, 'marker_acc')} | "
                f"{m['words']:.0f} | {m['fkgl']:.1f} |"
            )
    return head + "\n".join(rows)


def average_table(reports: list[dict]) -> str:
    head = ("| System | R-1 | R-2 | R-Lsum | Style acc. (lexical) | Style acc. (markers) |\n"
            "|---|---:|---:|---:|---:|---:|\n")
    rows = []
    for rep in reports:
        ms = list(rep["per_style"].values())
        if not ms:
            continue
        keys = [k for k in ("rouge1", "rouge2", "rougeLsum", "style_acc", "marker_acc") if k in ms[0]]
        avg = {k: sum(m[k] for m in ms) / len(ms) for k in keys}
        rows.append(f"| {LABEL.get(rep['system'], rep['system'])} | {avg['rouge1']:.1f} | "
                    f"{avg['rouge2']:.1f} | {avg['rougeLsum']:.1f} | {avg['style_acc']:.0f}% | "
                    f"{_pct(avg, 'marker_acc')} |")
    return head + "\n".join(rows)


def control_table(reports: list[dict]) -> str:
    out = []
    for rep in reports:
        c = rep.get("controllability")
        if not c:
            continue
        out.append(f"**{LABEL.get(rep['system'], rep['system'])}** - same documents, every style requested\n")
        out.append("| Requested style | Style acc. (lexical) | Style acc. (markers) | Words | FK grade | Contractions % | 1st-person % |")
        out.append("|---|---:|---:|---:|---:|---:|---:|")
        for style in STYLE_NAMES:
            m = c[style]
            out.append(f"| {style} | {m['style_acc']:.0f}% | {_pct(m, 'marker_acc')} | {m['words']:.0f} | {m['fkgl']:.1f} | "
                       f"{m['contractions_pct']:.2f} | {m['first_person_pct']:.2f} |")
        out.append("")
    return "\n".join(out)


def render(results: Path) -> str:
    reports = _load(results)
    return "\n\n".join([
        "# Results",
        "## Average over styles",
        average_table(reports),
        "## Per-style quality (each test example in its own style)",
        quality_table(reports),
        "## Style controllability",
        control_table(reports),
    ]) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("results"))
    a = ap.parse_args()
    md = render(a.results)
    (a.results / "RESULTS.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
