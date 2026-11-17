"""Evaluate summarizers on the multi-style test set.

Two experiments:

1. **Per-style quality** - each test example is summarized in its *own* style;
   we report ROUGE against the reference plus style metrics.
2. **Controllability** - the same documents are summarized in *every* style;
   we check that the style classifier recognises the requested style and that
   readability moves in the expected direction.

Usage::

    python -m stylesum.evaluate --data data --systems lead3 textrank zeroshot finetuned \
        --model outputs/flan-t5-small-style --limit 100
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from .data import read_jsonl
from .extractive import EXTRACTIVE
from .metrics import StyleClassifier, mean_markers, rouge
from .model import DEFAULT_MODEL, StyleSummarizer
from .styles import STYLE_NAMES


def by_style(rows: list[dict], limit: int | None) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if limit is None or len(out[r["style"]]) < limit:
            out[r["style"]].append(r)
    return out


def get_classifier(data: Path, path: Path) -> StyleClassifier:
    if path.exists():
        return StyleClassifier.load(path)
    train = read_jsonl(data / "train.jsonl")
    clf = StyleClassifier().fit([r["summary"] for r in train], [r["style"] for r in train])
    clf.save(path)
    return clf


def make_system(name: str, model_path: str, device: str | None = None, beams: int = 4):
    """Return ``fn(list_of_docs, style) -> list_of_summaries``."""
    if name in EXTRACTIVE:
        f = EXTRACTIVE[name]
        return lambda docs, style: [f(d) for d in docs]
    if name == "reference":
        return None
    path = DEFAULT_MODEL if name == "zeroshot" else model_path
    m = StyleSummarizer(path, device=device)
    return lambda docs, style: m.summarize(docs, style, num_beams=beams)


def evaluate_system(name, fn, groups, clf) -> tuple[dict, list[dict]]:
    res, samples = {}, []
    for style in STYLE_NAMES:
        rows = groups.get(style, [])
        if not rows:
            continue
        refs = [r["summary"] for r in rows]
        preds = refs if fn is None else fn([r["document"] for r in rows], style)
        res[style] = {
            "n": len(rows),
            **rouge(preds, refs),
            "style_acc": clf.accuracy(preds, [style] * len(preds)),
            **mean_markers(preds),
        }
        samples += [{"id": r["id"], "style": style, "pred": p, "ref": r["summary"]}
                    for r, p in list(zip(rows, preds))[:3]]
        print(f"[{name}] {style:9s} {res[style]}")
    return res, samples


def controllability(fn, docs: list[str], clf) -> dict:
    """Generate every style for the same docs (style must come from the prefix)."""
    out = {}
    for style in STYLE_NAMES:
        preds = fn(docs, style)
        out[style] = {"style_acc": clf.accuracy(preds, [style] * len(preds)),
                      "predicted": dict(sorted(_count(clf.predict(preds)).items())),
                      **mean_markers(preds)}
        print(f"[control] {style:9s} {out[style]}")
    return out


def _count(xs):
    d = defaultdict(int)
    for x in xs:
        d[x] += 1
    return dict(d)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate style-controlled summarizers")
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--model", default="outputs/flan-t5-small-style")
    ap.add_argument("--systems", nargs="+",
                    default=["reference", "lead3", "textrank", "zeroshot", "finetuned"])
    ap.add_argument("--limit", type=int, default=100, help="test examples per style")
    ap.add_argument("--control-docs", type=int, default=50)
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--clf", type=Path, default=Path("outputs/style_clf.pkl"))
    ap.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    ap.add_argument("--beams", type=int, default=4)
    a = ap.parse_args()

    a.results.mkdir(parents=True, exist_ok=True)
    clf = get_classifier(a.data, a.clf)
    test = read_jsonl(a.data / "test.jsonl")
    groups = by_style(test, a.limit)
    # Controllability docs: a mix of sources, one doc per PLOS paper.
    seen, ctrl_docs = set(), []
    for r in test:
        if r["document"] not in seen:
            seen.add(r["document"])
            ctrl_docs.append(r["document"])
    ctrl_docs = ctrl_docs[: a.control_docs]

    for name in a.systems:
        fn = make_system(name, a.model, a.device, a.beams)
        res, samples = evaluate_system(name, fn, groups, clf)
        report = {"system": name, "per_style": res}
        if name in ("zeroshot", "finetuned"):
            report["controllability"] = controllability(fn, ctrl_docs, clf)
        (a.results / f"{name}.json").write_text(json.dumps(report, indent=2))
        (a.results / f"{name}_samples.json").write_text(json.dumps(samples, indent=2))
    print("results ->", a.results)


if __name__ == "__main__":
    main()
