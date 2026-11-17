import json

from stylesum.report import render


def _metrics(r1):
    return {"n": 2, "rouge1": r1, "rouge2": 1.0, "rougeLsum": 2.0, "style_acc": 50.0,
            "words": 10.0, "fkgl": 8.0, "flesch": 60.0, "avg_word_len": 4.0,
            "contractions_pct": 0.0, "first_person_pct": 0.0}


def test_render_tables(tmp_path):
    rep = {"system": "finetuned",
           "per_style": {"academic": _metrics(40.0), "casual": _metrics(20.0)},
           "controllability": {s: {**_metrics(0), "predicted": {}}
                               for s in ["academic", "simple", "formal", "casual"]}}
    (tmp_path / "finetuned.json").write_text(json.dumps(rep))
    md = render(tmp_path)
    assert "| FLAN-T5-small fine-tuned + style prefix (4 epochs) | 30.0 |" in md  # averaged R-1
    assert "| FLAN-T5-small fine-tuned + style prefix (4 epochs) | academic | 40.0 |" in md
    assert "| casual | 50% |" in md
