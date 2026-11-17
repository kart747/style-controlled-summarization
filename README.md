# Style-Controlled Text Summarization

**Theme:** Summarization · Style Transfer · Transformer Models

One summarization model that can produce **four styles** for the same document. You choose the style by
changing a natural-language *control prefix* on the input:

| Style | Prefix fed to the model | What it looks like | Training targets |
|---|---|---|---|
| `academic` | `Write an academic abstract summarizing: …` | formal scientific abstract | PLOS papers → author abstract |
| `simple` | `Summarize in simple language for a general audience: …` | plain-language lay summary | PLOS papers → lay summary |
| `formal` | `Write formal news highlights summarizing: …` | concise neutral news highlights | CNN/DailyMail → highlights |
| `casual` | `Give a casual tl;dr of: …` | informal one-line TL;DR | Reddit TIFU → tl;dr |

The `academic` / `simple` pair is the most informative part of the data. Both targets come from the **same
paper body** (the abstract is removed from the input), so the only thing that changes between them is the
requested style. The model cannot pick the register from the topic; it has to read the prefix.

## Approach

```
                ┌──────────── long document (> 512 tokens) ────────────┐
document ──►    │ TextRank sentence ranking → keep most central         │ ──► "<style prefix>: <doc>"
                │ sentences that fit the budget, in original order      │            │
                └───────────────────────────────────────────────────────┘            ▼
                                                              FLAN-T5 (fine-tuned on all 4 styles)
                                                                                     │
                                                       style-dependent decoding (length, beams)
                                                                                     ▼
                                                                                  summary
```

* **Abstractive vs. extractive.** Lead-3 and TextRank are the extractive baselines. They copy source sentences,
  so they return the *same* output whatever style you ask for and cannot change register. The abstractive
  model rewrites the content. For long inputs we use a **hybrid**: TextRank first picks the most central
  sentences that fit T5's 512-token window, then the abstractive model writes the summary
  (`stylesum.extractive.compress_to_budget`).
* **Style tokens / control prefixes.** We follow T5's text-to-text task-prefix idea, as in CTRLsum and
  Control Prefixes, but the control attribute here is **register**, not keywords or length. All four styles
  are mixed in one fine-tuning run. Length is part of style, so decoding settings also depend on the style
  (`stylesum.model.GEN_CONFIG`).
* **Model.** `google/flan-t5-small` (77M parameters). It fits a 4 GB laptop GPU, and the code works
  unchanged with `flan-t5-base`/`t5-base`/`facebook/bart-base` (`--model`).

## Evaluation

* **Content:** ROUGE-1/2/Lsum F1 against the reference summary of the requested style.
* **Style accuracy:** a TF-IDF + logistic-regression classifier trained on the *reference* summaries
  labels each generated summary. We report how often it matches the requested style, the standard
  automatic metric in style-transfer work.
* **Style markers:** Flesch-Kincaid grade, reading ease, length, contraction rate and first-person rate.
* **Controllability test:** the *same* documents are summarized in all four styles. If style control works,
  each output should be classified as the requested style and the readability numbers should move in the
  expected direction (academic > formal > simple > casual in grade level).

See [`results/RESULTS.md`](results/RESULTS.md) for the full tables.

### Headline results (test set, 50 documents per style)

| System | ROUGE-1 | ROUGE-2 | ROUGE-Lsum | Style acc. |
|---|---:|---:|---:|---:|
| Reference (human) | 100.0 | 100.0 | 100.0 | 88% |
| Lead-3 (extractive) | 25.0 | 6.8 | 21.9 | 72% |
| TextRank (extractive) | 27.9 | 8.0 | 24.4 | 75% |
| FLAN-T5-small zero-shot + style prompt | 24.3 | 6.8 | 21.9 | 68% |
| **FLAN-T5-small fine-tuned + style prefix** | **34.2** | **10.2** | **31.2** | 72% |

Per style, fine-tuning raises ROUGE-1 from 25.4 to **41.0** (academic), 26.5 to **40.2** (simple), 31.6 to **34.4**
(formal) and 13.8 to **21.2** (casual) over the zero-shot model. In-style accuracy is 94% for formal and 98% for casual.

**What the control prefix changes.** Lengths follow the requested style on the same inputs: academic ≈ 106
words, simple ≈ 93, formal ≈ 45 and casual ≈ 32 in the controllability test. Formal outputs read like
headline-style news highlights and casual outputs like one-line takeaways (see
[`results/demo_example.json`](results/demo_example.json)).

**Limits of the style control.**
* The model mostly changes *length and compression*. It changes vocabulary much less: a 77M-parameter
  model trained for 2 epochs stays close to the source wording.
* On the controllability test, the classifier puts only 10–40% of outputs in the requested style. Part of
  this is the classifier itself: it also learns *topic*, so a "casual" summary of a biology paper is still
  labelled scientific.
* Academic vs. simple is the hardest pair. Even the human references are only separable about 80% of the
  time, because PLOS "lay" summaries are written by the authors and stay fairly technical.

Setup: zero-shot results use 2 beams and fine-tuned results use 4. Training ran for 2 epochs on 7,200
examples on an RTX 3050 (4 GB), took 37 min, and validation loss went 2.76 → 2.74.

## Project layout

```
src/stylesum/
  styles.py       style registry + control-prefix construction
  data.py         builds the multi-style corpus (PLOS, CNN/DM, Reddit TIFU) via the HF rows API
  extractive.py   Lead-k, TextRank, TextRank compression for long docs
  model.py        StyleSummarizer inference wrapper (style-aware decoding)
  train.py        fine-tuning with Seq2SeqTrainer
  metrics.py      ROUGE, readability/formality markers, style classifier
  evaluate.py     per-style quality + controllability experiments
  report.py       results/*.json → results/RESULTS.md
app.py            Gradio demo (single style or side-by-side comparison)
tests/            unit tests (pytest)
```

## Usage

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .

# 1. build the dataset (~1.8k train docs per style, 100 val, 200 test)
python -m stylesum.data --out data --train 1800 --val 100 --test 200

# 2. fine-tune (≈37 min on an RTX 3050 4 GB)
python -m stylesum.train --data data --out outputs/flan-t5-small-style --epochs 2 --batch 4 --grad-accum 4 --grad-checkpointing

# 3. evaluate all systems and write the report
python -m stylesum.evaluate --data data --model outputs/flan-t5-small-style --limit 50 --control-docs 30
python -m stylesum.report --results results

# 4. demo
python app.py --model outputs/flan-t5-small-style

# tests
pytest -q
```

Python API:

```python
from stylesum.model import StyleSummarizer
m = StyleSummarizer("outputs/flan-t5-small-style")
m.summarize(long_text, style="simple")
m.summarize_all_styles(long_text)   # {"academic": ..., "simple": ..., "formal": ..., "casual": ...}
```

## Related work and resources

* Raffel et al., 2020 - *T5: Exploring the Limits of Transfer Learning* (task prefixes); Chung et al., 2022 - *FLAN-T5*.
* He et al., 2020 - *CTRLsum: Towards Generic Controllable Text Summarization* ([salesforce/ctrl-sum](https://github.com/salesforce/ctrl-sum)), with keyword and length control.
* Clive et al., 2021 - *Control Prefixes for Parameter-Efficient Text Generation* ([arXiv:2110.08329](https://arxiv.org/abs/2110.08329)).
* Zhang et al., 2023 - *MACSum: Controllable Summarization with Mixed Attributes* (TACL).
* Goldsack et al., 2022 - *Making Science Simple: Corpora for the Lay Summarisation of Scientific Literature* (PLOS / eLife).
* Mihalcea & Tarau, 2004 - *TextRank*; Rao & Tetreault, 2018 - *GYAFC* (formality style transfer evaluation).
* Datasets: [PLOS lay summarisation](https://huggingface.co/datasets/pszemraj/scientific_lay_summarisation-plos-norm),
  [CNN/DailyMail](https://huggingface.co/datasets/abisee/cnn_dailymail),
  [Reddit TIFU](https://huggingface.co/datasets/dany0407/reddit_tifu_long).

None of the existing open-source controllable summarizers we found controls *register* (formal / casual /
academic / simple). They control length, keywords or entities. That gap is why this project builds its own
pseudo-parallel multi-style corpus.

## Limitations and future work

* The style classifier can partly use *topic* (science vs. news vs. Reddit) as a cue. The academic vs. simple
  comparison is the clean test because the source documents are identical.
* `flan-t5-small` is small; `flan-t5-base`, or LoRA on a larger model, should improve ROUGE and fluency.
* Further ideas: BERTScore / factual-consistency (e.g. SummaC) metrics, eLife lay summaries as extra data,
  and learned soft style prefixes (prefix-tuning) instead of text prefixes.
