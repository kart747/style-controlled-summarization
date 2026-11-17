"""Build the multi-style summarization corpus.

Each public dataset is *streamed* (no full download) and a few thousand
examples per style are written to ``data/{train,val,test}.jsonl`` with fields
``id, style, document, summary``.

* PLOS (Goldsack et al. 2022, "Making Science Simple") gives two targets for the
  same paper: the author abstract (academic) and the lay summary (simple).
  The abstract is removed from the input so the model cannot copy it.
* CNN/DailyMail highlights -> formal news style.
* Reddit TIFU tl;dr -> casual style.

Usage::

    python -m stylesum.data --out data --train 2500 --val 100 --test 200
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections.abc import Iterable, Iterator
from pathlib import Path

PLOS = "pszemraj/scientific_lay_summarisation-plos-norm"
CNN = ("abisee/cnn_dailymail", "3.0.0")
TIFU = "dany0407/reddit_tifu_long"

MAX_DOC_WORDS = 600  # inputs are truncated to 512 tokens anyway


# --------------------------------------------------------------------------- #
# Cleaning helpers (pure functions, unit-tested)
# --------------------------------------------------------------------------- #
_WS = re.compile(r"\s+")


def normalize_ws(text: str) -> str:
    return _WS.sub(" ", text or "").strip()


def truncate_words(text: str, n: int = MAX_DOC_WORDS) -> str:
    words = text.split()
    return " ".join(words[:n])


def word_count(text: str) -> int:
    return len(text.split())


def split_plos_article(article: str) -> tuple[str, str]:
    """Return ``(abstract, body)``; sections in the PLOS data are newline-separated
    and the first section is always the abstract."""
    parts = [p for p in (article or "").split("\n") if p.strip()]
    if len(parts) < 2:
        return "", ""
    return normalize_ws(parts[0]), normalize_ws(" ".join(parts[1:]))


_CNN_LEAD = re.compile(r"^.{0,80}?\(CNN\)\s*(--)?\s*")
_DM_BOILER = re.compile(
    r"^(By \. .*? \. )?(PUBLISHED: \. .*? \. )?(\| \. )?(UPDATED: \. .*? \. )?",
)
_DM_LAST_UPDATED = re.compile(r"Last updated at .*? \d{4} \. ")


def clean_cnn_article(text: str) -> str:
    text = normalize_ws(text)
    text = _CNN_LEAD.sub("", text, count=1)
    text = _DM_BOILER.sub("", text, count=1)
    text = _DM_LAST_UPDATED.sub("", text, count=1)
    return text.strip()


def clean_highlights(text: str) -> str:
    """CNN/DM highlights are newline-separated bullet sentences."""
    lines = [normalize_ws(x).rstrip(" .") for x in (text or "").split("\n")]
    return " ".join(f"{x}." for x in lines if x)


_MD_JUNK = re.compile(r"[*_#>]+")


def clean_tldr(text: str) -> str:
    text = _MD_JUNK.sub("", text or "")
    text = re.sub(r"^\s*(tl\s*;?\s*dr\s*[:\-]?)\s*", "", text, flags=re.I)
    return normalize_ws(text)


def clean_reddit_post(text: str) -> str:
    text = _MD_JUNK.sub("", text or "")
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\bedit\s*:.*$", "", text, flags=re.I)
    return normalize_ws(text)


# --------------------------------------------------------------------------- #
# Per-source example generators
# --------------------------------------------------------------------------- #
ROWS_API = "https://datasets-server.huggingface.co/rows"


def _get_json(url: str, retries: int = 8, timeout: float = 120.0) -> dict:
    import time
    import urllib.request

    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "stylesum-data/0.1"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.load(resp)
        except Exception as e:  # network is flaky: back off and retry
            if attempt == retries - 1:
                raise
            wait = min(60, 2 ** attempt)
            print(f"  retry {attempt + 1}/{retries} in {wait}s ({type(e).__name__}: {e})")
            time.sleep(wait)
    raise RuntimeError("unreachable")


def _stream(name: str, config: str | None = None, split: str = "train",
            page: int = 100) -> Iterator[dict]:
    """Lazily page through a Hub dataset via the dataset-viewer *rows* API.

    This avoids downloading whole parquet files: the PLOS train parquet is a
    single 500 MB row group, which plain streaming must fetch in full before
    yielding a single row."""
    from urllib.parse import urlencode

    offset, total = 0, None
    while total is None or offset < total:
        q = urlencode({"dataset": name, "config": config or "default", "split": split,
                       "offset": offset, "length": page})
        d = _get_json(f"{ROWS_API}?{q}")
        total = d["num_rows_total"]
        rows = d["rows"]
        if not rows:
            return
        for r in rows:
            yield r["row"]
        offset += len(rows)


def _chain(*iters: Iterable[dict]) -> Iterator[dict]:
    for it in iters:
        yield from it


def plos_examples(rows: Iterable[dict]) -> Iterator[list[dict]]:
    """Yield *pairs* (academic, simple) per paper so both styles share splits."""
    for i, r in enumerate(rows):
        abstract, body = split_plos_article(r["article"])
        lay = normalize_ws(r["summary"])
        if not (80 <= word_count(abstract) <= 400 and 80 <= word_count(lay) <= 400):
            continue
        if word_count(body) < 300:
            continue
        doc = truncate_words(body)
        yield [
            {"id": f"plos-{i}-a", "style": "academic", "document": doc, "summary": abstract},
            {"id": f"plos-{i}-s", "style": "simple", "document": doc, "summary": lay},
        ]


def cnn_examples(rows: Iterable[dict]) -> Iterator[list[dict]]:
    for r in rows:
        doc = clean_cnn_article(r["article"])
        summ = clean_highlights(r["highlights"])
        if word_count(doc) < 150 or not (15 <= word_count(summ) <= 90):
            continue
        yield [{"id": f"cnn-{r['id']}", "style": "formal",
                "document": truncate_words(doc), "summary": summ}]


def tifu_examples(rows: Iterable[dict]) -> Iterator[list[dict]]:
    for i, r in enumerate(rows):
        doc = clean_reddit_post(r["documents"])
        summ = clean_tldr(r["tldr"])
        if word_count(doc) < 100 or not (6 <= word_count(summ) <= 60):
            continue
        yield [{"id": f"tifu-{i}", "style": "casual",
                "document": truncate_words(doc), "summary": summ}]


def take_groups(groups: Iterator[list[dict]], n: int) -> list[list[dict]]:
    out = []
    for g in groups:
        out.append(g)
        if len(out) >= n:
            break
    return out


def split_groups(groups: list[list[dict]], n_train: int, n_val: int, n_test: int,
                 seed: int = 13) -> dict[str, list[dict]]:
    """Split by *group* (paper/article) so paired styles never leak across splits."""
    groups = list(groups)
    random.Random(seed).shuffle(groups)
    cuts = {"test": groups[:n_test],
            "val": groups[n_test:n_test + n_val],
            "train": groups[n_test + n_val:n_test + n_val + n_train]}
    return {k: [ex for g in v for ex in g] for k, v in cuts.items()}


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def build(out: Path, n_train: int, n_val: int, n_test: int, seed: int = 13) -> None:
    need = n_train + n_val + n_test
    sources = {
        # PLOS train is one 500 MB row group the rows API refuses to scan; the
        # validation + test splits give 2,752 distinct papers, which is plenty.
        "plos": lambda: plos_examples(_chain(_stream(PLOS, split="validation", page=50),
                                             _stream(PLOS, split="test", page=50))),
        "cnn": lambda: cnn_examples(_stream(*CNN)),
        "tifu": lambda: tifu_examples(_stream(TIFU)),
    }
    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for name, make in sources.items():
        # oversample then shuffle so we don't just take the head of the stream
        groups = take_groups(make(), int(need * 1.2))
        if len(groups) < n_val + n_test + 100:
            raise RuntimeError(f"{name}: only {len(groups)} usable examples, need {need}")
        print(f"{name}: collected {len(groups)} groups", flush=True)
        # If a source is short (PLOS), train gets whatever remains after val/test.
        for k, v in split_groups(groups, n_train, n_val, n_test, seed).items():
            splits[k].extend(v)
    for k, rows in splits.items():
        random.Random(seed).shuffle(rows)
        write_jsonl(out / f"{k}.jsonl", rows)
        print(f"wrote {len(rows):>6} rows -> {out / f'{k}.jsonl'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--train", type=int, default=2500, help="groups per source")
    ap.add_argument("--val", type=int, default=100)
    ap.add_argument("--test", type=int, default=200)
    ap.add_argument("--seed", type=int, default=13)
    a = ap.parse_args()
    build(a.out, a.train, a.val, a.test, a.seed)


if __name__ == "__main__":
    main()
