from stylesum.extractive import lead_k, split_sentences, textrank

DOC = (
    "Dr. Smith studied cats in 2020. Cats sleep a lot during the day. "
    "Cats also hunt mice at night. The weather was sunny. "
    "Sleeping cats and hunting cats are both common cats. Is it true? Yes!"
)


def test_split_sentences_handles_abbreviations():
    sents = split_sentences(DOC)
    assert sents[0] == "Dr. Smith studied cats in 2020."
    assert len(sents) == 7


def test_split_sentences_empty():
    assert split_sentences("   ") == []


def test_lead_k():
    assert lead_k(DOC, 2) == "Dr. Smith studied cats in 2020. Cats sleep a lot during the day."


def test_textrank_returns_k_sentences_in_order():
    out = textrank(DOC, k=2)
    sents = split_sentences(out)
    assert len(sents) == 2
    src = split_sentences(DOC)
    assert [src.index(s) for s in sents] == sorted(src.index(s) for s in sents)


def test_textrank_prefers_central_sentence():
    assert "common cats" in textrank(DOC, k=1)


def test_textrank_short_input():
    assert textrank("One short sentence here.", k=3) == "One short sentence here."
