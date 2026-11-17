from stylesum.data import (
    clean_cnn_article,
    clean_highlights,
    clean_reddit_post,
    clean_tldr,
    cnn_examples,
    plos_examples,
    split_groups,
    split_plos_article,
    tifu_examples,
    truncate_words,
)


def words(n, w="word"):
    return " ".join([w] * n)


def test_split_plos_article_separates_abstract():
    abstract, body = split_plos_article("The abstract.\nIntro text.\n\nResults here.")
    assert abstract == "The abstract."
    assert body == "Intro text. Results here."


def test_split_plos_article_single_section():
    assert split_plos_article("only one") == ("", "")


def test_clean_cnn_article_strips_dateline():
    assert clean_cnn_article("LONDON, England (CNN) -- The news.") == "The news."


def test_clean_cnn_article_strips_dailymail_byline():
    raw = "By . John Doe . PUBLISHED: . 10:00 EST, 1 May 2013 . | . UPDATED: . 11:00 EST, 1 May 2013 . Body starts."
    assert clean_cnn_article(raw) == "Body starts."


def test_clean_highlights_joins_bullets():
    assert clean_highlights("First point\nSecond point .\n") == "First point. Second point."


def test_clean_tldr_removes_prefix_and_markdown():
    assert clean_tldr("**TL;DR: i broke the  sink**") == "i broke the sink"


def test_clean_reddit_post_removes_urls_and_edits():
    out = clean_reddit_post("so this happened http://x.co/a today. EDIT: thanks guys")
    assert "http" not in out and "thanks" not in out


def test_truncate_words():
    assert truncate_words("a b c d", 2) == "a b"


def test_plos_examples_produce_paired_styles():
    row = {"article": words(100, "abs") + "\n" + words(400, "body"), "summary": words(100, "lay")}
    (pair,) = list(plos_examples([row]))
    assert [e["style"] for e in pair] == ["academic", "simple"]
    assert pair[0]["document"] == pair[1]["document"]
    assert "abs" not in pair[0]["document"]  # abstract must not leak into input


def test_plos_examples_filter_short():
    row = {"article": "short\n" + words(400), "summary": words(100)}
    assert list(plos_examples([row])) == []


def test_cnn_and_tifu_examples():
    cnn = list(cnn_examples([{"id": "x", "article": words(200), "highlights": words(20)}]))
    assert cnn[0][0]["style"] == "formal"
    tifu = list(tifu_examples([{"documents": words(150), "tldr": "tl;dr " + words(10)}]))
    assert tifu[0][0]["style"] == "casual"


def test_split_groups_keeps_pairs_together():
    groups = [[{"id": f"{i}a"}, {"id": f"{i}s"}] for i in range(20)]
    s = split_groups(groups, n_train=10, n_val=4, n_test=6)
    assert (len(s["train"]), len(s["val"]), len(s["test"])) == (20, 8, 12)
    ids = lambda rows: {r["id"][:-1] for r in rows}  # noqa: E731
    assert not ids(s["train"]) & ids(s["test"])
    assert not ids(s["train"]) & ids(s["val"])
