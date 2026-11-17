from stylesum.metrics import StyleClassifier, mean_markers, rouge, style_markers


def test_rouge_identical_is_100():
    r = rouge(["The cat sat on the mat."], ["The cat sat on the mat."])
    assert r == {"rouge1": 100.0, "rouge2": 100.0, "rougeLsum": 100.0}


def test_rouge_disjoint_is_0():
    assert rouge(["alpha beta"], ["gamma delta"])["rouge1"] == 0.0


def test_rouge_empty():
    assert rouge([], [])["rouge1"] == 0.0


def test_style_markers_casual_vs_academic():
    casual = style_markers("lol i can't believe i did that, my bad")
    academic = style_markers(
        "We characterize transcriptional heterogeneity using comprehensive genomic methodologies."
    )
    assert casual["first_person_pct"] > 0 and academic["first_person_pct"] == 0
    assert casual["contractions_pct"] > 0
    assert academic["fkgl"] > casual["fkgl"]
    assert academic["avg_word_len"] > casual["avg_word_len"]


def test_style_markers_empty_text():
    assert style_markers("")["words"] == 0


def test_mean_markers():
    m = mean_markers(["a b c", "a b c d e"])
    assert m["words"] == 4.0


def test_style_classifier_roundtrip(tmp_path):
    casual = [f"lol i totally messed up my {x} today haha" for x in "car dog cat phone keys".split()]
    formal = [f"Officials announced the {x} policy on Tuesday." for x in "tax trade health energy water".split()]
    texts, labels = casual + formal, ["casual"] * 5 + ["formal"] * 5
    clf = StyleClassifier().fit(texts, labels)
    assert clf.accuracy(texts, labels) == 100.0
    clf.save(tmp_path / "clf.pkl")
    again = StyleClassifier.load(tmp_path / "clf.pkl")
    assert again.predict(["haha i messed up my bike lol"]) == ["casual"]


def test_marker_classifier_ignores_topic():
    from stylesum.metrics import MarkerStyleClassifier

    casual = [f"lol i can't believe i broke my {x}. my bad haha." for x in "car dog cat phone keys bike".split()]
    academic = [f"We comprehensively characterize heterogeneous {x} regulation using quantitative "
                f"transcriptomic methodologies across experimental conditions." for x in
                "metabolic immunological neuronal hormonal genomic cellular".split()]
    clf = MarkerStyleClassifier().fit(casual + academic, ["casual"] * 6 + ["academic"] * 6)
    # same topic word ("genomic"), casual register -> casual
    assert clf.predict(["haha i can't believe i lost my genomic notes lol."]) == ["casual"]
