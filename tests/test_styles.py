import pytest

from stylesum.styles import STYLE_NAMES, build_input, get_style


def test_all_styles_registered():
    assert set(STYLE_NAMES) == {"academic", "simple", "formal", "casual"}


def test_build_input_prefixes_instruction():
    out = build_input("  The cat sat.  ", "simple")
    assert out.startswith(get_style("simple").instruction + ": ")
    assert out.endswith("The cat sat.")


def test_prefixes_are_distinct():
    prefixes = {build_input("x", s) for s in STYLE_NAMES}
    assert len(prefixes) == len(STYLE_NAMES)


def test_style_lookup_is_case_insensitive():
    assert get_style(" Academic ").name == "academic"


def test_unknown_style_raises():
    with pytest.raises(ValueError):
        get_style("pirate")
