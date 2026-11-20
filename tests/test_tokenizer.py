from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from hybrid_retrieval.tokenizer import STOPWORDS, Tokenizer, TokenizerConfig, light_stem


@pytest.mark.parametrize(
    ("word", "stem"),
    [
        ("cells", "cell"),
        ("studies", "study"),
        ("studied", "study"),
        ("studying", "study"),
        ("increased", "increas"),
        ("increase", "increas"),
        ("increasing", "increas"),
        ("running", "run"),
        ("stopped", "stop"),
        ("falling", "fall"),
        ("missed", "miss"),
        ("boxes", "box"),
        ("analysis", "analysis"),
        ("virus", "virus"),
        ("speed", "speed"),
        ("thing", "thing"),
        ("covid19", "covid19"),
        ("e1047", "e1047"),
        ("bed", "bed"),
    ],
)
def test_light_stem_examples(word: str, stem: str) -> None:
    assert light_stem(word) == stem


def test_inflections_conflate() -> None:
    forms = ["increase", "increases", "increased", "increasing"]
    assert len({light_stem(f) for f in forms}) == 1


@given(st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=20))
def test_light_stem_never_grows_or_empties(word: str) -> None:
    stem = light_stem(word)
    assert 0 < len(stem) <= len(word)
    assert stem[0] == word[0]


def test_tokenizer_pipeline() -> None:
    tok = Tokenizer()
    assert tok("The Naïve IL-6 assays were RUNNING!") == ["naiv", "il", "6", "assay", "run"]


def test_stopwords_removed_and_kept() -> None:
    text = "what is the effect of the drug"
    assert Tokenizer()(text) == ["effect", "drug"]
    keep = Tokenizer(TokenizerConfig(remove_stopwords=False, stemmer="none"))
    assert keep(text) == text.split()


def test_no_stemming_and_min_length() -> None:
    tok = Tokenizer(TokenizerConfig(stemmer="none", min_length=3))
    assert tok("Cells in a 3D culture grew") == ["cells", "culture", "grew"]


def test_underscore_and_punctuation_split() -> None:
    assert Tokenizer(TokenizerConfig(stemmer="none"))("snake_case,comma;semi") == ["snake", "case", "comma", "semi"]


@given(st.text(max_size=200))
def test_tokens_are_clean(text: str) -> None:
    for token in Tokenizer(TokenizerConfig(stemmer="none"))(text):
        assert token and token == token.lower() and token not in STOPWORDS
        assert all(ch.isalnum() for ch in token)
