"""Text analysis for the sparse retriever: normalise, split, drop stopwords, optionally stem.

The pipeline, in order:

1. Unicode NFKD normalisation with combining marks removed, so ``"naïve"`` becomes ``"naive"``.
2. Lowercasing.
3. Splitting on anything that is not a letter or digit (``"IL-6"`` gives ``"il"``, ``"6"``).
4. Dropping English stopwords (a fixed list of 125 function words, below).
5. Optional *light* stemming (:func:`light_stem`): plural and ``-ed``/``-ing`` suffix stripping plus a trailing
   ``e`` rule. It is deliberately conservative: it conflates ``cell``/``cells`` and ``increase``/``increased``/
   ``increasing`` but leaves ``activation``/``activate`` apart, which keeps precision on technical text.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

STOPWORDS: frozenset[str] = frozenset(
    """
    a about above after again against all am an and any are as at be because been before being below between both
    but by can could did do does doing down during each few for from further had has have having he her here hers
    herself him himself his how i if in into is it its itself just me more most my myself no nor not of off on once
    only or other our ours ourselves out over own same she should so some such than that the their theirs them
    themselves then there these they this those through to too under until up very was we were what when where which
    while who whom why will with would you your yours yourself yourselves
    """.split()
)

_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)
_VOWELS = frozenset("aeiouy")
_NO_UNDOUBLE = frozenset("lsz")


def _has_vowel(stem: str) -> bool:
    return any(ch in _VOWELS for ch in stem)


def _undouble(stem: str) -> str:
    """``runn`` -> ``run``, but keep ``fall``, ``miss``, ``buzz``."""
    if len(stem) >= 2 and stem[-1] == stem[-2] and stem[-1] not in _VOWELS and stem[-1] not in _NO_UNDOUBLE:
        return stem[:-1]
    return stem


@lru_cache(maxsize=200_000)
def light_stem(word: str) -> str:
    """Conservative English suffix stripping (plurals, ``-ed``, ``-ing``, trailing ``e``).

    Words of three characters or fewer and anything containing a digit are returned unchanged, so codes such as
    ``"covid19"`` or ``"e1047"`` survive intact.
    """
    if len(word) <= 3 or not word.isalpha():
        return word
    # 1. Plurals.
    if word.endswith("ies") and len(word) > 4 and word[-4] not in "ae":
        word = word[:-3] + "y"
    elif word.endswith("sses"):
        word = word[:-2]
    elif word.endswith(("ches", "shes", "xes", "zes")):
        word = word[:-2]
    elif word.endswith("s") and not word.endswith(("ss", "us", "is")):
        word = word[:-1]
    # 2. Past tense and progressive forms.
    if word.endswith("ied") and len(word) > 4:
        word = word[:-3] + "y"
    elif word.endswith("eed"):
        pass
    elif word.endswith("ing") and len(word) - 3 >= 3 and _has_vowel(word[:-3]):
        word = _undouble(word[:-3])
    elif word.endswith("ed") and len(word) - 2 >= 3 and _has_vowel(word[:-2]):
        word = _undouble(word[:-2])
    # 3. Trailing e, so "increase" and "increas(ed)" meet.
    if word.endswith("e") and len(word) > 4:
        word = word[:-1]
    return word


class TokenizerConfig(BaseModel):
    """Settings for :class:`Tokenizer`; stored in the index manifest so queries are analysed like documents."""

    model_config = ConfigDict(frozen=True)

    lowercase: bool = True
    fold_accents: bool = True
    remove_stopwords: bool = True
    stemmer: Literal["none", "light"] = "light"
    min_length: int = Field(default=1, ge=1)


class Tokenizer:
    """Turns text into the token list both BM25 and the offline stand-ins use."""

    def __init__(self, config: TokenizerConfig | None = None) -> None:
        self.config = config or TokenizerConfig()

    def normalise(self, text: str) -> str:
        """Apply accent folding and lowercasing only."""
        if self.config.fold_accents:
            text = "".join(ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch))
        return text.lower() if self.config.lowercase else text

    def __call__(self, text: str) -> list[str]:
        cfg = self.config
        tokens = _TOKEN.findall(self.normalise(text))
        if cfg.remove_stopwords:
            tokens = [t for t in tokens if t not in STOPWORDS]
        if cfg.min_length > 1:
            tokens = [t for t in tokens if len(t) >= cfg.min_length]
        if cfg.stemmer == "light":
            tokens = [light_stem(t) for t in tokens]
        return tokens
