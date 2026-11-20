from __future__ import annotations

import re

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hybrid_retrieval.filters import (
    And,
    Eq,
    FilterError,
    In,
    MetadataTable,
    Not,
    Or,
    Range,
    coerce_filter,
    parse_filter,
    to_text,
)

RECORDS = [
    {"year": 2019, "category": "ml", "source": "blog", "reviewed": True},
    {"year": 2021, "category": "databases", "source": "docs", "reviewed": False},
    {"year": 2023, "category": "databases", "source": "forum"},
    {"year": None, "category": "security", "source": "docs", "reviewed": True},
    {"year": 2025.5, "source": "runbook"},
]
TABLE = MetadataTable(RECORDS)


def ids(mask: np.ndarray) -> list[int]:
    return np.flatnonzero(mask).tolist()


def test_column_types() -> None:
    assert TABLE.fields == {"category": "categorical", "reviewed": "boolean", "source": "categorical",
                            "year": "numeric"}


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('category == "databases"', [1, 2]),
        ("category = databases", [1, 2]),
        ("category != databases", [0, 3, 4]),
        ("year >= 2021", [1, 2, 4]),
        ("year < 2021", [0]),
        ("year between 2020 and 2023", [1, 2]),
        ("source in [docs, 'forum']", [1, 2, 3]),
        ("reviewed == true", [0, 3]),
        ("not reviewed == true", [1, 2, 4]),
        ("category == databases and year > 2021", [2]),
        ("category == ml or category == security and year > 2000", [0]),
        ("(category == ml or category == security) and source == docs", [3]),
        ("NOT (year >= 2020) AND source == blog", [0]),
        ("year == 2021", [1]),
        ("category == 2021", []),
    ],
)
def test_parse_and_mask(text: str, expected: list[int]) -> None:
    assert ids(TABLE.mask(parse_filter(text))) == expected


def test_precedence_and_binds_tighter_than_or() -> None:
    expr = parse_filter("a == 1 or b == 2 and c == 3")
    assert isinstance(expr, Or)
    assert isinstance(expr.clauses[1], And)


def test_missing_values_never_match_a_comparison() -> None:
    assert 3 not in ids(TABLE.mask(parse_filter("year >= 0")))
    assert 4 not in ids(TABLE.mask(parse_filter("category in [ml, databases, security]")))


@pytest.mark.parametrize(
    ("text", "fragment"),
    [
        ("", "empty"),
        ("year >=", "position 7"),
        ("year >= abc", "expected a number"),
        ("category == ", "expected a value"),
        ("(year > 1", "expected ')'"),
        ("year > 1 extra", "expected end of input"),
        ("== 3", "expected a field name"),
        ("year ~ 3", "unexpected character"),
        ("source in []", "expected a value"),
    ],
)
def test_syntax_errors_are_reported(text: str, fragment: str) -> None:
    with pytest.raises(FilterError, match=re.escape(fragment)):
        parse_filter(text)


@pytest.mark.parametrize(
    ("expr", "fragment"),
    [
        (Eq(field="colour", value="red"), "unknown metadata field"),
        (Range(field="category", gte=1), "numeric field"),
        (Eq(field="year", value="recent"), "numeric"),
        (Eq(field="reviewed", value=1), "boolean"),
    ],
)
def test_semantic_errors(expr: object, fragment: str) -> None:
    with pytest.raises(FilterError, match=fragment):
        TABLE.mask(expr)  # type: ignore[arg-type]


def test_range_requires_a_bound() -> None:
    with pytest.raises(ValueError):
        Range(field="year")


def test_json_form_and_coercion() -> None:
    expr = coerce_filter({"op": "and", "clauses": [{"op": "in", "field": "category", "values": ["databases"]},
                                                   {"op": "range", "field": "year", "lte": 2021}]})
    assert ids(TABLE.mask(expr)) == [1]  # type: ignore[arg-type]
    assert coerce_filter(None) is None
    assert coerce_filter(expr) is expr
    with pytest.raises(FilterError):
        coerce_filter({"op": "xor", "clauses": []})


def test_round_trip_examples() -> None:
    for text in ['category == "databases" and year >= 2021', "not source in [blog, forum]",
                 "year between 2019 and 2020 or reviewed == false", 'source == "say \\"hi\\""']:
        expr = parse_filter(text)
        assert parse_filter(to_text(expr)) == expr


fields = st.sampled_from(["year", "category", "source"])
leaf = st.one_of(
    st.builds(lambda v: Eq(field="category", value=v), st.sampled_from(["ml", "databases", "x y", 'q"uote'])),
    st.builds(lambda vs: In(field="source", values=vs), st.lists(st.sampled_from(["docs", "blog", "forum"]),
                                                                 min_size=1, max_size=3)),
    st.builds(lambda lo, hi: Range(field="year", gte=lo, lt=hi), st.integers(2015, 2026), st.integers(2015, 2027)),
    st.builds(lambda b: Eq(field="reviewed", value=b), st.booleans()),
)
exprs = st.recursive(leaf, lambda children: st.one_of(
    st.builds(lambda cs: And(clauses=cs), st.lists(children, min_size=1, max_size=3)),
    st.builds(lambda cs: Or(clauses=cs), st.lists(children, min_size=1, max_size=3)),
    st.builds(lambda c: Not(clause=c), children),
), max_leaves=8)


@settings(max_examples=150, deadline=None)
@given(exprs)
def test_text_round_trip_preserves_the_mask(expr: object) -> None:
    reparsed = parse_filter(to_text(expr))  # type: ignore[arg-type]
    np.testing.assert_array_equal(TABLE.mask(reparsed), TABLE.mask(expr))  # type: ignore[arg-type]


@given(exprs)
def test_not_is_the_complement(expr: object) -> None:
    np.testing.assert_array_equal(TABLE.mask(Not(clause=expr)), ~TABLE.mask(expr))  # type: ignore[arg-type]
