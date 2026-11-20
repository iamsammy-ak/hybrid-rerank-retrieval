"""Typed metadata filters: an expression tree, a small text syntax, and compilation to boolean masks.

Expressions are Pydantic models, so the HTTP API accepts them as JSON::

    {"op": "and", "clauses": [
        {"op": "eq", "field": "category", "value": "databases"},
        {"op": "range", "field": "year", "gte": 2021}]}

The CLI accepts the same thing as text::

    category == "databases" and year >= 2021
    not (source in ["blog", "forum"]) or year between 2019 and 2020

Semantics (two-valued, by design): a document whose field is missing never satisfies ``eq``, ``in`` or
``range``; ``not`` simply inverts the mask, so ``not category == "x"`` *does* include documents without a
category. Filtering an unknown field raises :class:`FilterError` instead of silently matching nothing.

A compiled filter is a boolean NumPy mask over the corpus. The searcher can apply it *before* top-k selection
inside each retriever (pre-filtering: every returned hit matches, and ``k`` hits come back whenever ``k``
matching documents exist) or *after* retrieval (post-filtering: cheaper to bolt on, but selective filters leave
fewer than ``k`` hits). Both are implemented and benchmarked.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, model_validator

from .schemas import MetaValue

Scalar = bool | int | float | str


class FilterError(ValueError):
    """Raised for malformed filter text or a filter that does not fit the metadata."""


class _Node(BaseModel):
    model_config = ConfigDict(frozen=True)


class Eq(_Node):
    """``field == value``."""

    op: Literal["eq"] = "eq"
    field: str
    value: Scalar


class In(_Node):
    """``field in [v1, v2, ...]``."""

    op: Literal["in"] = "in"
    field: str
    values: list[Scalar] = Field(min_length=1)


class Range(_Node):
    """Numeric bounds on ``field``; at least one bound is required."""

    op: Literal["range"] = "range"
    field: str
    gt: float | None = None
    gte: float | None = None
    lt: float | None = None
    lte: float | None = None

    @model_validator(mode="after")
    def _needs_a_bound(self) -> Range:
        if self.gt is None and self.gte is None and self.lt is None and self.lte is None:
            raise ValueError("range needs at least one of gt, gte, lt, lte")
        return self


class And(_Node):
    """All clauses hold."""

    op: Literal["and"] = "and"
    clauses: list[FilterExpr] = Field(min_length=1)


class Or(_Node):
    """At least one clause holds."""

    op: Literal["or"] = "or"
    clauses: list[FilterExpr] = Field(min_length=1)


class Not(_Node):
    """The clause does not hold."""

    op: Literal["not"] = "not"
    clause: FilterExpr


FilterExpr = Annotated[Eq | In | Range | And | Or | Not, Field(discriminator="op")]

for _model in (And, Or, Not):
    _model.model_rebuild()

_FILTER_ADAPTER: TypeAdapter[FilterExpr] = TypeAdapter(FilterExpr)


def to_text(expr: FilterExpr) -> str:
    """Render an expression in the text syntax (round-trips through :func:`parse_filter`)."""

    def lit(v: Scalar) -> str:
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, str):
            return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
        if isinstance(v, float) and v.is_integer():
            return str(int(v))
        return repr(v)

    if isinstance(expr, Eq):
        return f"{expr.field} == {lit(expr.value)}"
    if isinstance(expr, In):
        return f"{expr.field} in [{', '.join(lit(v) for v in expr.values)}]"
    if isinstance(expr, Range):
        if expr.gte is not None and expr.lte is not None and expr.gt is None and expr.lt is None:
            return f"{expr.field} between {lit(expr.gte)} and {lit(expr.lte)}"
        parts = [f"{expr.field} {sym} {lit(val)}" for sym, val in
                 ((">", expr.gt), (">=", expr.gte), ("<", expr.lt), ("<=", expr.lte)) if val is not None]
        return parts[0] if len(parts) == 1 else "(" + " and ".join(parts) + ")"
    if isinstance(expr, And):
        return "(" + " and ".join(to_text(c) for c in expr.clauses) + ")"
    if isinstance(expr, Or):
        return "(" + " or ".join(to_text(c) for c in expr.clauses) + ")"
    return f"not {to_text(expr.clause)}"


# ---------------------------------------------------------------------------------------------- text syntax
_TOKEN_SPEC = [
    ("STRING", r'"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\''),
    ("NUMBER", r"-?\d+(?:\.\d+)?(?![A-Za-z_])"),
    ("OP", r"==|!=|<=|>=|=|<|>"),
    ("PUNCT", r"[()\[\],]"),
    ("WORD", r"[A-Za-z_][A-Za-z0-9_.\-]*"),
    ("SPACE", r"\s+"),
]
_LEXER = re.compile("|".join(f"(?P<{name}>{pattern})" for name, pattern in _TOKEN_SPEC))
_KEYWORDS = {"and", "or", "not", "in", "between", "true", "false"}
_KIND_NAMES = {"EOF": "end of input", "NUMBER": "a number", "WORD": "a field name", "OP": "an operator"}


@dataclass(frozen=True)
class _Tok:
    kind: str
    text: str
    pos: int


def _lex(text: str) -> list[_Tok]:
    tokens: list[_Tok] = []
    pos = 0
    while pos < len(text):
        match = _LEXER.match(text, pos)
        if match is None:
            raise FilterError(f"unexpected character {text[pos]!r} at position {pos}")
        kind = match.lastgroup or ""
        if kind != "SPACE":
            value = match.group()
            if kind == "WORD" and value.lower() in _KEYWORDS:
                kind, value = "KW", value.lower()
            tokens.append(_Tok(kind, value, pos))
        pos = match.end()
    tokens.append(_Tok("EOF", "", len(text)))
    return tokens


class _Parser:
    """Recursive-descent parser. Precedence, loosest first: ``or``, ``and``, ``not``, comparison."""

    def __init__(self, text: str) -> None:
        self.tokens = _lex(text)
        self.i = 0

    def peek(self) -> _Tok:
        return self.tokens[self.i]

    def take(self, kind: str, text: str | None = None) -> _Tok:
        tok = self.peek()
        if tok.kind != kind or (text is not None and tok.text != text):
            want = repr(text) if text else _KIND_NAMES.get(kind, kind.lower())
            got = repr(tok.text) if tok.text else "end of input"
            raise FilterError(f"expected {want} at position {tok.pos}, got {got}")
        self.i += 1
        return tok

    def accept(self, kind: str, text: str | None = None) -> bool:
        tok = self.peek()
        if tok.kind == kind and (text is None or tok.text == text):
            self.i += 1
            return True
        return False

    def parse(self) -> FilterExpr:
        expr = self.or_expr()
        self.take("EOF")
        return expr

    def or_expr(self) -> FilterExpr:
        clauses = [self.and_expr()]
        while self.accept("KW", "or"):
            clauses.append(self.and_expr())
        return clauses[0] if len(clauses) == 1 else Or(clauses=clauses)

    def and_expr(self) -> FilterExpr:
        clauses = [self.not_expr()]
        while self.accept("KW", "and"):
            clauses.append(self.not_expr())
        return clauses[0] if len(clauses) == 1 else And(clauses=clauses)

    def not_expr(self) -> FilterExpr:
        if self.accept("KW", "not"):
            return Not(clause=self.not_expr())
        if self.accept("PUNCT", "("):
            expr = self.or_expr()
            self.take("PUNCT", ")")
            return expr
        return self.comparison()

    def literal(self) -> Scalar:
        tok = self.peek()
        self.i += 1
        if tok.kind == "STRING":
            return re.sub(r"\\(.)", r"\1", tok.text[1:-1])
        if tok.kind == "NUMBER":
            return float(tok.text) if "." in tok.text else int(tok.text)
        if tok.kind == "KW" and tok.text in ("true", "false"):
            return tok.text == "true"
        if tok.kind == "WORD":
            return tok.text
        raise FilterError(f"expected a value at position {tok.pos}, got {tok.text or 'end of input'!r}")

    def number(self) -> float:
        tok = self.take("NUMBER")
        return float(tok.text)

    def comparison(self) -> FilterExpr:
        field = self.take("WORD").text
        tok = self.peek()
        if tok.kind == "KW" and tok.text == "in":
            self.i += 1
            self.take("PUNCT", "[")
            values = [self.literal()]
            while self.accept("PUNCT", ","):
                values.append(self.literal())
            self.take("PUNCT", "]")
            return In(field=field, values=values)
        if tok.kind == "KW" and tok.text == "between":
            self.i += 1
            low = self.number()
            self.take("KW", "and")
            return Range(field=field, gte=low, lte=self.number())
        op = self.take("OP").text
        if op in ("==", "="):
            return Eq(field=field, value=self.literal())
        if op == "!=":
            return Not(clause=Eq(field=field, value=self.literal()))
        bound = self.number()
        key = {"<": "lt", "<=": "lte", ">": "gt", ">=": "gte"}[op]
        return Range.model_validate({"field": field, key: bound})


def parse_filter(text: str) -> FilterExpr:
    """Parse the text syntax into an expression tree; raises :class:`FilterError` with a position on errors."""
    if not text.strip():
        raise FilterError("empty filter")
    return _Parser(text).parse()


# ---------------------------------------------------------------------------------------------- compilation
@dataclass
class _Column:
    kind: Literal["numeric", "boolean", "categorical"]
    values: np.ndarray  # float64 with NaN for numeric; int8 (-1 missing) for boolean; int32 codes (-1) otherwise
    categories: dict[str, int]


class MetadataTable:
    """Column-oriented copy of document metadata that compiles filters into boolean masks.

    Each field gets a column type from its non-missing values: all ``bool`` -> boolean, all ``int``/``float`` ->
    numeric (float64, NaN = missing), anything else -> categorical (strings stored as integer codes).
    """

    def __init__(self, records: Sequence[Mapping[str, MetaValue]]) -> None:
        self.n_docs = len(records)
        fields = sorted({key for rec in records for key in rec})
        self.columns: dict[str, _Column] = {f: self._build_column([rec.get(f) for rec in records]) for f in fields}

    @staticmethod
    def _build_column(raw: list[MetaValue]) -> _Column:
        present = [v for v in raw if v is not None]
        if present and all(isinstance(v, bool) for v in present):
            values = np.array([-1 if v is None else int(bool(v)) for v in raw], dtype=np.int8)
            return _Column("boolean", values, {})
        if present and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in present):
            values = np.array([np.nan if v is None else float(v) for v in raw], dtype=np.float64)
            return _Column("numeric", values, {})
        categories: dict[str, int] = {}
        codes = np.array([-1 if v is None else categories.setdefault(str(v), len(categories)) for v in raw],
                         dtype=np.int32)
        return _Column("categorical", codes, categories)

    @property
    def fields(self) -> dict[str, str]:
        """Field name -> column kind."""
        return {name: col.kind for name, col in self.columns.items()}

    def _column(self, field: str) -> _Column:
        if field not in self.columns:
            known = ", ".join(sorted(self.columns)) or "none"
            raise FilterError(f"unknown metadata field {field!r} (known: {known})")
        return self.columns[field]

    def _equals(self, col: _Column, field: str, value: Scalar) -> np.ndarray:
        if col.kind == "categorical":
            code = col.categories.get(value if isinstance(value, str) else to_text_literal(value))
            return np.zeros(self.n_docs, dtype=bool) if code is None else col.values == code
        if col.kind == "boolean":
            if not isinstance(value, bool):
                raise FilterError(f"field {field!r} is boolean; compare it with true or false")
            return col.values == int(value)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise FilterError(f"field {field!r} is numeric; got {value!r}")
        return col.values == float(value)

    def mask(self, expr: FilterExpr) -> np.ndarray:
        """Boolean mask of documents that satisfy ``expr``."""
        if isinstance(expr, Eq):
            return self._equals(self._column(expr.field), expr.field, expr.value)
        if isinstance(expr, In):
            col = self._column(expr.field)
            out = np.zeros(self.n_docs, dtype=bool)
            for value in expr.values:
                out |= self._equals(col, expr.field, value)
            return out
        if isinstance(expr, Range):
            col = self._column(expr.field)
            if col.kind != "numeric":
                raise FilterError(f"range filters need a numeric field; {expr.field!r} is {col.kind}")
            vals = col.values
            out = ~np.isnan(vals)
            with np.errstate(invalid="ignore"):
                if expr.gt is not None:
                    out &= vals > expr.gt
                if expr.gte is not None:
                    out &= vals >= expr.gte
                if expr.lt is not None:
                    out &= vals < expr.lt
                if expr.lte is not None:
                    out &= vals <= expr.lte
            return out
        if isinstance(expr, (And, Or)):
            masks = [self.mask(c) for c in expr.clauses]
            combined = masks[0].copy()
            for other in masks[1:]:
                if isinstance(expr, And):
                    combined &= other
                else:
                    combined |= other
            return combined
        return ~self.mask(expr.clause)


def to_text_literal(value: Scalar) -> str:
    """How a non-string literal is matched against a categorical column (``2021`` -> ``"2021"``)."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def coerce_filter(value: str | FilterExpr | Mapping[str, object] | None) -> FilterExpr | None:
    """Accept filter text, an expression, or its JSON form; return an expression (or ``None``)."""
    if value is None:
        return None
    if isinstance(value, str):
        return parse_filter(value)
    if isinstance(value, Mapping):
        try:
            return _FILTER_ADAPTER.validate_python(value)
        except ValidationError as exc:
            raise FilterError(f"invalid filter: {exc.errors()[0]['msg']}") from exc
    return value
