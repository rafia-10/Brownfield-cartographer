"""
sql_lineage.py
~~~~~~~~~~~~~~
Parses SQL files using sqlglot to extract table-level data lineage.

For each SQL statement it returns a LineageRecord describing:
  • which tables/views are read  (sources)
  • which table/view is written  (target)
  • what kind of operation it is (SELECT / INSERT / CTAS)

Supported statement types
  SELECT                → sources only (no target)
  INSERT INTO … SELECT  → sources + target  (INSERT)
  CREATE TABLE … AS     → sources + target  (CTAS)
  CREATE VIEW … AS      → sources + target  (CTAS, kind=VIEW)

Dialect support: any dialect sqlglot understands (postgres, bigquery,
snowflake, spark, …).  Pass dialect=None to let sqlglot auto-detect.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import sqlglot
import sqlglot.expressions as exp


# ---------------------------------------------------------------------------
# Public data classes
# ---------------------------------------------------------------------------


@dataclass
class LineageRecord:
    """Table-level lineage extracted from a single SQL statement."""

    sources: list[str]          # tables/views read in this statement
    target: str | None          # table/view written; None for bare SELECT
    operation: str              # "SELECT" | "INSERT" | "CTAS"
    source_file: str = ""
    statement_index: int = 0    # 0-based index within the file


@dataclass
class SQLAnalysisResult:
    path: str
    dialect: str | None
    records: list[LineageRecord] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _qualified_name(table_expr: exp.Table) -> str:
    """Return a fully-qualified 'schema.table' or just 'table' string."""
    parts = []
    if table_expr.args.get("db"):
        parts.append(table_expr.args["db"].name)
    parts.append(table_expr.name)
    return ".".join(p for p in parts if p)


def _source_tables(expression: exp.Expression) -> list[str]:
    """Walk the expression tree and collect all FROM / JOIN tables."""
    tables: list[str] = []
    for tbl in expression.find_all(exp.Table):
        name = _qualified_name(tbl)
        if name:
            tables.append(name)
    return _dedupe(tables)


def _dedupe(seq: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in seq:
        low = item.lower()
        if low not in seen:
            seen.add(low)
            out.append(item)
    return out


# ---------------------------------------------------------------------------
# Statement dispatchers
# ---------------------------------------------------------------------------


def _handle_select(stmt: exp.Select, idx: int, path: str) -> LineageRecord | None:
    sources = _source_tables(stmt)
    if not sources:
        return None
    return LineageRecord(
        sources=sources,
        target=None,
        operation="SELECT",
        source_file=path,
        statement_index=idx,
    )


def _handle_insert(stmt: exp.Insert, idx: int, path: str) -> LineageRecord | None:
    target_expr = stmt.args.get("this")
    target = _qualified_name(target_expr) if isinstance(target_expr, exp.Table) else None

    # The SELECT part lives inside the insert expression
    sources = _source_tables(stmt)
    # Remove the target from sources if sqlglot also picked it up
    if target:
        sources = [s for s in sources if s.lower() != target.lower()]

    return LineageRecord(
        sources=sources,
        target=target,
        operation="INSERT",
        source_file=path,
        statement_index=idx,
    )


def _handle_create(stmt: exp.Create, idx: int, path: str) -> LineageRecord | None:
    """Handles CREATE TABLE AS SELECT and CREATE VIEW AS SELECT."""
    kind_token = stmt.args.get("kind", "")
    kind_str = str(kind_token).upper() if kind_token else ""

    target_expr = stmt.args.get("this")
    if isinstance(target_expr, exp.Table):
        target = _qualified_name(target_expr)
    elif hasattr(target_expr, "name"):
        target = target_expr.name
    else:
        target = None

    # The nested SELECT
    sources = _source_tables(stmt)
    if target:
        sources = [s for s in sources if s.lower() != target.lower()]

    if not target and not sources:
        return None

    return LineageRecord(
        sources=sources,
        target=target,
        operation="CTAS",
        source_file=path,
        statement_index=idx,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def analyze_file(
    path: str | Path,
    dialect: str | None = None,
) -> SQLAnalysisResult:
    """
    Parse a SQL file and return all lineage records.

    Parameters
    ----------
    path:
        Absolute or relative path to the .sql file.
    dialect:
        sqlglot dialect name (e.g. "bigquery", "postgres", "snowflake").
        Pass None to let sqlglot auto-detect.

    Returns
    -------
    SQLAnalysisResult — never raises; errors are captured in result.errors.
    """
    p = Path(path)
    result = SQLAnalysisResult(path=str(p), dialect=dialect)

    try:
        sql_text = p.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        result.errors.append(f"Cannot read file: {exc}")
        return result

    try:
        statements = sqlglot.parse(sql_text, dialect=dialect, error_level=sqlglot.ErrorLevel.WARN)
    except Exception as exc:  # noqa: BLE001
        result.errors.append(f"Parse error: {exc}")
        return result

    for idx, stmt in enumerate(statements):
        if stmt is None:
            continue
        try:
            record: LineageRecord | None = None

            if isinstance(stmt, exp.Select):
                record = _handle_select(stmt, idx, str(p))
            elif isinstance(stmt, exp.Insert):
                record = _handle_insert(stmt, idx, str(p))
            elif isinstance(stmt, exp.Create):
                record = _handle_create(stmt, idx, str(p))

            if record is not None:
                result.records.append(record)

        except Exception as exc:  # noqa: BLE001
            result.errors.append(f"Statement {idx}: {exc}")

    return result
