"""
tree_sitter_analyzer.py
~~~~~~~~~~~~~~~~~~~~~~~
Parses a Python source file using tree-sitter and extracts:

  • import / from-import statements     → DependencyEdge candidates
  • top-level class and function names  → ModuleNode.classes / .functions
  • data I/O calls (pandas-style)       → raw DataIOCall records for the Hydrologist

Requires: tree-sitter >= 0.23, tree-sitter-python >= 0.23
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# tree-sitter public API (≥0.23 uses Language objects directly)
import tree_sitter_python as tspython
from tree_sitter import Language, Node, Parser

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

PY_LANGUAGE = Language(tspython.language())
_PARSER = Parser(PY_LANGUAGE)

# Pandas / file I/O patterns we want to flag for the Hydrologist
_READ_PATTERNS: dict[str, str] = {
    "read_csv": "READ_CSV",
    "read_parquet": "READ_CSV",  # treat parquet reads the same as csv for lineage
    "read_sql": "READ_SQL",
    "read_sql_query": "READ_SQL",
    "read_sql_table": "READ_SQL",
    "read_excel": "READ_CSV",
    "read_json": "READ_CSV",
}

_WRITE_PATTERNS: dict[str, str] = {
    "to_csv": "WRITE",
    "to_parquet": "WRITE",
    "to_sql": "WRITE",
    "to_excel": "WRITE",
    "to_json": "WRITE",
}


# ---------------------------------------------------------------------------
# Public data classes (plain dataclasses so they're dependency-free)
# ---------------------------------------------------------------------------


@dataclass
class ImportRecord:
    module: str               # e.g. "os.path" / "pandas"
    names: list[str]          # specific names imported; empty = whole-module import
    alias: str | None         # import numpy as np → "np"
    is_wildcard: bool = False
    line: int = 0


@dataclass
class DataIOCall:
    """A detected pandas/file-read/write call in a Python file."""

    operation: str            # READ_CSV | READ_SQL | WRITE
    method_name: str          # e.g. read_csv
    args: list[str]           # positional string args (best-effort)
    kwargs: dict[str, str]    # keyword string args (best-effort)
    line: int = 0


@dataclass
class AnalysisResult:
    path: str
    language: str = "python"
    classes: list[str] = field(default_factory=list)
    functions: list[str] = field(default_factory=list)
    imports: list[ImportRecord] = field(default_factory=list)
    io_calls: list[DataIOCall] = field(default_factory=list)
    loc: int = 0
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _node_text(node: Node, src: bytes) -> str:
    return src[node.start_byte: node.end_byte].decode("utf-8", errors="replace")


def _string_value(node: Node, src: bytes) -> str | None:
    """Extract the string content from a string node, stripping quotes."""
    text = _node_text(node, src).strip()
    if (text.startswith('"') and text.endswith('"')) or (
        text.startswith("'") and text.endswith("'")
    ):
        return text[1:-1]
    if text.startswith('"""') or text.startswith("'''"):
        return text[3:-3]
    return None


def _extract_string_args(call_node: Node, src: bytes) -> tuple[list[str], dict[str, str]]:
    """Best-effort extraction of string literals from a call's argument list."""
    positional: list[str] = []
    keyword: dict[str, str] = {}

    args_node = next(
        (c for c in call_node.children if c.type == "argument_list"), None
    )
    if args_node is None:
        return positional, keyword

    for child in args_node.children:
        if child.type in ("string", "concatenated_string"):
            val = _string_value(child, src)
            if val:
                positional.append(val)
        elif child.type == "keyword_argument":
            key_node = child.child_by_field_name("name")
            val_node = child.child_by_field_name("value")
            if key_node and val_node and val_node.type in ("string", "concatenated_string"):
                val = _string_value(val_node, src)
                if val:
                    keyword[_node_text(key_node, src)] = val

    return positional, keyword


# ---------------------------------------------------------------------------
# Tree-sitter visitors
# ---------------------------------------------------------------------------


def _walk(node: Node, src: bytes, result: AnalysisResult, depth: int = 0) -> None:  # noqa: C901
    """Recursive tree walk — collects imports, class/function names, I/O calls."""

    ntype = node.type

    # --- imports ---
    if ntype == "import_statement":
        # import foo, bar as baz
        for name_node in node.named_children:
            if name_node.type == "dotted_name":
                result.imports.append(
                    ImportRecord(
                        module=_node_text(name_node, src),
                        names=[],
                        alias=None,
                        line=node.start_point[0] + 1,
                    )
                )
            elif name_node.type == "aliased_import":
                mod = name_node.child_by_field_name("name")
                alias = name_node.child_by_field_name("alias")
                if mod:
                    result.imports.append(
                        ImportRecord(
                            module=_node_text(mod, src),
                            names=[],
                            alias=_node_text(alias, src) if alias else None,
                            line=node.start_point[0] + 1,
                        )
                    )

    elif ntype == "import_from_statement":
        # from foo.bar import baz, * 
        mod_node = node.child_by_field_name("module_name")
        module = _node_text(mod_node, src) if mod_node else ""
        names: list[str] = []
        is_wildcard = False

        for child in node.named_children:
            if child.type == "wildcard_import":
                is_wildcard = True
            elif child.type in ("dotted_name", "aliased_import"):
                if child != mod_node:
                    names.append(_node_text(child, src))

        result.imports.append(
            ImportRecord(
                module=module,
                names=names,
                alias=None,
                is_wildcard=is_wildcard,
                line=node.start_point[0] + 1,
            )
        )

    # --- top-level class / function names (depth 1 = module body) ---
    elif ntype == "class_definition" and depth <= 1:
        name_node = node.child_by_field_name("name")
        if name_node:
            result.classes.append(_node_text(name_node, src))

    elif ntype == "function_definition" and depth <= 1:
        name_node = node.child_by_field_name("name")
        if name_node:
            result.functions.append(_node_text(name_node, src))

    # --- data I/O calls ---
    elif ntype == "call":
        func_node = node.child_by_field_name("function")
        if func_node is not None:
            func_text = _node_text(func_node, src)
            # Match tail of dotted call: pd.read_csv / df.to_csv / read_csv(...)
            method = func_text.rsplit(".", 1)[-1]

            op: str | None = None
            if method in _READ_PATTERNS:
                op = _READ_PATTERNS[method]
            elif method in _WRITE_PATTERNS:
                op = _WRITE_PATTERNS[method]

            if op:
                args, kwargs = _extract_string_args(node, src)
                result.io_calls.append(
                    DataIOCall(
                        operation=op,
                        method_name=method,
                        args=args,
                        kwargs=kwargs,
                        line=node.start_point[0] + 1,
                    )
                )

    # Recurse
    for child in node.children:
        _walk(child, src, result, depth + 1)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def analyze_file(path: str | Path) -> AnalysisResult:
    """
    Parse a single Python file and return an AnalysisResult.

    Returns an AnalysisResult with errors populated if the file cannot be
    read or parsed; never raises.
    """
    p = Path(path)
    result = AnalysisResult(path=str(p))

    try:
        src_bytes = p.read_bytes()
    except OSError as exc:
        result.errors.append(f"Cannot read file: {exc}")
        return result

    result.loc = src_bytes.count(b"\n") + 1

    try:
        tree = _PARSER.parse(src_bytes)
    except Exception as exc:  # noqa: BLE001
        result.errors.append(f"Parse error: {exc}")
        return result

    _walk(tree.root_node, src_bytes, result, depth=0)
    return result
