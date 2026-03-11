"""
hydrologist.py
~~~~~~~~~~~~~~
The Hydrologist agent traces data flow across a repository and builds a
LineageGraph.

It ingests two kinds of signals:

  1. SQL files  → parsed by sql_lineage.analyze_file
                  Each LineageRecord maps source tables → target table.

  2. Python files → parsed by tree_sitter_analyzer.analyze_file
                    DataIOCall objects with operation READ_CSV / READ_SQL /
                    WRITE tell us which flat files or tables are consumed /
                    produced.

The agent then:
  • Creates a TableNode for every unique table/file name encountered.
  • Creates a LineageEdge for every source → target pair.
  • Labels every TableNode as is_source (no in-edges) or is_sink (no out-edges).
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.analyzers.sql_lineage import SQLAnalysisResult, analyze_file as analyze_sql
from src.analyzers.tree_sitter_analyzer import AnalysisResult, analyze_file as analyze_py
from src.models.nodes import (
    LineageEdge,
    LineageGraph,
    LineageMetadata,
    LineageOperation,
    TableKind,
    TableNode,
)

log = logging.getLogger(__name__)

_SKIP_DIRS = {
    ".git", "__pycache__", ".venv", "venv", "env",
    "node_modules", ".tox", "dist", "build", ".eggs",
    ".cartography",
}

_OP_MAP: dict[str, LineageOperation] = {
    "SELECT": LineageOperation.SELECT,
    "INSERT": LineageOperation.INSERT,
    "CTAS": LineageOperation.CTAS,
    "READ_CSV": LineageOperation.READ_CSV,
    "READ_SQL": LineageOperation.READ_SQL,
    "WRITE": LineageOperation.WRITE,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _table_id(name: str) -> str:
    """Normalise a table / file identifier to lowercase, stripped."""
    return name.strip().lower()


def _kind_from_name(name: str) -> TableKind:
    """Guess whether an identifier refers to a file or a DB table."""
    low = name.lower()
    if any(low.endswith(ext) for ext in (".csv", ".parquet", ".xlsx", ".json", ".tsv")):
        return TableKind.FILE
    return TableKind.TABLE


def _is_excluded(path: Path, skip: set[str]) -> bool:
    return any(part in skip for part in path.parts)


# ---------------------------------------------------------------------------
# Public agent
# ---------------------------------------------------------------------------


class Hydrologist:
    """
    Builds a data-lineage graph for the given repository.

    Parameters
    ----------
    repo_root : str | Path
        Root of the repository to scan.
    sql_dialect : str | None
        sqlglot dialect (e.g. "bigquery", "postgres").  None = auto-detect.
    exclude_dirs : set[str] | None
        Extra directories to skip.
    """

    def __init__(
        self,
        repo_root: str | Path,
        sql_dialect: str | None = None,
        exclude_dirs: set[str] | None = None,
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.sql_dialect = sql_dialect
        self.skip_dirs = _SKIP_DIRS | (exclude_dirs or set())

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def _sql_files(self) -> list[Path]:
        return sorted(
            p
            for p in self.repo_root.rglob("**/*.sql")
            if not _is_excluded(p, self.skip_dirs)
        )

    def _py_files(self) -> list[Path]:
        return sorted(
            p
            for p in self.repo_root.rglob("**/*.py")
            if not _is_excluded(p, self.skip_dirs)
        )

    def _discover_dbt(self) -> dict[str, Any]:
        """Look for dbt_project.yml and related models."""
        topology = {"models": [], "sources": []}
        for p in self.repo_root.rglob("dbt_project.yml"):
            if not _is_excluded(p, self.skip_dirs):
                topology["project_file"] = str(p)
                # Look for sources in the same or subdirs
                for src_file in p.parent.rglob("*.yml"):
                    if "sources" in src_file.name or "schema" in src_file.name:
                        topology["sources"].append(str(src_file))
        return topology

    # ------------------------------------------------------------------
    # Graph accumulators
    # ------------------------------------------------------------------

    def _ensure_node(
        self,
        name: str,
        nodes: dict[str, TableNode],
        source_file: str = "",
    ) -> TableNode:
        nid = _table_id(name)
        if nid not in nodes:
            nodes[nid] = TableNode(
                id=nid,
                name=name,
                kind=_kind_from_name(name),
                source_file=source_file or None,
            )
        return nodes[nid]

    def _add_edge(
        self,
        src_name: str,
        tgt_name: str,
        operation: LineageOperation,
        source_file: str,
        nodes: dict[str, TableNode],
        edges: list[LineageEdge],
        seen: set[tuple[str, str]],
        line: int | None = None,
    ) -> None:
        src_id = _table_id(src_name)
        tgt_id = _table_id(tgt_name)
        self._ensure_node(src_name, nodes, source_file)
        self._ensure_node(tgt_name, nodes, source_file)

        key = (src_id, tgt_id)
        if key not in seen:
            seen.add(key)
            edges.append(
                LineageEdge(
                    source=src_id,
                    target=tgt_id,
                    operation=operation,
                    source_file=source_file,
                    line=line,
                )
            )

    # ------------------------------------------------------------------
    # SQL processing
    # ------------------------------------------------------------------

    def _process_sql(
        self,
        path: Path,
        nodes: dict[str, TableNode],
        edges: list[LineageEdge],
        seen: set[tuple[str, str]],
    ) -> None:
        result: SQLAnalysisResult = analyze_sql(path, dialect=self.sql_dialect)
        rel = str(path.relative_to(self.repo_root))

        if result.errors:
            log.warning("Hydrologist [SQL] %s — %s", path.name, "; ".join(result.errors))

        for rec in result.records:
            if rec.target:
                for src in rec.sources:
                    op = _OP_MAP.get(rec.operation, LineageOperation.SELECT)
                    self._add_edge(src, rec.target, op, rel, nodes, edges, seen)
            else:
                # Bare SELECT — just ensure source nodes exist
                for src in rec.sources:
                    self._ensure_node(src, nodes, rel)

    # ------------------------------------------------------------------
    # Python processing
    # ------------------------------------------------------------------

    def _process_python(
        self,
        path: Path,
        nodes: dict[str, TableNode],
        edges: list[LineageEdge],
        seen: set[tuple[str, str]],
    ) -> None:
        result: AnalysisResult = analyze_py(path)
        rel = str(path.relative_to(self.repo_root))
        mod_id = rel  # use the file path as the "module" target for WRITE edges

        if result.errors:
            log.warning("Hydrologist [PY] %s — %s", path.name, "; ".join(result.errors))

        for call in result.io_calls:
            op = _OP_MAP.get(call.operation, LineageOperation.READ_CSV)

            # Best-effort: first positional arg is usually the file/table name
            resource = call.args[0] if call.args else call.kwargs.get("path_or_buf", "")
            if not resource:
                resource = call.kwargs.get("name", "")
            if not resource:
                # Synthesise a placeholder so the edge still appears
                resource = f"<{call.method_name}:{path.stem}:{call.line}>"

            if op == LineageOperation.WRITE:
                # Python file produces → resource
                self._add_edge(mod_id, resource, op, rel, nodes, edges, seen, call.line)
            else:
                # resource is consumed by the Python file
                self._add_edge(resource, mod_id, op, rel, nodes, edges, seen, call.line)

    # ------------------------------------------------------------------
    # Source / sink labelling
    # ------------------------------------------------------------------

    @staticmethod
    def _label_sources_sinks(
        nodes: dict[str, TableNode],
        edges: list[LineageEdge],
    ) -> None:
        has_in: set[str] = set()
        has_out: set[str] = set()
        for e in edges:
            has_out.add(e.source)
            has_in.add(e.target)

        for nid, node in nodes.items():
            node.is_source = nid not in has_in
            node.is_sink = nid not in has_out

    # ------------------------------------------------------------------
    # Query helpers (Public)
    # ------------------------------------------------------------------

    def find_sources(self, graph: LineageGraph) -> list[str]:
        """Return IDs of all source nodes."""
        return [n.id for n in graph.nodes if n.is_source]

    def find_sinks(self, graph: LineageGraph) -> list[str]:
        """Return IDs of all sink nodes."""
        return [n.id for n in graph.nodes if n.is_sink]

    def get_blast_radius(self, graph: LineageGraph, node_id: str, depth: int = 2) -> list[str]:
        """
        Calculates downstream impact of a change to node_id.
        Requires build KnowledgeGraph for traversal.
        """
        from src.graph.knowledge_graph import KnowledgeGraph
        kg = KnowledgeGraph.from_lineage_graph(graph)
        return list(kg.blast_radius(node_id, depth))

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> LineageGraph:
        """Scan the repo and return a complete LineageGraph."""
        log.info("Hydrologist: scanning %s", self.repo_root)

        nodes: dict[str, TableNode] = {}
        edges: list[LineageEdge] = []
        seen: set[tuple[str, str]] = set()

        for sql_file in self._sql_files():
            log.debug("Hydrologist: SQL %s", sql_file.name)
            self._process_sql(sql_file, nodes, edges, seen)

        for py_file in self._py_files():
            log.debug("Hydrologist: PY  %s", py_file.name)
            self._process_python(py_file, nodes, edges, seen)

        self._label_sources_sinks(nodes, edges)

        source_count = sum(1 for n in nodes.values() if n.is_source)
        sink_count = sum(1 for n in nodes.values() if n.is_sink)

        metadata = LineageMetadata(
            repo_path=str(self.repo_root),
            node_count=len(nodes),
            edge_count=len(edges),
            source_count=source_count,
            sink_count=sink_count,
        )

        return LineageGraph(
            metadata=metadata,
            nodes=list(nodes.values()),
            edges=edges,
        )
