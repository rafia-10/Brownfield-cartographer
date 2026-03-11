"""
Pydantic v2 models for every node / edge type used across the Brownfield Cartographer.

ModuleGraph  → produced by the Surveyor agent
LineageGraph → produced by the Hydrologist agent
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class DependencyKind(str, Enum):
    """How one Python module references another."""

    IMPORT = "import"          # import foo
    FROM_IMPORT = "from_import"  # from foo import bar
    WILDCARD = "wildcard"      # from foo import *
    DYNAMIC = "dynamic"        # importlib / __import__


class LineageOperation(str, Enum):
    """What kind of data movement connects two nodes."""

    SELECT = "SELECT"
    INSERT = "INSERT"
    CTAS = "CTAS"         # CREATE TABLE AS SELECT
    READ_CSV = "READ_CSV"
    READ_SQL = "READ_SQL"
    WRITE = "WRITE"       # to_csv / to_parquet / to_sql


class TableKind(str, Enum):
    TABLE = "table"
    VIEW = "view"
    FILE = "file"         # CSV / Parquet / flat file


class Language(str, Enum):
    PYTHON = "python"
    SQL = "sql"
    YAML = "yaml"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Module graph models
# ---------------------------------------------------------------------------


class ModuleNode(BaseModel):
    """A single Python/YAML file treated as a module."""

    id: str = Field(..., description="Unique key — dot-separated module path, e.g. src.agents.surveyor")
    path: str = Field(..., description="Absolute or repo-relative file path")
    language: Language = Language.PYTHON
    classes: list[str] = Field(default_factory=list)
    functions: list[str] = Field(default_factory=list)
    imports: list[str] = Field(default_factory=list, description="Raw import strings found in the file")
    is_entry_point: bool = False
    loc: int = Field(default=0, description="Lines of code")
    extra: dict[str, Any] = Field(default_factory=dict)


class DependencyEdge(BaseModel):
    """A directed dependency from one module to another."""

    source: str = Field(..., description="Module id of the importer")
    target: str = Field(..., description="Module id of the imported")
    kind: DependencyKind = DependencyKind.IMPORT
    line: int | None = None


class GraphMetadata(BaseModel):
    repo_path: str
    generated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    node_count: int = 0
    edge_count: int = 0
    circular_dependency_count: int = 0
    hub_modules: list[str] = Field(default_factory=list, description="Top modules by betweenness centrality")
    extra: dict[str, Any] = Field(default_factory=dict)


class ModuleGraph(BaseModel):
    """Root object for the module / dependency graph."""

    metadata: GraphMetadata
    nodes: list[ModuleNode] = Field(default_factory=list)
    edges: list[DependencyEdge] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Lineage graph models
# ---------------------------------------------------------------------------


class TableNode(BaseModel):
    """A data artefact: SQL table, view, or flat file."""

    id: str = Field(..., description="Unique key — qualified table name or file path")
    name: str
    kind: TableKind = TableKind.TABLE
    source_file: str | None = None   # which .sql or .py file introduced it
    is_source: bool = False          # no in-edges → raw source
    is_sink: bool = False            # no out-edges → final output
    extra: dict[str, Any] = Field(default_factory=dict)


class LineageEdge(BaseModel):
    """A directed data-flow edge from one TableNode to another."""

    source: str = Field(..., description="TableNode id — data origin")
    target: str = Field(..., description="TableNode id — data destination")
    operation: LineageOperation
    source_file: str | None = None
    line: int | None = None


class LineageMetadata(BaseModel):
    repo_path: str
    generated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    node_count: int = 0
    edge_count: int = 0
    source_count: int = 0
    sink_count: int = 0
    extra: dict[str, Any] = Field(default_factory=dict)


class LineageGraph(BaseModel):
    """Root object for the data-lineage graph."""

    metadata: LineageMetadata
    nodes: list[TableNode] = Field(default_factory=list)
    edges: list[LineageEdge] = Field(default_factory=list)
