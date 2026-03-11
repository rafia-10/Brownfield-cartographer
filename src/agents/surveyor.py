"""
surveyor.py
~~~~~~~~~~~
The Surveyor agent walks a repository and builds a ModuleGraph.

Responsibilities
  1. Discover all Python (and YAML) files under the repo root.
  2. Parse each Python file with tree_sitter_analyzer.
  3. Translate ImportRecord objects into DependencyEdge objects,
     resolving relative imports to dot-separated module ids.
  4. Detect circular dependencies via DFS.
  5. Return a fully populated ModuleGraph.
"""

from __future__ import annotations

import logging
from pathlib import Path
from src.analyzers.universal_analyzer import UniversalAnalyzer, UniversalResult
from src.utils.git_utils import GitMetrics
from src.models.nodes import (
    DependencyEdge,
    DependencyKind,
    GraphMetadata,
    Language,
    ModuleGraph,
    ModuleNode,
)

log = logging.getLogger(__name__)

# File globs the Surveyor will collect
_PYTHON_GLOB = "**/*.py"
_YAML_GLOBS = ("**/*.yaml", "**/*.yml")

# Directories to always skip
_SKIP_DIRS = {
    ".git", "__pycache__", ".venv", "venv", "env",
    "node_modules", ".tox", "dist", "build", ".eggs",
    ".cartography",
}


# ---------------------------------------------------------------------------
# Path → module id helpers
# ---------------------------------------------------------------------------


def _path_to_module_id(file_path: Path, repo_root: Path) -> str:
    """
    Convert an absolute file path to a dot-separated module id relative to
    the repo root.

    Examples
    --------
    /repo/src/agents/surveyor.py  →  src.agents.surveyor
    /repo/src/models/__init__.py  →  src.models
    """
    try:
        rel = file_path.relative_to(repo_root)
    except ValueError:
        rel = file_path

    parts = list(rel.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts) if parts else str(rel)


def _resolve_import(
    raw_module: str,
    importer_id: str,
    is_relative: bool = False,
) -> str:
    """
    Best-effort: turn a raw import string into a dot-separated module id.

    For absolute imports we just normalise the string.
    For relative imports we resolve against the importer's package.
    """
    if not is_relative:
        return raw_module.strip(".")

    # relative: climb up from the importer's package
    parts = importer_id.split(".")
    # strip trailing module name to get the package
    package_parts = parts[:-1]

    # count leading dots
    dots = len(raw_module) - len(raw_module.lstrip("."))
    climb = max(dots - 1, 0)
    base = package_parts[: len(package_parts) - climb] if climb <= len(package_parts) else []
    rest = raw_module.lstrip(".")
    if rest:
        base.append(rest)
    return ".".join(base)


# ---------------------------------------------------------------------------
# Circular dependency detection
# ---------------------------------------------------------------------------


def _find_cycles(nodes: list[str], edges: list[tuple[str, str]]) -> list[list[str]]:
    """
    Return all strongly connected components with > 1 node (cycles).
    Uses Tarjan's algorithm via an iterative DFS.
    """
    adj: dict[str, list[str]] = {n: [] for n in nodes}
    for src, tgt in edges:
        if src in adj:
            adj[src].append(tgt)

    index_counter = [0]
    stack: list[str] = []
    lowlink: dict[str, int] = {}
    index: dict[str, int] = {}
    on_stack: dict[str, bool] = {}
    sccs: list[list[str]] = []

    def strongconnect(v: str) -> None:
        index[v] = index_counter[0]
        lowlink[v] = index_counter[0]
        index_counter[0] += 1
        stack.append(v)
        on_stack[v] = True

        for w in adj.get(v, []):
            if w not in index:
                strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink.get(w, lowlink[v]))
            elif on_stack.get(w, False):
                lowlink[v] = min(lowlink[v], index[w])

        if lowlink[v] == index[v]:
            scc: list[str] = []
            while True:
                w = stack.pop()
                on_stack[w] = False
                scc.append(w)
                if w == v:
                    break
            if len(scc) > 1:
                sccs.append(scc)

    import sys
    sys.setrecursionlimit(max(sys.getrecursionlimit(), len(nodes) * 2 + 100))

    for v in nodes:
        if v not in index:
            strongconnect(v)

    return sccs


# ---------------------------------------------------------------------------
# Public agent
# ---------------------------------------------------------------------------


class Surveyor:
    """
    Walks a repository and builds a ModuleGraph.

    Parameters
    ----------
    repo_root : str | Path
        Root directory of the repository to scan.
    exclude_dirs : set[str] | None
        Extra directory names to skip (merged with the built-in skip list).
    """

    def __init__(
        self,
        repo_root: str | Path,
        exclude_dirs: set[str] | None = None,
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.skip_dirs = _SKIP_DIRS | (exclude_dirs or set())
        self.git = GitMetrics(self.repo_root)
        self.analyzer = UniversalAnalyzer()

    def _is_excluded(self, path: Path) -> bool:
        return any(part in self.skip_dirs for part in path.parts)

    def _discover_python_files(self) -> list[Path]:
        files: list[Path] = []
        for p in self.repo_root.rglob(_PYTHON_GLOB):
            if not self._is_excluded(p):
                files.append(p)
        return sorted(files)

    def _discover_yaml_files(self) -> list[Path]:
        files: list[Path] = []
        for glob in _YAML_GLOBS:
            for p in self.repo_root.rglob(glob):
                if not self._is_excluded(p):
                    files.append(p)
        return sorted(files)

    def run(self) -> ModuleGraph:
        """Scan the repo and return a complete ModuleGraph."""
        log.info("Surveyor: scanning %s", self.repo_root)

        nodes: dict[str, ModuleNode] = {}
        raw_edges: list[tuple[str, str, DependencyKind, int | None]] = []

        # --- Analysis (Universal) ---
        all_files = self._discover_python_files() + self._discover_yaml_files()
        
        for f in all_files:
            mod_id = _path_to_module_id(f, self.repo_root)
            
            # Use universal analyzer for symbols/metadata
            uni_res: UniversalResult = self.analyzer.analyze(f)
            
            # Use specialized python analyzer for imports/IO
            if uni_res.language == "python":
                py_res: AnalysisResult = analyze_file(f)
                result_classes = py_res.classes
                result_functions = py_res.functions
                result_imports = [ir.module for ir in py_res.imports]
                result_loc = py_res.loc
                
                for imp in py_res.imports:
                    is_rel = imp.module.startswith(".")
                    target_id = _resolve_import(imp.module, mod_id, is_rel)
                    kind = (
                        DependencyKind.WILDCARD
                        if imp.is_wildcard
                        else DependencyKind.FROM_IMPORT
                        if imp.names
                        else DependencyKind.IMPORT
                    )
                    raw_edges.append((mod_id, target_id, kind, imp.line))
            else:
                result_classes = []
                result_functions = []
                result_imports = []
                result_loc = uni_res.loc

            node = ModuleNode(
                id=mod_id,
                path=str(f.relative_to(self.repo_root)),
                language=Language(uni_res.language),
                classes=result_classes,
                functions=result_functions + uni_res.symbols if uni_res.language == "yaml" else result_functions,
                imports=result_imports,
                loc=result_loc,
                git_velocity=self.git.get_velocity(f),
            )
            nodes[mod_id] = node

        # --- Deduplicate edges ---
        seen_edges: set[tuple[str, str]] = set()
        edges: list[DependencyEdge] = []
        for src, tgt, kind, line in raw_edges:
            key = (src, tgt)
            if key not in seen_edges:
                seen_edges.add(key)
                edges.append(
                    DependencyEdge(source=src, target=tgt, kind=kind, line=line)
                )

        # --- Cycle detection ---
        all_node_ids = list(nodes.keys())
        edge_pairs = [(e.source, e.target) for e in edges]
        cycles = _find_cycles(all_node_ids, edge_pairs)
        if cycles:
            log.warning(
                "Surveyor: %d circular dependency group(s) detected", len(cycles)
            )

        # --- Dead code detection (module level) ---
        import_targets = {e.target for e in edges}
        for mod_id, node in nodes.items():
            if mod_id not in import_targets and not node.is_entry_point:
                node.extra["dead_code_candidate"] = True

        # --- Assemble graph ---
        metadata = GraphMetadata(
            repo_path=str(self.repo_root),
            node_count=len(nodes),
            edge_count=len(edges),
            circular_dependency_count=len(cycles),
        )

        return ModuleGraph(
            metadata=metadata,
            nodes=list(nodes.values()),
            edges=edges,
        )
