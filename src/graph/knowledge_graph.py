"""
knowledge_graph.py
~~~~~~~~~~~~~~~~~~
Wraps a ModuleGraph or LineageGraph in a NetworkX DiGraph and provides:

  • Graph metrics: node count, edge count, hub nodes (betweenness centrality)
  • JSON serialisation to the .cartography/ output directory

Usage
-----
    from src.graph.knowledge_graph import KnowledgeGraph
    from src.models.nodes import ModuleGraph

    mg: ModuleGraph = surveyor.run()
    kg = KnowledgeGraph.from_module_graph(mg)
    hub_modules = kg.hub_nodes(top_n=5)
    kg.export_json(Path(".cartography/module_graph.json"))
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import networkx as nx

from src.models.nodes import LineageGraph, ModuleGraph

log = logging.getLogger(__name__)


class KnowledgeGraph:
    """
    A thin NetworkX wrapper around a ModuleGraph or LineageGraph.

    Attributes
    ----------
    graph : nx.DiGraph
        The underlying directed graph.
    raw : ModuleGraph | LineageGraph
        The original Pydantic model so it can be round-tripped to JSON.
    """

    def __init__(self, graph: nx.DiGraph, raw: ModuleGraph | LineageGraph) -> None:
        self.graph = graph
        self.raw = raw

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_module_graph(cls, mg: ModuleGraph) -> "KnowledgeGraph":
        """Build a NetworkX DiGraph from a ModuleGraph."""
        G = nx.DiGraph()

        for node in mg.nodes:
            G.add_node(
                node.id,
                path=node.path,
                language=node.language.value,
                classes=node.classes,
                functions=node.functions,
                loc=node.loc,
            )

        for edge in mg.edges:
            G.add_edge(edge.source, edge.target, kind=edge.kind.value, line=edge.line)

        return cls(G, mg)

    @classmethod
    def from_lineage_graph(cls, lg: LineageGraph) -> "KnowledgeGraph":
        """Build a NetworkX DiGraph from a LineageGraph."""
        G = nx.DiGraph()

        for node in lg.nodes:
            G.add_node(
                node.id,
                name=node.name,
                kind=node.kind.value,
                is_source=node.is_source,
                is_sink=node.is_sink,
                source_file=node.source_file,
            )

        for edge in lg.edges:
            G.add_edge(
                edge.source,
                edge.target,
                operation=edge.operation.value,
                source_file=edge.source_file,
                line=edge.line,
            )

        return cls(G, lg)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def hub_nodes(self, top_n: int = 10) -> list[str]:
        """
        Return the top-N nodes by betweenness centrality.
        """
        if self.graph.number_of_nodes() < 2:
            return list(self.graph.nodes)[:top_n]

        try:
            centrality: dict[str, float] = nx.betweenness_centrality(
                self.graph, normalized=True
            )
            return sorted(centrality, key=centrality.get, reverse=True)[:top_n]
        except Exception:
            return sorted(
                self.graph.nodes, key=lambda n: self.graph.in_degree(n), reverse=True
            )[:top_n]

    def compute_pagerank(self) -> dict[str, float]:
        """Compute PageRank scores for all nodes."""
        if self.graph.number_of_nodes() < 2:
            return {node: 1.0 for node in self.graph.nodes}
        try:
            return nx.pagerank(self.graph, alpha=0.85)
        except Exception:
            return {node: 0.0 for node in self.graph.nodes}

    def strongly_connected_components(self) -> list[list[str]]:
        """Return SCCs with more than one node (i.e. cycles)."""
        return [list(scc) for scc in nx.strongly_connected_components(self.graph) if len(scc) > 1]

    def blast_radius(self, node_id: str, depth: int = 2) -> set[str]:
        """
        Find all downstream nodes affected by a change to node_id.
        Uses BFS on the graph.
        """
        if node_id not in self.graph:
            return set()
        
        affected = set()
        # For lineage, edges are A -> B (data flows from A to B)
        # So BFS forward finds downstream impact.
        # For module deps, edges are A -> B (A depends on B)
        # So for blast radius of B, we need to look at IN-EDGES (who depends on me).
        # We'll detect the type based on the 'raw' attribute.
        
        is_module = isinstance(self.raw, ModuleGraph)
        
        if is_module:
            # Who depends on me? (Reverse edges)
            edges = nx.bfs_edges(self.graph, node_id, reverse=True, depth_limit=depth)
            for _, v in edges:
                affected.add(v)
        else:
            # Where does my data go? (Forward edges)
            edges = nx.bfs_edges(self.graph, node_id, depth_limit=depth)
            for _, v in edges:
                affected.add(v)
        
        return affected

    def summary_stats(self) -> dict[str, Any]:
        """Return a dict of key metrics suitable for printing."""
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "cycles": len(self.strongly_connected_components()),
            "density": round(nx.density(self.graph), 4),
            "hub_nodes": self.hub_nodes(top_n=5),
        }

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def export_json(self, output_path: str | Path) -> Path:
        """Serialise to a JSON file with metadata enrichment."""
        out = Path(output_path).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)

        data = self.raw.model_dump(mode="json")
        stats = self.summary_stats()
        pagerank = self.compute_pagerank()

        # Update node metadata with PageRank in the raw model if it's a ModuleNode
        if isinstance(self.raw, ModuleGraph):
            for node in data.get("nodes", []):
                node_id = node.get("id")
                if node_id in pagerank:
                    node["pagerank"] = round(pagerank[node_id], 6)

        data.setdefault("metadata", {})
        data["metadata"].update({
            "hub_modules": stats["hub_nodes"],
            "node_count": stats["nodes"],
            "edge_count": stats["edges"],
        })

        out.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return out

    @classmethod
    def load_json(cls, path: str | Path, graph_type: str = "module") -> "KnowledgeGraph":
        """Load and validate a graph from JSON using Pydantic schemas."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Graph file not found: {p}")
        
        data = json.loads(p.read_text(encoding="utf-8"))
        if graph_type == "module":
            raw = ModuleGraph.model_validate(data)
            return cls.from_module_graph(raw)
        else:
            raw = LineageGraph.model_validate(data)
            return cls.from_lineage_graph(raw)
