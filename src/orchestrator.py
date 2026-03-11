"""
orchestrator.py
~~~~~~~~~~~~~~~
Coordinates the Surveyor and Hydrologist agents, writes their outputs to
.cartography/, and prints a Rich summary table to the terminal.

Public API
----------
    from src.orchestrator import run
    run(repo_path="/path/to/repo", output_dir=".cartography")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from src.agents.hydrologist import Hydrologist
from src.agents.surveyor import Surveyor
from src.graph.knowledge_graph import KnowledgeGraph
from src.models.nodes import LineageGraph, ModuleGraph

log = logging.getLogger(__name__)
console = Console()


# ---------------------------------------------------------------------------
# Rich helpers
# ---------------------------------------------------------------------------


def _module_summary_table(kg: KnowledgeGraph, mg: ModuleGraph) -> Table:
    stats = kg.summary_stats()
    table = Table(
        title="📦 Module Graph — Surveyor Report",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Metric", style="dim")
    table.add_column("Value", justify="right")
    table.add_row("Python modules", str(sum(1 for n in mg.nodes if n.language.value == "python")))
    table.add_row("YAML files", str(sum(1 for n in mg.nodes if n.language.value == "yaml")))
    table.add_row("Dependency edges", str(stats["edges"]))
    table.add_row("Circular dependency groups", str(stats["cycles"]))
    table.add_row("Graph density", str(stats["density"]))
    table.add_row("Top hub modules", ", ".join(stats["hub_nodes"]) or "—")
    return table


def _lineage_summary_table(kg: KnowledgeGraph, lg: LineageGraph) -> Table:
    stats = kg.summary_stats()
    table = Table(
        title="🌊 Lineage Graph — Hydrologist Report",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold blue",
    )
    table.add_column("Metric", style="dim")
    table.add_column("Value", justify="right")
    table.add_row("Data nodes (tables/files)", str(stats["nodes"]))
    table.add_row("Lineage edges", str(stats["edges"]))
    table.add_row("Source nodes (raw inputs)", str(lg.metadata.source_count))
    table.add_row("Sink nodes (final outputs)", str(lg.metadata.sink_count))
    table.add_row("Top hub tables", ", ".join(stats["hub_nodes"]) or "—")
    return table


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run(
    repo_path: str | Path,
    output_dir: str | Path = ".cartography",
    sql_dialect: str | None = None,
    quiet: bool = False,
) -> dict[str, Any]:
    """
    Run the full Brownfield Cartographer pipeline.

    Parameters
    ----------
    repo_path : str | Path
        Root of the repository to scan.
    output_dir : str | Path
        Directory where JSON outputs are written (created if absent).
        Relative paths are resolved relative to repo_path.
    sql_dialect : str | None
        sqlglot dialect for SQL parsing (None = auto-detect).
    quiet : bool
        Suppress Rich terminal output.

    Returns
    -------
    dict
        {
            "module_graph_path": Path,
            "lineage_graph_path": Path,
            "module_stats": dict,
            "lineage_stats": dict,
        }
    """
    repo = Path(repo_path).resolve()
    if not Path(output_dir).is_absolute():
        out_dir = repo / output_dir
    else:
        out_dir = Path(output_dir)

    if not quiet:
        console.print(
            Panel.fit(
                f"[bold green]The Brownfield Cartographer[/bold green]\n"
                f"[dim]Scanning:[/dim] {repo}",
                border_style="green",
            )
        )

    # ── Surveyor ──────────────────────────────────────────────────────
    if not quiet:
        console.print("\n[cyan]🔭 Running Surveyor…[/cyan]")

    surveyor = Surveyor(repo_root=repo)
    mg: ModuleGraph = surveyor.run()
    kg_module = KnowledgeGraph.from_module_graph(mg)
    module_path = kg_module.export_json(out_dir / "module_graph.json")

    if not quiet:
        console.print(_module_summary_table(kg_module, mg))

    # ── Hydrologist ────────────────────────────────────────────────────
    if not quiet:
        console.print("\n[blue]🌊 Running Hydrologist…[/blue]")

    hydrologist = Hydrologist(repo_root=repo, sql_dialect=sql_dialect)
    lg: LineageGraph = hydrologist.run()
    kg_lineage = KnowledgeGraph.from_lineage_graph(lg)
    lineage_path = kg_lineage.export_json(out_dir / "lineage_graph.json")

    if not quiet:
        console.print(_lineage_summary_table(kg_lineage, lg))

    if not quiet:
        console.print(
            f"\n[bold green]✅ Done![/bold green]  "
            f"Outputs in [underline]{out_dir}[/underline]"
        )

    return {
        "module_graph_path": module_path,
        "lineage_graph_path": lineage_path,
        "module_stats": kg_module.summary_stats(),
        "lineage_stats": kg_lineage.summary_stats(),
    }
