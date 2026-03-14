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
import shutil
import tempfile
import subprocess
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from src.agents.hydrologist import Hydrologist
from src.agents.surveyor import Surveyor
from src.agents.semanticist import Semanticist
from src.agents.archivist import Archivist
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
    use_semantic: bool = False,
    run_archivist: bool = False,
    incremental: bool = False,
    quiet: bool = False,
    summary_only: bool = False,
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
    # --- GitHub URL Support ---
    is_temp_repo = False
    repo_to_scan = repo_path
    if str(repo_path).startswith(("http://", "https://", "git@")):
        temp_dir = Path(tempfile.mkdtemp(prefix="cartographer_"))
        if not quiet:
            console.print(f"[cyan]Cloning remote repository...[/cyan]")
        try:
            subprocess.run(["git", "clone", str(repo_path), str(temp_dir)], check=True, capture_output=True)
            repo_to_scan = temp_dir
            is_temp_repo = True
        except Exception as e:
            console.print(f"[bold red]Error cloning repository:[/bold red] {e}")
            raise

    repo = Path(repo_to_scan).resolve()
    if not repo.exists():
        console.print(f"[bold red]Error:[/bold red] Repo path {repo} does not exist.")
        return {}

    if not Path(output_dir).is_absolute():
        # If we cloned to temp, put outputs in current dir or specific path
        if is_temp_repo:
            out_dir = Path.cwd() / output_dir
        else:
            out_dir = repo / output_dir
    else:
        out_dir = Path(output_dir)

    if not quiet:
        console.print(
            Panel.fit(
                f"[bold green]The Brownfield Cartographer[/bold green]\n"
                f"[dim]Scanning:[/dim] {repo if not is_temp_repo else repo_path}",
                border_style="green",
            )
        )

    archivist = Archivist(repo_root=repo, output_dir=out_dir)
    
    # ── Surveyor ──────────────────────────────────────────────────────
    if not quiet:
        console.print("\n[cyan]🔭 Running Surveyor…[/cyan]")

    surveyor = Surveyor(repo_root=repo)
    
    # Handle incremental mode
    if incremental:
        changed_files = archivist.get_changed_files()
        if changed_files:
            if not quiet:
                console.print(f"[dim]Incremental mode: {len(changed_files)} changed files detected.[/dim]")
            # Note: A real incremental update would merge into existing graph.
            # For v1, we'll just log it and proceed with full scan if no existing graph found,
            # or in simpler implementations, we might just filter discovery.
            # To keep it simple for now, we'll inform the user but stick to full scan or 
            # we can pass the file list to Surveyor if supported.
            # Let's assume Surveyor should only scan these if provided.
            # (Self-correction: Surveyor doesn't take a file list yet, sticking to full scan for MVP 
            # to avoid complex merging logic, but logging the intent as requested).
            archivist.log_trace("Surveyor", "Incremental Check", {"changed_files": [str(p) for p in changed_files]})

    mg: ModuleGraph = surveyor.run()
    kg_module = KnowledgeGraph.from_module_graph(mg)
    module_path = None
    if not summary_only:
        module_path = kg_module.export_json(out_dir / "module_graph.json")
    # Sync hubs back to the model object so later agents see them
    mg.metadata.hub_modules = kg_module.hub_nodes(top_n=5)

    if not quiet:
        console.print(_module_summary_table(kg_module, mg))

    # ── Semanticist (Phase 3) ──────────────────────────────────────────
    if use_semantic:
        if not quiet:
            console.print("\n[magenta]🧠 Running Semanticist…[/magenta]")
        
        semanticist = Semanticist(repo_root=repo, output_dir=out_dir)
        mg = semanticist.run(mg)
        # Re-export with semantic metadata
        kg_module = KnowledgeGraph.from_module_graph(mg)
        if not summary_only:
            module_path = kg_module.export_json(out_dir / "module_graph.json")

    # ── Hydrologist ────────────────────────────────────────────────────
    if not quiet:
        console.print("\n[blue]🌊 Running Hydrologist…[/blue]")

    hydrologist = Hydrologist(repo_root=repo, sql_dialect=sql_dialect)
    lg: LineageGraph = hydrologist.run()
    kg_lineage = KnowledgeGraph.from_lineage_graph(lg)
    lineage_path = None
    if not summary_only:
        lineage_path = kg_lineage.export_json(out_dir / "lineage_graph.json")

    if not quiet:
        console.print(_lineage_summary_table(kg_lineage, lg))

    # ── Archivist (Phase 4) ────────────────────────────────────────────
    codebase_md_path = None
    onboarding_brief_path = None
    if run_archivist and not summary_only:
        if not quiet:
            console.print("\n[yellow]📁 Running Archivist…[/yellow]")
        artifacts = archivist.run(mg, lg)
        codebase_md_path = artifacts.get("codebase_md")
        onboarding_brief_path = artifacts.get("onboarding_brief")

    if not quiet:
        console.print(
            f"\n[bold green]✅ Done![/bold green]  "
            f"Outputs in [underline]{out_dir}[/underline]"
        )

    # ── Cleanup ────────────────────────────────────────────────────────
    if is_temp_repo and repo.exists():
        try:
            shutil.rmtree(repo)
        except Exception as e:
            log.warning(f"Could not delete temp repo {repo}: {e}")

    return {
        "module_graph_path": module_path,
        "lineage_graph_path": lineage_path,
        "codebase_md_path": codebase_md_path,
        "onboarding_brief_path": onboarding_brief_path,
        "output_dir": str(out_dir),
        "module_stats": kg_module.summary_stats(),
        "lineage_stats": kg_lineage.summary_stats(),
    }
