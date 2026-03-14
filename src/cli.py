"""
cli.py
~~~~~~
Typer-powered CLI for The Brownfield Cartographer.

Commands
--------
  cartographer analyze <repo_path>  Full scan → writes .cartography/ JSON
  cartographer query <repo_path>    Start a conversational session
  cartographer summary <repo_path>  Print stats without writing files
  cartographer blast-radius <repo> <module> Calculate impact sphere
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

import typer
from rich.console import Console
from rich.logging import RichHandler

app = typer.Typer(
    name="cartographer",
    help="🗺  The Brownfield Cartographer — map an unfamiliar codebase in minutes.",
    add_completion=False,
    no_args_is_help=True,
)

console = Console()


# ---------------------------------------------------------------------------
# Shared logging setup
# ---------------------------------------------------------------------------


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True, markup=True)],
    )


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command()
def analyze(
    repo_path: str = typer.Argument(
        ...,
        help="Root directory or GitHub URL of the repository to scan.",
    ),
    output_dir: str = typer.Option(
        ".cartography",
        "--output", "-o",
        help="Directory for JSON output files (relative to repo or absolute).",
    ),
    dialect: Optional[str] = typer.Option(
        None,
        "--dialect", "-d",
        help="SQL dialect for sqlglot (e.g. bigquery, postgres, snowflake). Auto-detected if omitted.",
    ),
    semantic: bool = typer.Option(
        False,
        "--semantic", "-s",
        help="Enable LLM-powered semantic analysis (purpose detection, drift, domains).",
    ),
    archivist: bool = typer.Option(
        True,
        "--archivist", "-a",
        help="Generate CODEBASE.md and audit trace log.",
    ),
    incremental: bool = typer.Option(
        False,
        "--incremental", "-i",
        help="Re-analyze only modified files (git diff).",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
) -> None:
    """
    Scan a repository and write module_graph.json + lineage_graph.json.
    Reports execution time for benchmarking.
    """
    _configure_logging(verbose)

    from src.orchestrator import run  # local import to keep startup fast

    start_time = time.time()
    try:
        with console.status("[bold blue]Mapping the codebase...[/bold blue]", spinner="dots"):
            result = run(
                repo_path=repo_path,
                output_dir=output_dir,
                sql_dialect=dialect,
                use_semantic=semantic,
                run_archivist=archivist,
                incremental=incremental,
                quiet=False,
            )
        elapsed = time.time() - start_time
        console.print(f"\n[bold green]Analysis Complete![/bold green] ✨")
        console.print(f"⏱  Time elapsed: [bold cyan]{elapsed:.2f}s[/bold cyan]")
        console.print(f"📂 Output directory: [blue]{result.get('output_dir') or output_dir}[/blue]")
        
    except Exception as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        if verbose:
            raise
        raise typer.Exit(code=1) from exc

    raise typer.Exit(code=0)


@app.command()
def summary(
    repo_path: str = typer.Argument(
        ...,
        help="Root directory or GitHub URL of the repository to inspect.",
    ),
    dialect: Optional[str] = typer.Option(
        None,
        "--dialect", "-d",
        help="SQL dialect for sqlglot.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
) -> None:
    """
    Print a rich summary of the module and lineage graphs without writing
    any output files.
    """
    _configure_logging(verbose)

    from src.orchestrator import run

    try:
        with console.status("[bold blue]Mapping the codebase...[/bold blue]", spinner="dots"):
            run(
                repo_path=repo_path,
                sql_dialect=dialect,
                quiet=False,
                summary_only=True,
            )
    except Exception as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        if verbose:
            raise
        raise typer.Exit(code=1) from exc

    raise typer.Exit(code=0)


@app.command()
def query(
    repo_path: str = typer.Argument(
        ...,
        help="Root directory of the repository.",
    ),
    output_dir: str = typer.Option(
        ".cartography",
        "--output", "-o",
        help="Path where existing graphs live (relative to repo).",
    ),
):
    """
    Start a conversational session with the Navigator to explore the codebase.
    Supports lineage queries with file:line citations.
    """
    from src.agents.navigator import Navigator
    from src.models.nodes import ModuleGraph, LineageGraph
    import json
    
    # Handle local path resolution
    if str(repo_path).startswith(("http://", "https://", "git@")):
        repo_dir = Path.cwd()
    else:
        repo_dir = Path(repo_path).resolve()
        
    out_dir = repo_dir / output_dir
    mg_path = out_dir / "module_graph.json"
    lg_path = out_dir / "lineage_graph.json"
    
    if not mg_path.exists() or not lg_path.exists():
        if str(repo_path).startswith(("http://", "https://", "git@")):
            console.print(f"[bold red]Error:[/bold red] Graphs not found in local {output_dir}. Run 'cartographer analyze' for this URL first.")
        else:
            console.print(f"[bold red]Error:[/bold red] Graphs not found at {out_dir}. Run 'cartographer analyze' first.")
        raise typer.Exit(code=1)
        
    with open(mg_path) as f:
        mg = ModuleGraph.model_validate(json.load(f))
    with open(lg_path) as f:
        lg = LineageGraph.model_validate(json.load(f))
        
    navigator = Navigator(repo_root=repo_dir, module_graph=mg, lineage_graph=lg)
    
    console.print(f"\n[bold green]Welcome to the Navigator session for {repo_dir.name}[/bold green]")
    console.print("[dim]Type 'exit' to quit. Ask about lineage, purpose, or citations.[/dim]\n")
    
    while True:
        question = typer.prompt("Navigator")
        if question.lower() in ("exit", "quit"):
            break
        
        with console.status("[blue]Thinking...[/blue]"):
            answer = navigator.ask(question)
        
        console.print(f"\n[bold cyan]Navigator:[/bold cyan]\n{answer}\n")


@app.command()
def vector_search(
    repo_path: str = typer.Argument(..., help="Path or URL to the repository."),
    query: str = typer.Argument(..., help="Conceptual query (e.g. 'payment processing')."),
    output_dir: str = typer.Option(".cartography", "--output", "-o"),
):
    """
    Conceptual search using semantic embeddings. Use after 'analyze -s'.
    """
    from src.graph.knowledge_graph import KnowledgeGraph
    from src.models.nodes import ModuleGraph, LineageGraph
    from src.agents.navigator import Navigator
    import json
    
    if str(repo_path).startswith(("http://", "https://", "git@")):
        repo_dir = Path.cwd()
    else:
        repo_dir = Path(repo_path).resolve()

    out_dir = repo_dir / output_dir
    mg_path = out_dir / "module_graph.json"
    lg_path = out_dir / "lineage_graph.json"
    
    if not mg_path.exists():
        console.print(f"[bold red]Error:[/bold red] Analysis results not found at {out_dir}. Run 'analyze -s' first.")
        raise typer.Exit(code=1)
        
    with open(mg_path) as f:
        mg = ModuleGraph.model_validate(json.load(f))
    with open(lg_path) as f:
        lg = LineageGraph.model_validate(json.load(f))
        
    navigator = Navigator(repo_root=repo_dir, module_graph=mg, lineage_graph=lg)
    results = navigator.ask(f"Use vector_search to find: {query}")
    console.print(Panel(results, title=f"Concept Search: {query}", border_style="magenta"))

@app.command()
def blast_radius(
    repo_path: str = typer.Argument(..., help="Path or URL to the repository."),
    module_id: str = typer.Argument(..., help="Module ID to analyze."),
    depth: int = typer.Option(2, "--depth", "-d", help="Traversal depth."),
    output_dir: str = typer.Option(".cartography", "--output", "-o"),
):
    """
    Standalone command to calculate the impact sphere of a module change.
    """
    from src.graph.knowledge_graph import KnowledgeGraph
    from src.models.nodes import ModuleGraph, LineageGraph
    import json
    
    if str(repo_path).startswith(("http://", "https://", "git@")):
        repo_dir = Path.cwd()
    else:
        repo_dir = Path(repo_path).resolve()

    out_dir = repo_dir / output_dir
    mg_path = out_dir / "module_graph.json"
    
    if not mg_path.exists():
        console.print(f"[bold red]Error:[/bold red] module_graph.json not found at {out_dir}.")
        raise typer.Exit(code=1)
        
    with open(mg_path) as f:
        mg = ModuleGraph.model_validate(json.load(f))
    
    # Unified Blast Radius Path
    lg_path = out_dir / "lineage_graph.json"
    lg = None
    if lg_path.exists():
        with open(lg_path) as f:
            lg = LineageGraph.model_validate(json.load(f))
            
    kg_mg = KnowledgeGraph.from_module_graph(mg)
    affected = kg_mg.blast_radius(module_id, depth=depth)
    
    # --- SQL Bridge Logic ---
    if lg and module_id in kg_mg.graph:
        module_path = kg_mg.graph.nodes[module_id].get("path")
        if module_path:
            # Find matching data nodes in lineage
            matching_data_nodes = [n.id for n in lg.nodes if n.source_file == module_path]
            if matching_data_nodes:
                kg_lg = KnowledgeGraph.from_lineage_graph(lg)
                for data_nid in matching_data_nodes:
                    downstream_data = kg_lg.blast_radius(data_nid, depth=depth)
                    # Map back to modules
                    for d_nid in downstream_data:
                        d_node = next((n for n in lg.nodes if n.id == d_nid), None)
                        if d_node and d_node.source_file:
                            # Convert path back to module id
                            # (Heuristic: search MG for node with this path)
                            affected_mod = next((n.id for n in mg.nodes if n.path == d_node.source_file), None)
                            if affected_mod:
                                affected.add(affected_mod)
    
    console.print(f"\n[bold yellow]Unified Blast Radius for '{module_id}' (depth={depth}):[/bold yellow]")
    if not affected:
        console.print("No downstream impact detected.")
    else:
        for node_id in affected:
            console.print(f"- {node_id}")
    console.print("")



# Entry point


if __name__ == "__main__":
    import os
    load_dotenv()
    # Alias GEMINI_API_KEY to GOOGLE_API_KEY if present
    if "GEMINI_API_KEY" in os.environ and "GOOGLE_API_KEY" not in os.environ:
        os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]
    app()
