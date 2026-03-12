"""
cli.py
~~~~~~
Typer-powered CLI for The Brownfield Cartographer.

Commands
--------
  cartographer scan <repo_path>    Full scan → writes .cartography/ JSON
  cartographer summary <repo_path> Print stats without writing files
"""

from __future__ import annotations

import logging
import sys
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
def scan(
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
        False,
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
    Scan a repository and write module_graph.json + lineage_graph.json
    to the output directory.
    """
    _configure_logging(verbose)

    from src.orchestrator import run  # local import to keep startup fast

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
def chat(
    repo_path: str = typer.Argument(
        ...,
        help="Root directory of the repository.",
    ),
    output_dir: str = typer.Option(
        ".cartography",
        "--output", "-o",
        help="Path where existing graphs live.",
    ),
):
    """
    Start a conversational session with the Navigator to explore the codebase.
    """
    from src.agents.navigator import Navigator
    from src.models.nodes import ModuleGraph, LineageGraph
    import json
    
    out_dir = repo_path / output_dir
    mg_path = out_dir / "module_graph.json"
    lg_path = out_dir / "lineage_graph.json"
    
    if not mg_path.exists() or not lg_path.exists():
        console.print("[bold red]Error:[/bold red] Graphs not found. Run 'cartographer scan' first.")
        raise typer.Exit(code=1)
        
    with open(mg_path) as f:
        mg = ModuleGraph.model_validate(json.load(f))
    with open(lg_path) as f:
        lg = LineageGraph.model_validate(json.load(f))
        
    navigator = Navigator(repo_root=repo_path, module_graph=mg, lineage_graph=lg)
    
    console.print(f"\n[bold green]Welcome to the Navigator session for {repo_path.name}[/bold green]")
    console.print("[dim]Type 'exit' to quit.[/dim]\n")
    
    while True:
        question = typer.prompt("Navigator")
        if question.lower() in ("exit", "quit"):
            break
        
        with console.status("[blue]Thinking...[/blue]"):
            answer = navigator.ask(question)
        
        console.print(f"\n[bold cyan]Navigator:[/bold cyan]\n{answer}\n")



# Entry point


if __name__ == "__main__":
    import os
    load_dotenv()
    # Alias GEMINI_API_KEY to GOOGLE_API_KEY if present
    if "GEMINI_API_KEY" in os.environ and "GOOGLE_API_KEY" not in os.environ:
        os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]
    app()
