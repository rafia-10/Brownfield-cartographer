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
    repo_path: Path = typer.Argument(
        ...,
        help="Root directory of the repository to scan.",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
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
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
) -> None:
    """
    Scan a repository and write module_graph.json + lineage_graph.json
    to the output directory.
    """
    _configure_logging(verbose)

    from src.orchestrator import run  # local import to keep startup fast

    try:
        result = run(
            repo_path=repo_path,
            output_dir=output_dir,
            sql_dialect=dialect,
            quiet=False,
        )
    except Exception as exc:  # noqa: BLE001
        console.print(f"[bold red]Error:[/bold red] {exc}")
        if verbose:
            raise
        raise typer.Exit(code=1) from exc

    raise typer.Exit(code=0)


@app.command()
def summary(
    repo_path: Path = typer.Argument(
        ...,
        help="Root directory of the repository to inspect.",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
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

    from src.agents.surveyor import Surveyor
    from src.agents.hydrologist import Hydrologist
    from src.graph.knowledge_graph import KnowledgeGraph
    from rich.columns import Columns
    from rich.panel import Panel

    try:
        surveyor = Surveyor(repo_root=repo_path)
        mg = surveyor.run()
        kg_m = KnowledgeGraph.from_module_graph(mg)

        hydrologist = Hydrologist(repo_root=repo_path, sql_dialect=dialect)
        lg = hydrologist.run()
        kg_l = KnowledgeGraph.from_lineage_graph(lg)

    except Exception as exc:  # noqa: BLE001
        console.print(f"[bold red]Error:[/bold red] {exc}")
        if verbose:
            raise
        raise typer.Exit(code=1) from exc

    m_stats = kg_m.summary_stats()
    l_stats = kg_l.summary_stats()

    console.print(Panel.fit(
        f"[bold green]Brownfield Cartographer — Summary[/bold green]\n[dim]{repo_path}[/dim]",
        border_style="green",
    ))

    console.print("\n[bold]Module Graph[/bold]")
    console.print(f"  Modules  : {m_stats['nodes']}")
    console.print(f"  Edges    : {m_stats['edges']}")
    console.print(f"  Cycles   : {m_stats['cycles']}")
    console.print(f"  Hubs     : {', '.join(m_stats['hub_nodes']) or '—'}")

    console.print("\n[bold]Lineage Graph[/bold]")
    console.print(f"  Data nodes : {l_stats['nodes']}")
    console.print(f"  Edges      : {l_stats['edges']}")
    console.print(f"  Sources    : {lg.metadata.source_count}")
    console.print(f"  Sinks      : {lg.metadata.sink_count}")
    console.print(f"  Hubs       : {', '.join(l_stats['hub_nodes']) or '—'}")

    raise typer.Exit(code=0)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    app()
