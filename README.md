# 🗺️ The Brownfield Cartographer

> **Google Maps for a messy, unfamiliar codebase.**

When you join a company or get dropped on a repo with no docs — 800k lines of code, engineers gone, docs outdated — this tool lets you understand the system in **minutes instead of days**.

---

## What it does

| Layer | Agent | Output |
|---|---|---|
| Module mapping | **Surveyor** | Who depends on whom? Hub modules, circular deps, dead code candidates |
| Data lineage | **Hydrologist** | SQL table flow + Python file I/O — source → transform → sink |

Both graphs are written as structured JSON to `.cartography/` — ready to feed into LLMs, dashboards, or your own tooling.

---

## Installation

```bash
git clone https://github.com/rafia-10/Brownfield-cartographer
cd Brownfield-cartographer
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

---

## Usage

### Scan a repo
```bash
cartographer scan /path/to/some/repo
```

Writes two files:
- `.cartography/module_graph.json` — file/module dependency graph
- `.cartography/lineage_graph.json` — data flow from source → transformations → outputs

### Print a summary without writing files
```bash
cartographer summary /path/to/some/repo
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--output / -o` | `.cartography` | Output directory |
| `--dialect / -d` | auto | SQL dialect (`bigquery`, `postgres`, `snowflake`, …) |
| `--verbose / -v` | off | Debug logging |

---

## Output format

### `module_graph.json`
```jsonc
{
  "metadata": { "node_count": 42, "edge_count": 130, "hub_modules": ["src.core", ...] },
  "nodes": [
    { "id": "src.agents.surveyor", "language": "python", "classes": [...], "functions": [...], "loc": 210 }
  ],
  "edges": [
    { "source": "src.cli", "target": "src.orchestrator", "kind": "import" }
  ]
}
```

### `lineage_graph.json`
```jsonc
{
  "metadata": { "source_count": 3, "sink_count": 2 },
  "nodes": [
    { "id": "raw_events", "kind": "table", "is_source": true },
    { "id": "report.csv",  "kind": "file",  "is_sink": true }
  ],
  "edges": [
    { "source": "raw_events", "target": "report.csv", "operation": "WRITE" }
  ]
}
```

---

## Project structure

```
src/
├─ cli.py                      # Typer CLI entry point
├─ orchestrator.py             # Coordinates agents, prints Rich tables
├─ models/nodes.py             # Pydantic v2 graph models
├─ analyzers/
│   ├─ tree_sitter_analyzer.py # Python AST parser (tree-sitter)
│   └─ sql_lineage.py          # SQL table extractor (sqlglot)
├─ agents/
│   ├─ surveyor.py             # Builds ModuleGraph
│   └─ hydrologist.py         # Builds LineageGraph
└─ graph/knowledge_graph.py   # NetworkX wrapper + JSON export
```

---

## Roadmap

- [ ] LLM-powered purpose detection per module
- [ ] Auto-generated `CODEBASE.md`
- [ ] Day-one onboarding brief
- [ ] Dead code candidate detection
- [ ] Semantic index injection for AI agents
