"""
archivist.py
~~~~~~~~~~~~
The Archivist agent produces the final deliverables (CODEBASE.md) and handles 
audit logging (cartography_trace.jsonl).

Responsibilities:
  1. Generate CODEBASE.md as a high-density reference for AI agents.
  2. Implement a Trace Logger for cartography_trace.jsonl.
  3. Incremental mode support (via git diff).
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Any, List, Optional

from src.models.nodes import ModuleGraph, TableNode, LineageGraph

log = logging.getLogger(__name__)

class Archivist:
    """
    Agent responsible for documenting and auditing the cartography run.
    """

    def __init__(self, repo_root: str | Path, output_dir: str | Path = ".cartography") -> None:
        self.repo_root = Path(repo_root).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.trace_path = self.output_dir / "cartography_trace.jsonl"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def log_trace(self, agent: str, action: str, evidence: Any, confidence: float = 1.0):
        """Append a trace record to cartography_trace.jsonl."""
        record = {
            "agent": agent,
            "action": action,
            "evidence": evidence,
            "confidence": confidence,
            "timestamp": logging.Formatter("%(asctime)s").format(logging.LogRecord("", 0, "", 0, "", None, None))
        }
        with open(self.trace_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def generate_CODEBASE_md(self, mg: ModuleGraph, lg: LineageGraph) -> Path:
        """
        Generates a CODEBASE.md file for AI agent injection.
        """
        md_content = f"# 🗺️ CODEBASE Context: {self.repo_root.name}\n\n"
        
        # 1. Architecture Overview
        md_content += "## 🏛️ Architecture Overview\n"
        md_content += mg.metadata.extra.get("day_one_report", "No report available.") + "\n\n"
        
        # 2. Critical Path (Top hubs)
        md_content += "## ⚡ Critical Path (Top 5 Hub Modules)\n"
        for hub in mg.metadata.hub_modules[:5]:
            md_content += f"- `{hub}`\n"
        md_content += "\n"
        
        # 3. Data Sources & Sinks
        md_content += "## 🌊 Data Sources & Sinks\n"
        sources = [n.name for n in lg.nodes if n.is_source]
        sinks = [n.name for n in lg.nodes if n.is_sink]
        md_content += f"**Sources:** {', '.join(sources[:10])}\n\n"
        md_content += f"**Sinks:** {', '.join(sinks[:10])}\n\n"
        
        # 4. Known Debt
        md_content += "## ⚠️ Known Debt & Risks\n"
        md_content += f"- Circular dependency groups: {mg.metadata.circular_dependency_count}\n"
        drift_count = sum(1 for n in mg.nodes if n.extra.get("doc_drift"))
        md_content += f"- Modules with Documentation Drift: {drift_count}\n\n"
        
        # 5. Domains
        md_content += "## 📂 Domain Map\n"
        domains = {}
        for n in mg.nodes:
            d = n.extra.get("domain", "Unknown")
            domains.setdefault(d, []).append(n.id)
            
        for domain, modules in domains.items():
            md_content += f"### {domain}\n"
            md_content += f"- {', '.join(modules[:5])} {'...' if len(modules) > 5 else ''}\n"
            
        out_path = self.output_dir / "CODEBASE.md"
        out_path.write_text(md_content)
        log.info(f"Archivist: Generated CODEBASE.md at {out_path}")
        return out_path

    def get_changed_files(self) -> List[Path]:
        """Uses git diff to find files changed since last commit (best effort)."""
        try:
            cmd = ["git", "ls-files", "--modified", "--others", "--exclude-standard"]
            output = subprocess.check_output(cmd, cwd=self.repo_root).decode("utf-8")
            return [self.repo_root / f for f in output.splitlines() if f.endswith(".py")]
        except Exception as e:
            log.warning(f"Archivist: Git diff failed, falling back to full scan. {e}")
            return []
            
    def run(self, mg: ModuleGraph, lg: LineageGraph) -> Path:
        """Main entry point to archive the run."""
        self.log_trace("Archivist", "Generate CODEBASE.md", {"repo": str(self.repo_root)})
        return self.generate_CODEBASE_md(mg, lg)
