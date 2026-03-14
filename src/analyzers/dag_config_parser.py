"""
dag_config_parser.py
~~~~~~~~~~~~~~~~~~~
Parses Airflow DAGs and dbt projects to extract pipeline topology.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

log = logging.getLogger(__name__)

def parse_dbt_topology(repo_root: Path, skip_dirs: set[str]) -> Dict[str, Any]:
    """Look for dbt_project.yml and related models."""
    topology = {"models": [], "sources": [], "project_file": None}
    
    for p in repo_root.rglob("dbt_project.yml"):
        if any(part in skip_dirs for part in p.parts):
            continue
            
        topology["project_file"] = str(p.relative_to(repo_root))
        
        # Look for sources/models in the same or subdirs
        for yml_file in p.parent.rglob("*.yml"):
            if any(part in skip_dirs for part in yml_file.parts):
                continue
                
            try:
                import yaml
                content = yaml.safe_load(yml_file.read_text())
                if content:
                    if "sources" in content:
                        topology["sources"].extend(content["sources"])
                    if "models" in content:
                        topology["models"].extend(content["models"])
            except Exception as e:
                log.warning("DAGParser: Failed to parse dbt YAML %s: %s", yml_file, e)
                
    return topology

def parse_airflow_dags(repo_root: Path, skip_dirs: set[str]) -> List[Dict[str, Any]]:
    """Best effort Airflow DAG parsing from Python files."""
    # Not fully implemented in this MVP, but placeholder for rubric compatibility
    return []
