"""
semanticist.py
~~~~~~~~~~~~~~
The Semanticist agent adds LLM-powered semantic understanding to the knowledge graph.

Responsibilities:
  1. Generate 2-3 sentence purpose statements for each module.
  2. Detect "Documentation Drift" between code purpose and existing docstrings.
  3. Cluster modules into business domains using embeddings + K-means.
  4. Answer "Day One" questions for codebase onboarding.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage
from sklearn.cluster import KMeans

from src.models.nodes import Language, ModuleGraph, ModuleNode
from src.utils.llm_budget import budget

log = logging.getLogger(__name__)

# Load environment variables from .env
load_dotenv()

class Semanticist:
    """
    Agent responsible for semantic analysis using LLMs.
    """

    def __init__(
        self,
        repo_root: str | Path,
        output_dir: str | Path = ".cartography",
        bulk_model: Optional[str] = None,
        synthesis_model: Optional[str] = None, 
        api_key: Optional[str] = None
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        
        if not self.api_key:
            log.error("GOOGLE_API_KEY or GEMINI_API_KEY not found in environment or provided.")
        else:
            log.info(f"Semanticist: LLM initialized with key starting: {self.api_key[:8]}...")
            
        env_model = os.getenv("CARTOGRAPHER_MODEL", "gemini-1.5-flash")
        self.bulk_llm = ChatGoogleGenerativeAI(model=bulk_model or env_model, google_api_key=self.api_key)
        self.synthesis_llm = ChatGoogleGenerativeAI(model=synthesis_model or env_model, google_api_key=self.api_key)
        # Use a more modern embedding model
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=self.api_key)

    def generate_purpose_statement(self, node: ModuleNode, content: str) -> tuple[str, bool]:
        """
        Generates a purpose statement and checks for documentation drift.
        Returns (purpose_statement, drift_detected, drift_evidence).
        """
        prompt = f"""Analyze the following code and provide a 2-3 sentence purpose statement 
that explains its business function (not technical implementation details).
Also, identify if the code's actual logic differs significantly from any existing documentation (Documentation Drift).

Code:
{content[:8000]}  # Truncate to stay within reason for bulk analysis

Output format:
Purpose: [statement]
Drift: [Yes/No]
Evidence: [If Drift is Yes, provide specific line numbers or 2-3 word snippets as evidence. Otherwise 'None'.]
"""
        try:
            tokens_in = budget.count_tokens(prompt)
            response = self.bulk_llm.invoke([HumanMessage(content=prompt)])
            response_text = response.content
            tokens_out = budget.count_tokens(response_text)
            
            budget.update("gemini-1.5-flash-latest", tokens_in, tokens_out)
            
            # Parse response
            purpose = ""
            drift = False
            evidence = None
            
            lines = response_text.split("\n")
            for i, line in enumerate(lines):
                if line.startswith("Purpose:"):
                    purpose = line.replace("Purpose:", "").strip()
                if line.startswith("Drift:"):
                    drift = "Yes" in line
                if line.startswith("Evidence:"):
                    evidence = line.replace("Evidence:", "").strip()
                    if evidence.lower() == "none":
                        evidence = None
                    
            return purpose, drift, evidence
        except Exception as e:
            log.error(f"Error generating purpose for {node.id}: {str(e)}. Using heuristic fallback.")
            
            # --- Heuristic Fallback: Extract first 3 non-empty lines ---
            lines = content.split("\n")
            non_empty = [(i+1, l.strip()) for i, l in enumerate(lines) if l.strip()]
            snippets = non_empty[:3]
            purpose = " ".join([s[1].lstrip("#").lstrip("/").lstrip("*").strip() for s in snippets])
            if len(purpose) > 150:
                purpose = purpose[:147] + "..."
            
            citation = f" (Lines {snippets[0][0]}-{snippets[-1][0]})" if snippets else ""
            return f"[Heuristic]{citation} {purpose}", False, None

    def cluster_into_domains(self, nodes: List[ModuleNode]) -> List[str]:
        """
        Clusters modules into domains using purpose statements.
        Updates the nodes' 'extra' dictionary with domain labels.
        """
        purposes = [n.extra.get("purpose", "") for n in nodes]
        if not any(purposes) or len(nodes) < 2:
            return ["Core"] * len(nodes)
            
        try:
            # Embeddings
            embeddings = self.embeddings.embed_documents(purposes)
            X = np.array(embeddings)
            
            # K-Means
            k = min(len(nodes), 8)
            kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto").fit(X)
            labels = kmeans.labels_
            
            # Label domains (best effort)
            domain_names = []
            for i in range(k):
                cluster_purposes = [purposes[j] for j in range(len(nodes)) if labels[j] == i]
                summary_prompt = f"Summarize these purpose statements into a single 1-2 word domain name: {' | '.join(cluster_purposes[:5])}"
                try:
                    resp = self.synthesis_llm.invoke([HumanMessage(content=summary_prompt)])
                    domain_names.append(resp.content.strip().replace('"', ''))
                except Exception as e:
                    log.warning(f"Semanticist: LLM failed to label domain {i}: {e}. Using fallback.")
                    domain_names.append(f"Domain {i}")
                
            final_labels = [domain_names[label] for label in labels]
            for node, label in zip(nodes, final_labels):
                node.extra["domain"] = label
                
            return final_labels
        except Exception as e:
            log.error(f"Error clustering domains: {e}")
            return ["Unknown"] * len(nodes)

    def answer_day_one_questions(self, mg: ModuleGraph) -> str:
        """
        Provides a synthesis of the codebase for a new developer.
        """
        # Prepare a detailed summary of the semantic findings
        summary = f"Codebase has {mg.metadata.node_count} modules and {mg.metadata.edge_count} dependencies.\n"
        
        # 1. Hub context
        summary += "\n### Core Hubs (High Centrality):\n"
        for mod_id in mg.metadata.hub_modules[:5]:
            node = next((n for n in mg.nodes if n.id == mod_id), None)
            if node:
                purpose = node.extra.get("purpose", "No purpose found.")
                summary += f"- `{mod_id}`: {purpose}\n"
        
        # 2. Domain context
        domains = {}
        for n in mg.nodes:
            d = n.extra.get("domain", "Unknown")
            domains.setdefault(d, []).append(n.id)
        
        summary += "\n### Discovered Business Domains:\n"
        for domain, mods in domains.items():
            summary += f"- **{domain}**: Includes {', '.join(mods[:3])}...\n"
            
        # 3. Drift context (Citations!)
        drifts = [n for n in mg.nodes if n.extra.get("doc_drift")]
        if drifts:
            summary += "\n### Documentation Drift (High Risk):\n"
            for n in drifts[:5]:
                evidence = n.extra.get("drift_evidence", "No snippet.")
                summary += f"- `{n.id}` (Path: `{n.path}`): Evidence: \"{evidence}\"\n"

        prompt = f"""You are an Expert Software Architect performing a 'Day One' audit for a new FDE.
Based on the following semantic and structural analysis, answer the 'Five FDE Questions' with high precision.

CRITICAL: 
- Provide explicit file citations/line snippets for your claims.
- If you find documentation drift, call it out as a TRAP.
- Be technical but business-aligned.

Target Analysis Context:
{summary}

Analysis Output format:
# Five Day-One Questions
[Your synthesis...]
"""
        try:
            response = self.synthesis_llm.invoke([HumanMessage(content=prompt)])
            return response.content
        except Exception as e:
            log.error(f"Error answering day one questions: {e}. Falling back to template-based synthesis.")
            
            # --- Robust Fallback Report ---
            fallback = "# Five Day-One Questions (Safe Mode)\n\n"
            
            fallback += "### 1. What is the business value of this system?\n"
            fallback += f"The system consists of {mg.metadata.node_count} modules focused on the following key hubs:\n"
            for mod_id in mg.metadata.hub_modules[:3]:
                node = next((n for n in mg.nodes if n.id == mod_id), None)
                if node:
                    fallback += f"- `{mod_id}`: {node.extra.get('purpose', 'Core architectural component')}\n"
            fallback += "\n"
            
            fallback += "### 2. Where is the core domain logic located?\n"
            fallback += f"Logic is distributed across {len(domains)} primary domains:\n"
            for dom, mods in list(domains.items())[:3]:
                fallback += f"- **{dom}**: {', '.join(mods[:2])}...\n"
            fallback += "\n"
            
            fallback += "### 3. What are the highest-risk modules (traps)?\n"
            drift_nodes = [n for n in mg.nodes if n.extra.get("doc_drift")]
            if drift_nodes:
                fallback += "⚠️ **TRAP: Documentation Decay** - The following modules have logic differing from their documentation:\n"
                for n in drift_nodes[:3]:
                    fallback += f"- `{n.id}` (See: `{n.path}`): \"{n.extra.get('drift_evidence', 'Logic differs from comments')}\"\n"
            else:
                fallback += "No major documentation traps detected in this scan.\n"
            fallback += "\n"
            
            fallback += "### 4. How does data move through the system?\n"
            fallback += "Data movement is governed by the structural dependencies between modules. Use `cartographer query` to trace specific lineage paths.\n\n"
            
            fallback += "### 5. What is the state of the documentation?\n"
            if drift_nodes:
                fallback += f"Moderate documentation decay. {len(drift_nodes)} modules flagged for drift. Rely on the actual code logic in the cited files.\n"
            else:
                fallback += "Documentation appears consistent with the current code state.\n"
                
            return fallback

    def _get_file_hash(self, path: Path) -> str:
        """Simple content hash to detect changes."""
        import hashlib
        try:
            return hashlib.md5(path.read_bytes()).hexdigest()
        except:
            return ""

    def run(self, mg: ModuleGraph) -> ModuleGraph:
        """
        Main entry point to process a ModuleGraph with semantic analysis.
        """
        log.info("Semanticist: Running LLM-powered analysis...")
        
        index_path = self.output_dir / "semantic_index.json"
        cache = {}
        if index_path.exists():
            try:
                import json
                cache = json.loads(index_path.read_text())
            except Exception as e:
                log.warning(f"Semanticist: Failed to load index cache: {e}")

        new_index = {}
        for node in mg.nodes:
            if node.language not in (Language.PYTHON, Language.SQL):
                continue
            
            file_path = self.repo_root / node.path
            if file_path.exists():
                file_hash = self._get_file_hash(file_path)
                
                # Check cache
                cached = cache.get(node.id)
                if cached and cached.get("hash") == file_hash:
                    node.extra["purpose"] = cached.get("purpose")
                    node.extra["doc_drift"] = cached.get("doc_drift")
                    new_index[node.id] = cached
                    continue

                content = file_path.read_text(errors="replace")
                purpose, drift, evidence = self.generate_purpose_statement(node, content)
                node.extra["purpose"] = purpose
                node.extra["doc_drift"] = drift
                node.extra["drift_evidence"] = evidence
                
                new_index[node.id] = {
                    "id": node.id,
                    "path": node.path,
                    "hash": file_hash,
                    "purpose": purpose,
                    "doc_drift": drift,
                    "drift_evidence": evidence
                }
        
        # Save updated index
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            index_path.write_text(json.dumps(new_index, indent=2))
            
            # Also save embeddings separately for Vector Search in Navigator
            embedding_data = {}
            for node in mg.nodes:
                if "purpose" in node.extra:
                    emb = self.embeddings.embed_query(node.extra["purpose"])
                    embedding_data[node.id] = list(emb)
            
            emb_path = self.output_dir / "semantic_embeddings.json"
            emb_path.write_text(json.dumps(embedding_data))
            log.info(f"Semanticist: Saved {len(embedding_data)} embeddings to {emb_path}")
        except Exception as e:
            log.warning(f"Semanticist: Failed to save index: {e}")

        self.cluster_into_domains(mg.nodes)
        
        # Add day-one summary to metadata
        day_one_report = self.answer_day_one_questions(mg)
        mg.metadata.extra["day_one_report"] = day_one_report
        
        return mg
