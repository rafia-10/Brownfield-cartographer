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
        bulk_model: str = "gemini-1.5-flash",
        synthesis_model: str = "gemini-1.5-flash", # Using flash for both as default if pro not configured
        api_key: Optional[str] = None
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        
        if not self.api_key:
            log.error("GOOGLE_API_KEY or GEMINI_API_KEY not found in environment or provided.")
        else:
            log.info(f"Semanticist: LLM initialized with key starting: {self.api_key[:8]}...")
            
        self.bulk_llm = ChatGoogleGenerativeAI(model=bulk_model, google_api_key=self.api_key)
        self.synthesis_llm = ChatGoogleGenerativeAI(model=synthesis_model, google_api_key=self.api_key)
        # Use a more modern embedding model
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=self.api_key)

    def generate_purpose_statement(self, node: ModuleNode, content: str) -> tuple[str, bool]:
        """
        Generates a purpose statement and checks for documentation drift.
        Returns (purpose_statement, drift_detected).
        """
        prompt = f"""Analyze the following code and provide a 2-3 sentence purpose statement 
that explains its business function (not technical implementation details).
Also, identify if the code's actual logic differs significantly from any existing documentation (Documentation Drift).

Code:
{content[:8000]}  # Truncate to stay within reason for bulk analysis

Output format:
Purpose: [statement]
Drift: [Yes/No] - [brief explanation if Yes]
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
            for line in response_text.split("\n"):
                if line.startswith("Purpose:"):
                    purpose = line.replace("Purpose:", "").strip()
                if line.startswith("Drift:"):
                    drift = "Yes" in line
                    
            return purpose, drift
        except Exception as e:
            log.error(f"Error generating purpose for {node.id}: {str(e)}")
            return "Analysis failed.", False

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
                resp = self.synthesis_llm.invoke([HumanMessage(content=summary_prompt)])
                domain_names.append(resp.content.strip().replace('"', ''))
                
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
        # Prepare a high-level summary of the graph
        summary = f"Codebase has {mg.metadata.node_count} modules and {mg.metadata.edge_count} dependencies.\n"
        summary += f"Hub modules: {', '.join(mg.metadata.hub_modules)}\n"
        
        prompt = f"""You are an Expert Software Architect. Based on this codebase summary, 
answer the 'Five FDE Questions' to help a new engineer on their first day:
1. What is the business value of this system?
2. Where is the core domain logic located?
3. What are the highest-risk modules (traps)?
4. How does data move through the system (at a high level)?
5. What is the state of the documentation?

Summary:
{summary}

Analysis:
"""
        try:
            response = self.synthesis_llm.invoke([HumanMessage(content=prompt)])
            return response.content
        except Exception as e:
            log.error(f"Error answering day one questions: {e}")
            return "Could not synthesize report."

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
                purpose, drift = self.generate_purpose_statement(node, content)
                node.extra["purpose"] = purpose
                node.extra["doc_drift"] = drift
                
                new_index[node.id] = {
                    "id": node.id,
                    "path": node.path,
                    "hash": file_hash,
                    "purpose": purpose,
                    "doc_drift": drift
                }
        
        # Save updated index
        try:
            import json
            self.output_dir.mkdir(parents=True, exist_ok=True)
            index_path.write_text(json.dumps(new_index, indent=2))
        except Exception as e:
            log.warning(f"Semanticist: Failed to save index: {e}")

        self.cluster_into_domains(mg.nodes)
        
        # Add day-one summary to metadata
        day_one_report = self.answer_day_one_questions(mg)
        mg.metadata.extra["day_one_report"] = day_one_report
        
        return mg
