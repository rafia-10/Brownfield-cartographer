"""
navigator.py
~~~~~~~~~~~~
The Navigator agent uses LangGraph to provide a conversational interface for 
exploring the codebase, backed by evidence from the Cartographer's analysis.
"""

from __future__ import annotations

import os
import json
import logging
from pathlib import Path
from typing import Annotated, Any, Dict, List, Sequence, TypedDict, Union

import numpy as np
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from src.agents.factory import LLMFactory
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from src.models.nodes import ModuleGraph, LineageGraph
from src.graph.knowledge_graph import KnowledgeGraph

log = logging.getLogger(__name__)

# --- State Definition ---

class NavigatorState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], "The messages in the conversation"]
    repo_path: str
    module_graph: Dict[str, Any]
    lineage_graph: Dict[str, Any]

# --- Tools ---

def create_navigator_tools(module_graph: Dict[str, Any], lineage_graph: Dict[str, Any], repo_root: Path, embeddings: Dict[str, List[float]], llm_embeddings: Any):
    
    @tool
    def vector_search(query: str) -> str:
        """Find modules conceptually related to a natural language query using semantic embeddings."""
        if not embeddings:
            return "Vector search index is empty. Run 'analyze' first."
        
        try:
            query_emb = np.array(llm_embeddings.embed_query(query))
            scores = []
            
            for mod_id, mod_emb in embeddings.items():
                mod_emb_vec = np.array(mod_emb)
                similarity = np.dot(query_emb, mod_emb_vec) / (np.linalg.norm(query_emb) * np.linalg.norm(mod_emb_vec))
                scores.append((mod_id, similarity))
            
            # Sort by similarity
            scores.sort(key=lambda x: x[1], reverse=True)
            top_3 = scores[:3]
            
            results = []
            for mod_id, score in top_3:
                results.append(f"• {mod_id} (Similarity: {score:.2f})")
            
            return "Top concept matches:\n" + "\n".join(results)
        except Exception as e:
            return f"Error performing vector search: {str(e)}"
    
    @tool
    def search_modules(query: str) -> str:
        """Search for modules by name, class, or function."""
        results = []
        for node in module_graph.get("nodes", []):
            if query.lower() in node["id"].lower() or \
               any(query.lower() in c.lower() for c in node.get("classes", [])) or \
               any(query.lower() in f.lower() for f in node.get("functions", [])):
                results.append(f"Module: {node['id']} (Path: {node['path']})")
        return "\n".join(results[:10]) if results else "No modules found."

    @tool
    def get_dependencies(module_id: str) -> str:
        """Get the dependencies (imports) for a specific module."""
        deps = [e["target"] for e in module_graph.get("edges", []) if e["source"] == module_id]
        return f"Module '{module_id}' depends on: {', '.join(deps)}" if deps else f"No dependencies found for '{module_id}'."

    @tool
    def get_lineage(table_or_file: str) -> str:
        """Trace data flow for a specific table or file."""
        upstream = [e["source"] for e in lineage_graph.get("edges", []) if e["target"].lower() == table_or_file.lower()]
        downstream = [e["target"] for e in lineage_graph.get("edges", []) if e["source"].lower() == table_or_file.lower()]
        return f"Lineage for '{table_or_file}':\n  Sources: {', '.join(upstream)}\n  Sinks: {', '.join(downstream)}"

    @tool
    def explain_module(module_id: str) -> str:
        """Explain the business purpose and documentation state of a module."""
        for node in module_graph.get("nodes", []):
            if node["id"] == module_id:
                purpose = node.get("extra", {}).get("purpose", "No purpose statement available.")
                drift = node.get("extra", {}).get("doc_drift", False)
                drift_msg = "⚠️ Documentation drift detected!" if drift else "✅ Documentation is up to date."
                return f"Module: {module_id}\nPurpose: {purpose}\nStatus: {drift_msg}"
        return f"Module '{module_id}' not found."

    @tool
    def get_blast_radius(module_id: str, depth: int = 2) -> str:
        """Calculate the impact radius (downstream dependencies) if this module changes."""
        mg = ModuleGraph.model_validate(module_graph)
        kg = KnowledgeGraph.from_module_graph(mg)
        affected = kg.blast_radius(module_id, depth=depth)
        
        if not affected:
            return f"No downstream impact detected for '{module_id}' at depth {depth}."
        
        return f"Changing '{module_id}' may impact (depth={depth}):\n- " + "\n- ".join(list(affected)[:20])

    @tool
    def read_file_segment(path: str, start_line: int, end_line: int) -> str:
        """Read a specific segment of a file (e.g. for citing sources)."""
        p = repo_root / path
        if not p.exists():
            return f"File '{path}' not found."
        try:
            lines = p.read_text().splitlines()
            segment = lines[max(0, start_line-1):min(len(lines), end_line)]
            return "\n".join(segment)
        except Exception as e:
            return f"Error reading file: {str(e)}"

    tools = [search_modules, get_dependencies, get_lineage, read_file_segment, explain_module, get_blast_radius]
    if llm_embeddings:
        tools.append(vector_search)
    return tools

# --- Agent Build ---

class Navigator:
    def __init__(self, repo_root: str | Path, module_graph: ModuleGraph, lineage_graph: LineageGraph, model_name: Optional[str] = None):
        load_dotenv()
        self.repo_root = Path(repo_root).resolve()
        self.mg_dict = module_graph.model_dump(mode="json")
        self.lg_dict = lineage_graph.model_dump(mode="json")
        
        self.llm_embeddings = LLMFactory.create_embeddings()
        
        # Load embeddings if they exist
        self.embeddings = {}
        emb_path = self.repo_root / ".cartography" / "semantic_embeddings.json"
        if emb_path.exists():
            try:
                self.embeddings = json.loads(emb_path.read_text())
            except Exception as e:
                log.warning(f"Navigator: Failed to load embeddings: {e}")

        self.tools = create_navigator_tools(self.mg_dict, self.lg_dict, self.repo_root, self.embeddings, self.llm_embeddings)
        self.llm = LLMFactory.create_llm(model_name=model_name).bind_tools(self.tools)
        
        # Build Graph
        workflow = StateGraph(NavigatorState)
        
        def call_model(state: NavigatorState):
            messages = state['messages']
            try:
                response = self.llm.invoke(messages)
            except Exception as e:
                log.error(f"Navigator: Model call failed: {e}. Check API keys and provider settings.")
                raise e
            return {"messages": [response]}

        workflow.add_node("agent", call_model)
        workflow.add_node("tools", ToolNode(self.tools))
        
        workflow.set_entry_point("agent")
        
        def should_continue(state: NavigatorState):
            last_message = state['messages'][-1]
            if last_message.tool_calls:
                return "tools"
            return END

        workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
        workflow.add_edge("tools", "agent")
        
        self.app = workflow.compile()

    def ask(self, question: str) -> str:
        tool_names = [t.name for t in self.tools]
        system_content = f"""You are the Navigator, a senior code architect agent.
Your goal is to help developers explore the codebase using the tools provided: {', '.join(tool_names)}.
Always aim for 100% accuracy and provide evidence for your claims.

CRITICAL: 
- Use the available tools to answer questions.
- If the user's query is conceptual, use 'search_modules' (or 'vector_search' if available) to find relevant code.
- Always use 'read_file_segment' to cite specific code lines when explaining logic.
- If you find Documentation Drift, explicitly mention the Drift Evidence.
- Cite files and line numbers for every significant claim.
"""
        # Add tool-calling nudge for Groq/Llama
        system_content += "\nIMPORTANT: You MUST invoke tools using the standard JSON tool-calling format. Do not use custom XML tags or <function> tags."
        
        system_msg = SystemMessage(content=system_content)
        
        state = {
            "messages": [system_msg, HumanMessage(content=question)],
            "repo_path": str(self.repo_root),
            "module_graph": self.mg_dict,
            "lineage_graph": self.lg_dict
        }
        final_state = self.app.invoke(state)
        return final_state["messages"][-1].content
