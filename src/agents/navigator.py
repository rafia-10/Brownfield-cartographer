"""
navigator.py
~~~~~~~~~~~~
The Navigator agent uses LangGraph to provide a conversational interface for 
exploring the codebase, backed by evidence from the Cartographer's analysis.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated, Any, Dict, List, Sequence, TypedDict, Union

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from src.models.nodes import ModuleGraph, LineageGraph

log = logging.getLogger(__name__)

# --- State Definition ---

class NavigatorState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], "The messages in the conversation"]
    repo_path: str
    module_graph: Dict[str, Any]
    lineage_graph: Dict[str, Any]

# --- Tools ---

def create_navigator_tools(module_graph: Dict[str, Any], lineage_graph: Dict[str, Any], repo_root: Path):
    
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
    def read_file_segment(path: str, start_line: int, end_line: int) -> str:
        """Read a specific line range from a file for evidence citation."""
        full_path = repo_root / path
        if not full_path.exists():
            return "File not found."
        try:
            lines = full_path.read_text(errors="replace").splitlines()
            segment = lines[max(0, start_line-1): min(len(lines), end_line)]
            return "\n".join(segment)
        except Exception as e:
            return f"Error reading file: {e}"

    return [search_modules, get_dependencies, get_lineage, read_file_segment]

# --- Agent Build ---

class Navigator:
    def __init__(self, repo_root: str | Path, module_graph: ModuleGraph, lineage_graph: LineageGraph):
        self.repo_root = Path(repo_root).resolve()
        self.mg_dict = module_graph.model_dump(mode="json")
        self.lg_dict = lineage_graph.model_dump(mode="json")
        
        self.tools = create_navigator_tools(self.mg_dict, self.lg_dict, self.repo_root)
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash").bind_tools(self.tools)
        
        # Build Graph
        workflow = StateGraph(NavigatorState)
        
        def call_model(state: NavigatorState):
            messages = state['messages']
            response = self.llm.invoke(messages)
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
        state = {
            "messages": [HumanMessage(content=question)],
            "repo_path": str(self.repo_root),
            "module_graph": self.mg_dict,
            "lineage_graph": self.lg_dict
        }
        final_state = self.app.invoke(state)
        return final_state["messages"][-1].content
