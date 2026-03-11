"""
universal_analyzer.py
~~~~~~~~~~~~~~~~~~~~~
Routs file analysis based on extension to specialized tree-sitter parsers.
Supports: Python, SQL, YAML.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tree_sitter import Language, Parser, Node
import tree_sitter_python
import tree_sitter_sql
import tree_sitter_yaml

# Languages
PY_LANG = Language(tree_sitter_python.language())
SQL_LANG = Language(tree_sitter_sql.language())
YAML_LANG = Language(tree_sitter_yaml.language())

@dataclass
class UniversalResult:
    path: str
    language: str
    loc: int = 0
    symbols: list[str] = field(default_factory=list)  # Classes, functions, top-level YAML keys
    metadata: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

class UniversalAnalyzer:
    def __init__(self):
        self.parsers = {
            "python": Parser(PY_LANG),
            "sql": Parser(SQL_LANG),
            "yaml": Parser(YAML_LANG),
        }

    def _node_text(self, node: Node, src: bytes) -> str:
        return src[node.start_byte : node.end_byte].decode("utf-8", errors="replace")

    def analyze(self, path: str | Path) -> UniversalResult:
        p = Path(path)
        ext = p.suffix.lower()
        
        lang_key = "unknown"
        if ext == ".py":
            lang_key = "python"
        elif ext == ".sql":
            lang_key = "sql"
        elif ext in (".yaml", ".yml"):
            lang_key = "yaml"

        result = UniversalResult(path=str(p), language=lang_key)
        
        try:
            src_bytes = p.read_bytes()
            result.loc = src_bytes.count(b"\n") + 1
        except Exception as e:
            result.errors.append(f"Read error: {e}")
            return result

        if lang_key not in self.parsers:
            return result

        parser = self.parsers[lang_key]
        try:
            tree = parser.parse(src_bytes)
            root = tree.root_node
            
            if lang_key == "python":
                self._analyze_python(root, src_bytes, result)
            elif lang_key == "sql":
                self._analyze_sql(root, src_bytes, result)
            elif lang_key == "yaml":
                self._analyze_yaml(root, src_bytes, result)
                
        except Exception as e:
            result.errors.append(f"Parse error: {e}")

        return result

    def _analyze_python(self, root: Node, src: bytes, result: UniversalResult):
        # Basic extraction similar to tree_sitter_analyzer.py but simplified for universal use
        for child in root.children:
            if child.type == "class_definition":
                name_node = child.child_by_field_name("name")
                if name_node:
                    result.symbols.append(self._node_text(name_node, src))
            elif child.type == "function_definition":
                name_node = child.child_by_field_name("name")
                if name_node:
                    result.symbols.append(self._node_text(name_node, src))

    def _analyze_sql(self, root: Node, src: bytes, result: UniversalResult):
        # Extract table references (CREATE TABLE, SELECT FROM)
        # Note: tree-sitter-sql grammar varies, but we look for identifiers after FROM/JOIN/INTO
        def walk(node: Node):
            if node.type == "table_reference":
                result.symbols.append(self._node_text(node, src))
            elif node.type == "relation": # Some dialects use relation
                result.symbols.append(self._node_text(node, src))
            for c in node.children:
                walk(c)
        walk(root)

    def _analyze_yaml(self, root: Node, src: bytes, result: UniversalResult):
        # Extract key hierarchy
        def walk(node: Node, prefix=""):
            if node.type == "block_mapping_pair":
                key_node = node.child_by_field_name("key")
                val_node = node.child_by_field_name("value")
                if key_node:
                    key_text = self._node_text(key_node, src).strip()
                    full_key = f"{prefix}.{key_text}" if prefix else key_text
                    result.symbols.append(full_key)
                    if val_node:
                        walk(val_node, full_key)
            elif node.type == "block_node":
                for c in node.children:
                    walk(c, prefix)
            elif node.type == "block_mapping":
                for c in node.children:
                    walk(c, prefix)
        walk(root)
