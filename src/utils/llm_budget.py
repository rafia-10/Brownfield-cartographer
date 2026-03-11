"""
llm_budget.py
~~~~~~~~~~~~~
Tracks token usage and estimated spend across LLM calls to prevent "bill shock".
Supports tiered model selection and context window monitoring.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional

import tiktoken

log = logging.getLogger(__name__)

@dataclass
class ModelInfo:
    name: str
    input_price_per_1k: float
    output_price_per_1k: float
    max_tokens: int

# Pricing as of early 2024 (best effort estimates)
MODEL_REGISTRY = {
    "gemini-1.5-flash-latest": ModelInfo("gemini-1.5-flash-latest", 0.000125, 0.000375, 1000000),
    "gemini-1.5-pro-latest": ModelInfo("gemini-1.5-pro-latest", 0.0035, 0.0105, 2000000),
    "claude-3-haiku": ModelInfo("claude-3-haiku", 0.00025, 0.00125, 200000),
    "claude-3-5-sonnet": ModelInfo("claude-3-5-sonnet", 0.003, 0.015, 200000),
    "gpt-4o-mini": ModelInfo("gpt-4o-mini", 0.00015, 0.0006, 128000),
}

@dataclass
class ContextWindowBudget:
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_spend: float = 0.0
    limit_usd: float = 5.0  # Default $5 limit
    
    def update(self, model: str, input_tokens: int, output_tokens: int):
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        
        info = MODEL_REGISTRY.get(model)
        if info:
            cost = (input_tokens / 1000 * info.input_price_per_1k) + \
                   (output_tokens / 1000 * info.output_price_per_1k)
            self.total_spend += cost
            
        if self.total_spend > self.limit_usd:
            log.warning(f"BUDGET EXCEEDED: ${self.total_spend:.4f} / ${self.limit_usd:.2f}")

    @staticmethod
    def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))

budget = ContextWindowBudget()
