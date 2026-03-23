from __future__ import annotations

import asyncio
from collections import defaultdict
from datetime import datetime
from functools import wraps
from typing import Dict, List, Optional


class TokenTracker:
    """Tracks token usage (prompt, completion, reasoning, cached) per model.

    Cost calculation is intentionally omitted — prices change frequently and
    differ per provider tier. Use the raw token counts for cost estimation.
    """

    def __init__(self):
        self.token_counts: dict = defaultdict(
            lambda: {"prompt": 0, "completion": 0, "reasoning": 0, "cached": 0}
        )
        self.interactions: dict = defaultdict(list)

    def add_tokens(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        reasoning_tokens: int,
        cached_tokens: int,
    ) -> None:
        self.token_counts[model]["prompt"] += prompt_tokens
        self.token_counts[model]["completion"] += completion_tokens
        self.token_counts[model]["reasoning"] += reasoning_tokens
        self.token_counts[model]["cached"] += cached_tokens

    def add_interaction(
        self,
        model: str,
        system_message: str,
        prompt: str,
        response: str,
        timestamp: datetime,
    ) -> None:
        self.interactions[model].append(
            {
                "system_message": system_message,
                "prompt": prompt,
                "response": response,
                "timestamp": timestamp,
            }
        )

    def get_interactions(self, model: Optional[str] = None) -> Dict[str, List[Dict]]:
        if model:
            return {model: self.interactions[model]}
        return dict(self.interactions)

    def reset(self) -> None:
        self.token_counts = defaultdict(
            lambda: {"prompt": 0, "completion": 0, "reasoning": 0, "cached": 0}
        )
        self.interactions = defaultdict(list)

    def get_summary(self) -> Dict[str, Dict]:
        return {model: tokens.copy() for model, tokens in self.token_counts.items()}


# Global token tracker instance
token_tracker = TokenTracker()
