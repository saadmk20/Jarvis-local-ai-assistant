"""Self-contained web search engine: plan -> search -> fuse -> judge -> read -> answer."""

from .config import DEFAULT_LIMIT, MAX_LIMIT
from .pipeline import WebSearch

__all__ = ["WebSearch", "DEFAULT_LIMIT", "MAX_LIMIT"]
