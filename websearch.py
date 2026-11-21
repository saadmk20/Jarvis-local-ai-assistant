"""Compatibility shim: the search engine now lives in the `jarvis.search` package.

Import from `jarvis.search` in new code; this module only keeps old scripts working.
"""
from jarvis.search import DEFAULT_LIMIT, MAX_LIMIT, WebSearch

__all__ = ["WebSearch", "DEFAULT_LIMIT", "MAX_LIMIT"]
