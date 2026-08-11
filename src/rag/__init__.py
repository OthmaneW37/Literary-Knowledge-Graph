"""Local retrieval-augmented literary assistant.

``LiteraryAssistant`` is imported lazily so low-level modules such as
``retrieval`` can reuse ``rag.models`` without importing the full engine and
creating a circular dependency.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .engine import LiteraryAssistant

__all__ = ["LiteraryAssistant"]


def __getattr__(name: str) -> Any:
    if name == "LiteraryAssistant":
        from .engine import LiteraryAssistant

        return LiteraryAssistant
    raise AttributeError(name)
