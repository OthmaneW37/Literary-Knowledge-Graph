from __future__ import annotations

from rag.local_index import tokenize
from rag.models import Passage


def diversify(passages: list[Passage], top_k: int) -> list[Passage]:
    """Greedy diversity reranking: avoid spending context on overlapping chunks.

    This is a deterministic post-fusion step, not a neural cross-encoder.
    Rank relevance remains dominant; near duplicates receive a penalty.
    """
    remaining = list(enumerate(passages))
    selected = []
    selected_terms = []
    while remaining and len(selected) < top_k:
        def score(item):
            rank, passage = item
            terms = set(tokenize(passage.text))
            similarity = max((len(terms & other) / max(len(terms | other), 1) for other in selected_terms), default=0)
            return 1 / (rank + 1) - (0.75 if similarity > 0.8 else 0)
        best = max(remaining, key=score)
        remaining.remove(best)
        selected.append(best[1])
        selected_terms.append(set(tokenize(best[1].text)))
    return selected
