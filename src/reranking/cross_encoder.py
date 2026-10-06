from __future__ import annotations


class CrossEncoderReranker:
    """Lazy, optional multilingual cross-encoder over query/passage pairs."""

    def __init__(self, model_name: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", model=None) -> None:
        self.version = model_name
        self._model = model

    def rerank(self, query: str, passages: list, top_k: int) -> list:
        if not passages or top_k < 1:
            return []
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.version)
        scores = self._model.predict([(query, passage.text) for passage in passages])
        ranked = sorted(zip(passages, scores), key=lambda item: float(item[1]), reverse=True)
        from dataclasses import replace
        return [replace(passage, score=float(score)) for passage, score in ranked[:top_k]]
