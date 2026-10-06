# Évaluation

Le dépôt possède déjà `python -m rag.evaluate` pour son jeu de régression local. L’outil `training/evaluate_reranker.py` calcule Recall@5, nDCG@5 et MRR@10 sur des classements fournis en JSONL, avec `positive_ids` et `candidate_ids_by_system` pour les systèmes à comparer (par exemple baseline, cross-encoder initial et modèle entraîné).

```powershell
python -m rag.evaluate --output data/library/evaluation-latest.json
python training/evaluate_reranker.py .\data\library\ranking-candidates.jsonl --output .\data\library\reranker-evaluation.json
```

Les résultats n’existent qu’après exécution sur un dataset. Les tests unitaires ne sont pas des scores de qualité. L’évaluation automatisée couvre des règles de citations, spoiler, auth, images et annotations; les mesures de recherche Recall@5, nDCG@5, MRR@10 et les latences p50/p95 sont intégrées. La fidélité sémantique, MLflow et un benchmark comparatif multimodal restent à compléter.
