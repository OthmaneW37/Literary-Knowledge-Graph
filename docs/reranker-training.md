# Entraînement du reranker

Le chemin actuel utilise BM25 + embeddings + RRF et une diversification déterministe. Le cross-encoder s’active par `RERANKER_PROVIDER=cross_encoder`; un checkpoint local par `RERANKER_PROVIDER=trained` et `RERANKER_PATH`. En cas de paquet/modèle absent ou d’erreur, la diversification locale est conservée.

Préparez un JSONL dont chaque ligne contient `question`, `positive`, `hard_negatives` (liste non vide). Conserver aussi work_id, document_id ou group_id : trois groupes au minimum sont exigés. Le partage train/validation/test se fait par groupe avant la création des paires, avec rejet des passages identiques entre partitions. Les négatifs devraient couvrir mauvaise scène du même personnage, même chapitre/mauvaise question, proximité lexicale trompeuse et entités proches. N’ajoutez que des labels contrôlés par un lecteur.

```powershell
python -m pip install -e ".[training]"
python training/build_dataset.py .\training\examples.jsonl .\data\library\reranker-train.jsonl
python training/train_reranker.py .\data\library\reranker-train.jsonl --base-model cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 --output .\models\reranker --epochs 2
```

L’entraînement écrit son rapport dans le dossier du modèle. Il n’a pas été exécuté sur ce dépôt faute de jeu étiqueté. N’annoncez pas de modèle adapté ou de gain avant un run réel et une comparaison reproductible.
