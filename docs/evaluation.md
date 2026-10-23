# Évaluation : le banc d’essai de NarrativeLens

NarrativeLens est évalué sur un **jeu de 76 scénarios annotés** (`data/annotations/scenarios.json`), joués par le **même chemin que la discussion** : garde anti-spoiler, recherche hybride, réponse du modèle local, vérification des citations, contrôle des preuves. Chaque verdict est **déterministe et explicable** : aucun modèle ne note un autre modèle.

## Les huit catégories

| Catégorie | Scénarios | Ce qui est vérifié |
|---|---:|---|
| Faits du texte | 20 | Réponse rédigée, citation contenant le passage de référence, terme attendu, langue de la question |
| Personnages et relations | 14 | Mêmes contrôles, sur l’identité et les liens entre personnages |
| Anti-spoiler | 11 | Aucune preuve au-delà du chapitre lu, aucun terme révélateur, refus des questions sur la fin |
| Hors corpus | 8 | Abstention : aucune affirmation sans preuve dans les livres choisis |
| Fausse prémisse | 8 | La question suppose un fait faux ; le moteur ne doit pas le confirmer |
| Injection de consignes | 8 | Ordres cachés, faux mode debug, fausse autorité, fausse source, balise piégée |
| Comparaison de livres | 4 | Preuves tirées de chaque œuvre comparée |
| Question avec image | 3 | Une page ou un objet en image relié au bon passage |

Une sélection « vitrine » (un ou deux scénarios par catégorie) sert à la démonstration.

## Annotation et intégrité

Chaque scénario porte la question, le comportement attendu en clair, la ou les **citations de référence** et, selon le cas, des termes attendus ou interdits. `python -m evaluation validate` vérifie que **chaque citation existe mot pour mot** dans le corpus et qu’elle **ne dépasse pas la limite de chapitre** du scénario : une référence inventée ou elle-même « spoilante » est refusée. Les images des scénarios multimodaux sont produites depuis le texte indexé par `scripts/make_scenario_images.py`.

## Utilisation

- **Dans l’application** : onglet **Évaluation**. Choisis un scénario (filtre par catégorie, mode vitrine), clique **Jouer** : la vraie réponse, ses citations dépliables et chaque contrôle ✓/✗ s’affichent. **Tout jouer** lance le jeu complet en arrière-plan et affiche le score par catégorie et l’écart avec l’exécution précédente.
- **En ligne de commande** :

```powershell
python -m evaluation validate
python -m evaluation run
python -m evaluation run --category spoiler
python -m evaluation run --only fact-01,prem-01
```

Chaque exécution est enregistrée dans `data/library/evaluation/` (date, modèle, résumé, détail par scénario).

## Résultats mesurés

Voir la section « Avant / après » de [evaluation-results.md](evaluation-results.md).

## Autres mesures

`python -m rag.evaluate` mesure la recherche seule (Recall@5, nDCG@5, MRR@10, latences) sur `eval_questions.json`, et `training/evaluate_reranker.py` compare des classements fournis en JSONL. Les tests unitaires (`pytest`, `npm test`) vérifient le code ; ils ne sont pas des scores de qualité.

## Limites

Le corpus compte deux romans en anglais ; les chiffres ne se généralisent pas à d’autres livres sans nouveaux scénarios. Les contrôles textuels (termes attendus ou interdits) sont volontairement simples et lisibles : une réponse juste formulée autrement peut échouer, une fuite paraphrasée peut passer. La fidélité fine d’une réponse reste à confirmer par une lecture humaine des citations, que l’onglet Évaluation rend immédiate.
