# Correspondance de NarrativeLens au projet pédagogique

Référence lue intégralement : **Projet IA Générative Multi Modale V2**, sujet MaintiDoc, 10 pages. Audit du 6 octobre 2026. Le document est un modèle à adapter au domaine littéraire : les choix techniques sont présentés comme possibles, les chiffres de performance comme des objectifs.

## Identité conservée

Lectures et imports EPUB, questions françaises/anglaises, citations, progression anti-spoiler, graphe de personnages et enrichissement des fiches restent au centre du produit. L'action approuvée est l'annotation, équivalent littéraire du ticket de maintenance. Le stockage local et React restent des choix assumés ; remplacer ces composants par PostgreSQL/MinIO/Streamlit n'est pas nécessaire pour cette transposition.

## Matrice des 14 rubriques

| Rubrique du sujet | Correspondance et état | Preuves / reste à faire |
|---|---|---|
| 1. Résumé | Assistant littéraire multimodal, graphe et annotations | README, interface React, API |
| 2. Cadrage | Lecteurs/étudiants, livres textuels et illustrés | Corpus actuel : 2 livres, 570 passages ; cible 30-50 documents non atteinte |
| 3. Architecture | FastAPI, Ollama, LangGraph/LangChain Core, index local, MCP | Extras installés, Docker construit ; CLIP/OCR facultatifs ; bases externes non raccordées |
| 4. RAG | Ingestion texte/images, BM25/vecteurs/RRF, citations, anti-spoiler | VLM réel vérifié sur image synthétique ; benchmark de retrieval visuel à réaliser. Les images retrouvées ne sont pas toutes envoyées avec les extraits à un VLM de synthèse ; observations et réponses textuelles restent séparées |
| 5. Fine-tuning | Scripts cross-encoder, partage par livre, métriques | Fuite train/validation corrigée ; entraînement effectif absent faute de jeu humain de 800-1500 exemples |
| 6. Agent et MCP | StateGraph borné, RunnableLambda, client/serveur stdio, annotation approuvée | Transport réel testé. Validation humaine en deux routes avec jeton signé, pas de checkpoint LangGraph durable |
| 7. Prompts/sorties | YAML versionnés, Pydantic, modèles/prompts/index tracés | Les versions déclarées ne remplacent pas un manifeste complet des révisions de modèles |
| 8. Incertitude | Clarification, preuves absentes, observations explicitement incertaines | Détection générale de contradictions et notation humaine à étendre |
| 9. Sécurité | Auth, ACL utilisateur/groupe, images protégées, approbation, outils autorisés, budgets | Corpus partagé par défaut, ACL explicites configurables ; quotas cumulés, chiffrement, rétention et limite GPU à compléter |
| 10. Évaluation | 113 tests Python, 12 tests JS, 11 questions locales, métriques et latences | Les tests de code ne remplacent pas les 150 scénarios annotés. Seulement 5 questions ont des passages gold |
| 11. MLOps | requirements.lock, package-lock, CI, Docker, événements techniques et rapports JSON | Build/démarrage Docker vérifiés. CI distante non exécutée, MLflow et scan continu à intégrer |
| 12. Planning | Six semaines adaptées à la littérature | Planning indicatif, pas une déclaration de travaux terminés |
| 13. Livrables | Code, docs, tests, audit, PDF de dix pages | Benchmark final, modèle adapté et campagne de scénarios restent à produire |
| 14. Démonstration | Livre/image, graphe, preuve, MCP, annotation, refus/injection | Parcours à répéter sur corpus illustré avec comptes et ACL de démonstration |

## Corrections réalisées pendant cet audit

1. **nDCG@5** : les occurrences pertinentes au-delà du rang 5 ne contribuent plus au score ; déduplication des IDs.
2. **Train/validation/test** : partage par document/livre avant expansion des paires ; conservation des identifiants de source ; rejet des passages répétés entre partitions et des négatifs vides/identiques au positif.
3. **Accès aux livres et images** : filtrage de bibliothèque, passages, recherche, graphe, conversations et progression. Images authentifiées et chargées en Blob côté React pour éviter un jeton dans l'URL. Même politique côté MCP.
4. **MCP** : client stdio relié à FastAPI pour les annotations ; identité fixée par l'API, outils autorisés explicitement, mêmes vérifications serveur, erreurs normalisées.
5. **Orchestration** : LangGraph installé et exécuté ; RunnableLambda et limite explicite de huit étapes.
6. **Budgets** : compteurs par requête pour Ollama (vision et embeddings compris) et MCP ; temps réseau restant, exclusion concurrente, en-têtes de mesure. Délai coopératif, sans interruption forcée des calculs locaux/GPU ; imports et construction du graphe séparés.
7. **Vision** : configuration sur Qwen 3.5 4B installé et réellement compatible vision ; tokens et contexte bornés ; durée affichée incluant l'analyse d'image.
8. **Déploiement** : dépendances figées, Dockerfile multi-stage, conteneur non root, profil Compose et CI.
9. **Mesures** : Recall@5, nDCG@5, MRR@10, p50/p95 ajoutés à l'évaluateur ; rapport réellement exécuté.

## Résultats mesurés

- Python : **113 passed** ; JavaScript : **12 passed** ; TypeScript/Vite : compilation réussie.
- MCP réel : préparer, lire, sauvegarder, répéter sans doublon, isoler les notes d'un autre utilisateur, rejeter une modification après approbation et un outil non autorisé.
- Docker : build réussi ; API bibliothèque HTTP 200 avec deux livres et frontend HTTP 200, conteneur temporaire arrêté après test.
- Vision : lecture exacte de « CHAPITRE 1 / Alice ouvre le livre. » sur une image synthétique, Qwen 3.5 4B, 11,59 secondes. Cette image n'est pas un benchmark littéraire.
- Recherche : 11 questions, dont 10 avec une œuvre attendue et 5 avec passages de référence. Taux de récupération d'œuvre 0,800 ; Recall@5 1,000, nDCG@5 0,926, MRR@10 0,900 sur les cinq questions annotées. Médiane 53 ms et p95 2 414 ms sur les 11 recherches. Embeddings textuels désactivés, reste du moteur local conservé ; aucune génération de réponse pendant cette mesure.

Les résultats sont petits et non exhaustifs. Aucun score de fidélité globale, gain de fine-tuning ou garantie de sécurité n'est déduit de ces mesures.

## Pièces reproductibles

- `tests/test_project_alignment.py` : régressions ciblées et MCP réel.
- `docs/demo-profile.md` : configuration, politique d'accès, budgets et Docker.
- `docs/alignment-measurements.json` : résumé figé des mesures de cet audit.
- `scripts/build_project_pdf.py` : source du PDF (ReportLab, polices Calibri/Consolas Windows).
- `output/pdf/NarrativeLens_Projet_IA_Generative_Multimodale.pdf` : adaptation de dix pages, mêmes 14 rubriques que le modèle.

## Travaux nécessitant encore un corpus et des mesures

Constituer un corpus de livres illustrés autorisés ; faire vérifier les labels de pertinence ; réaliser l'entraînement et la comparaison sur un test distinct ; compléter les 150 scénarios, notamment contradictions, ACL et injections ; mesurer CLIP/OCR et génération multimodale sur le matériel ; raccorder MLflow et compléter les contrôles d'exploitation. Le PDF distingue explicitement ces objectifs des capacités déjà vérifiées.
