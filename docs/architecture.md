# Architecture technique

## État de départ et évolution

NarrativeLens conserve le moteur Literary Chat existant : FastAPI, React/TypeScript, Ollama, index BM25 local, embeddings textuels incrémentaux, fusion RRF, citations vérifiées, anti-spoiler par chapitre, graphe local et projection Neo4j facultative. Le chemin texte n’exige pas de VLM, CLIP, LangGraph, MCP ou base externe.

## Chemins exécutables

```text
Livre local EPUB/PDF/DOCX
  -> extraction texte + images intégrées
  -> passages JSON + métadonnées visuelles locales
  -> BM25 + cache d'embeddings Ollama
  -> cache d'embeddings visuels CLIP (si activé)

Question texte + image facultative
  -> FastAPI : format, signature et taille de l'image
  -> observations VLM optionnelles (texte d'image non fiable)
  -> recherche textuelle existante et recherche visuelle optionnelle
  -> réponse sur la question d'origine, observations seulement pour guider la recherche
  -> validation de citations + SpoilerPolicy
  -> UI : preuves texte, observations incertaines et visuels retrouvés
```

`NarrativeWorkflow` utilise un `StateGraph` LangGraph si l'extra est installé. Il exécute le chemin RAG actuel, vérifie les citations et choisit réponse ou clarification. Sans LangGraph, les mêmes étapes sont exécutées localement sans empêcher le démarrage.

## Stockage

- Textes, passages, catalogue, graphe local, historique en mode dev, progression et images restent sous `data/`.
- Les visuels sont sauvegardés dans `data/library/visuals/<work_id>/`; les descriptions et repères dans `data/processed/<work_id>.visuals.json`.
- Les vecteurs image sont mis en cache dans `data/library/visual_embeddings/` avec le modèle local.
- Quand `AUTH_ENABLED=true`, utilisateurs, progression, conversations et annotations sont séparés sous `data/library/users/<clé opaque>/`. La bibliothèque reste sur le disque local ; des ACL par livre peuvent restreindre sa consultation.
- Neo4j est une projection facultative du graphe. Qdrant, PostgreSQL et MLflow ne sont pas encore raccordés malgré leurs variables réservées.

## Anti-spoiler et evidence

`SpoilerPolicy` refuse les chapitres ultérieurs et les repères inconnus quand une limite existe. La recherche sémantique, les branches ciblées, le graphe, l'ajout de contexte adjacent, les citations visuelles et la préparation d'annotation appliquent la même règle. Le moteur texte vérifie les identifiants de source et les citations exactes avant affichage. Une observation VLM n'est pas incluse comme preuve et n'identifie pas seule un personnage.

## Services facultatifs

| Capacité | Composant | Sans le composant |
|---|---|---|
| Génération et embeddings texte | Ollama | Réponse extractive; recherche BM25 |
| Analyse d’image | VLM Ollama | Chat texte conservé; message d’indisponibilité |
| Similarité visuelle | sentence-transformers CLIP | Pas de retrieval visuel, le texte continue |
| OCR PDF scanné | pytesseract + Tesseract système | Le PDF scanné est refusé avec indication |
| Graphe persistant | Neo4j | Graphe local |
| Orchestration explicite | LangGraph | Fallback Python direct |
| Actions externes | SDK MCP, stdio | Appels API locaux pour annotation |
| Cross-encoder | sentence-transformers | Diversification locale existante |

## Sécurité et frontières

L'upload image accepte JPG, PNG et WebP, valide la signature réelle, impose `MAX_IMAGE_BYTES`, puis traite l'image dans un fichier temporaire supprimé après la requête. Les fichiers importés restent locaux. Auth utilise un hash PBKDF2 et des JWT HS256 lorsque activée. Les jetons d'approbation d'annotation lient le propriétaire, le livre, le chapitre, le texte et les IDs de preuves; le stockage est idempotent.

Le modèle, le VLM et les embeddings optionnels doivent être téléchargés séparément. L’évaluateur mesure les latences p50/p95. Le middleware journalise seulement route, durée, statut et compteurs d’appels. Les appels Ollama/MCP des conversations sont bornés ; MLflow n’est pas raccordé. La segmentation BD reste à la page.

## Tests

La suite pytest teste les composants locaux avec providers simulés, notamment la limite spoiler sur visuels, l'upload, les observations, les approbations d'annotation, l'idempotence, l'isolation et le workflow de repli. `npm run build` vérifie TypeScript et Vite.
