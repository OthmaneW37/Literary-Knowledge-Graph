# Literary Chat

Assistant littéraire local qui recherche dans des romans, répond avec des
citations vérifiables et affiche un graphe lorsque la question concerne des
personnages ou leurs relations.

Toutes les données et la génération restent sur la machine : aucune API
externe n'est nécessaire. Ollama exécute le modèle de langage localement.

## Fonctionnalités du MVP

- ingestion de romans EPUB et nettoyage du texte ;
- découpage par chapitre et en passages citables ;
- recherche locale BM25 dans un ou plusieurs romans ;
- expansion locale des questions françaises vers les textes anglais ;
- réponse par `qwen2.5:7b-instruct` avec validation des citations ;
- réponse extractive de secours si Ollama est indisponible ;
- diagrammes de relations et chronologies produits à partir des sources ;
- interface de chat Streamlit ;
- interface en ligne de commande ;
- base Neo4j optionnelle pour le graphe de connaissances permanent.

Le corpus de démonstration contient *Metamorphosis* et *The Trial* de Franz
Kafka, dans les éditions indiquées par
`data/annotations/work_manifest.json`.

## Installation

Prérequis :

- Python 3.11 ou plus récent ;
- [Ollama](https://ollama.com/) installé et lancé ;
- le modèle local `qwen2.5:7b-instruct`.

Sous PowerShell :

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
ollama pull qwen2.5:7b-instruct
Copy-Item .env.example .env
```

Le modèle est déjà présent si `ollama list` affiche
`qwen2.5:7b-instruct`.

## Lancer le chat

```powershell
streamlit run app\streamlit_app.py
```

Streamlit ouvre normalement l'application sur `http://localhost:8501`.
Choisissez les romans dans la barre latérale, puis posez une question.

Exemples :

- `Pourquoi Gregor veut-il cacher sa transformation ?`
- `Quelle autorité les gardiens exercent-ils sur Josef K. ?`
- `Montre les relations entre Gregor et sa famille.`
- `Quels thèmes relient les deux romans ?`

## Utilisation en ligne de commande

Après l'installation éditable :

```powershell
literary-chat "Pourquoi Josef K. est-il arrêté ?" --work the_trial
```

Ou directement avec Python :

```powershell
python -m rag.cli "Pourquoi Josef K. est-il arrêté ?" --work the_trial
```

## Réindexer les EPUB

Les EPUB sont décrits dans le manifeste et placés dans `data/raw`.

```powershell
python -m ingestion.ingestion_pipeline
python -m ingestion.build_all_chunks
```

Pour ajouter une œuvre :

1. placer son EPUB dans `data/raw` ;
2. ajouter ses métadonnées dans `data/annotations/work_manifest.json` ;
3. relancer les deux commandes d'ingestion ;
4. redémarrer Streamlit afin de recharger l'index.

## Architecture

```text
EPUB
  -> texte nettoyé
  -> chapitres et passages JSON
  -> index lexical local

question
  -> expansion multilingue locale
  -> récupération des meilleurs passages
  -> génération Ollama contrainte par les sources
  -> validation des citations et des arêtes
  -> réponse + passages + diagramme éventuel
```

Les modules principaux sont :

- `src/ingestion` : extraction, nettoyage et découpage ;
- `src/rag/local_index.py` : index BM25 sans service externe ;
- `src/retrieval/query_analyzer.py` : classification et réécriture multilingue ;
- `src/retrieval/hybrid_retriever.py` : orchestration BM25 + Neo4j ;
- `src/retrieval/graph_retriever.py` : relations sourcées entre personnages ;
- `src/rag/engine.py` : génération, validation et visualisation ;
- `app/streamlit_app.py` : interface utilisateur ;
- `src/extraction` : extraction structurée destinée au graphe permanent ;
- `src/graph` : chargement Neo4j optionnel.

## Tests

```powershell
python -m pytest -q
```

Les tests n'appellent pas Ollama : ils utilisent des réponses simulées pour
valider la recherche, les citations, les graphes et le mode de secours.

## Activer Neo4j

Neo4j reste facultatif : sans connexion, le chat continue avec les passages
textuels. Pour activer le graphe persistant, renseignez dans `.env` :

```text
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=votre_mot_de_passe
```

Après avoir produit les extractions structurées correspondant aux chunks
actuels, chargez le graphe :

```powershell
python -m graph.load_neo4j
```

Les personnages sont séparés par œuvre. Chaque relation conserve son type,
son `work_id`, son passage de preuve et la citation extraite.

## Limites actuelles

Ce MVP utilise des passages d'environ 600 mots et un index lexical enrichi par
une expansion multilingue locale.
L'étape suivante consiste à ajouter des embeddings locaux pour une recherche
sémantique plus fine, puis à rendre le graphe Neo4j persistant et entièrement
traçable jusqu'aux passages sources.
