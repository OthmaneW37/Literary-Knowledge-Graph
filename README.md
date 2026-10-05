# Literary Chat

Assistant littéraire local qui recherche dans des romans, répond avec des
citations vérifiables et affiche un graphe lorsque la question concerne des
personnages ou leurs relations.

Les romans installés et la génération restent sur la machine. Ollama exécute
le modèle de langage localement ; l'API publique Gutendex sert uniquement à
chercher et télécharger des ebooks du domaine public.

## Fonctionnalités

- ingestion de romans EPUB et nettoyage du texte ;
- catalogue Gutendex avec recherche par titre, auteur et langue ;
- téléchargement et indexation en un clic d'ebooks Project Gutenberg ;
- import direct de fichiers EPUB, TXT, Markdown, PDF textuels et DOCX ;
- découpage privilégiant les paragraphes, avec overlap et métadonnées de chapitre/page ;
- recherche hybride locale BM25 + embeddings multilingues avec cache incrémental ;
- expansion bilingue instantanée des questions françaises et anglaises ;
- réponse par `qwen3.5:4b-q4_K_M` avec validation des citations ;
- recherche sémantique par `qwen3-embedding:0.6b` ;
- réponse extractive de secours si Ollama est indisponible ;
- graphe de relations interactif (recherche, filtre, zoom, déplacement et preuve au clic), construit progressivement ;
- protection anti-spoiler jusqu'à un chapitre choisi ;
- sauvegarde locale des discussions, progression de lecture et export Markdown ;
- modes Ask, Explain, Analyze, Summarize, Characters, Quotes/Search et Compare ;
- interface web React + TypeScript, servie par Vite en développement ;
- API locale FastAPI pour relier l’interface au moteur RAG Python ;
- interface en ligne de commande ;
- base Neo4j optionnelle pour le graphe de connaissances permanent.

Le catalogue couvre les œuvres du domaine public. Pour un livre absent du
catalogue, l'utilisateur peut indexer un exemplaire numérique qu'il possède
légalement. Cette combinaison permet de travailler avec pratiquement tout
roman disponible en EPUB, texte ou PDF contenant une couche de texte, sans
envoyer son contenu vers un service externe.

Le corpus de démonstration contient *Metamorphosis* et *The Trial* de Franz
Kafka, dans les éditions indiquées par
`data/annotations/work_manifest.json`.

## Installation

Prérequis :

- Python 3.11 ou plus récent ;
- Node.js 20 ou plus récent et npm ;
- [Ollama](https://ollama.com/) installé et lancé ;
- les modèles locaux `qwen3.5:4b-q4_K_M` et `qwen3-embedding:0.6b`.

Sous PowerShell :

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
ollama pull qwen3.5:4b-q4_K_M
ollama pull qwen3-embedding:0.6b
Copy-Item .env.example .env
literary-library embeddings
```

Les modèles sont prêts si `ollama list` affiche les deux noms. La dernière
commande calcule une fois les vecteurs des passages ; les lancements suivants
réutilisent le cache et n'ajoutent que les nouveaux passages.

## Lancer le chat en développement

Dans un premier terminal, à la racine du projet :

```powershell
python -m uvicorn app.api:app --host 127.0.0.1 --port 8000
```

Dans un second terminal :

```powershell
cd web
npm install
npm run dev
```

Ouvre `http://127.0.0.1:5173`. Vite transmet les appels `/api` au serveur Python.
L’API seule est disponible sur `http://127.0.0.1:8000/docs`.

Pour une version de production locale, compile le frontend puis lance uniquement
le serveur Python :

```powershell
cd web
npm run build
cd ..
python -m uvicorn app.api:app --host 127.0.0.1 --port 8000
```

L'application complète est alors disponible sur `http://127.0.0.1:8000`.
La barre latérale gère la bibliothèque et les spoilers; les vues Discussion,
Chapitres, Recherche, Personnages et Graphe réutilisent le moteur RAG local.
Le catalogue permet de chercher et télécharger les œuvres du domaine public;
les fichiers EPUB, TXT, Markdown, DOCX et PDF textuels peuvent également être
importés depuis la machine. Les PDF scannés sans couche de texte nécessitent de
l’OCR, qui n’est pas encore intégré.

L’ancienne interface Streamlit est conservée comme option de transition :
installez-la avec `python -m pip install -e ".[legacy-ui]"` puis lancez
`streamlit run app\streamlit_app.py`.

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

Le catalogue est également disponible en ligne de commande :

```powershell
literary-library search "Dostoyevsky" --language en
literary-library import 2554
literary-library upload ".\mon-roman.epub" --title "Mon roman" --author "Auteur" --language fr
literary-library embeddings
```

Les livres ajoutés sont enregistrés dans `data/library/installed_books.json`,
et leurs fichiers générés restent sous `data/raw` et `data/processed`. Ces
données locales sont ignorées par Git.

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
4. redémarrer le serveur API afin de recharger l'index.

## Architecture

```text
API Gutendex ou EPUB local
  -> téléchargement local sécurisé
  -> texte nettoyé
  -> chapitres, passages par paragraphes et métadonnées JSON
  -> index lexical + cache vectoriel locaux

question FR ou EN
  -> analyse heuristique et expansion bilingue (< 5 ms)
  -> recherches BM25 et sémantique
  -> seuil de pertinence + fusion RRF
  -> un unique appel Ollama contraint par les sources
  -> validation des citations et des arêtes
  -> réponse + passages + diagramme éventuel
```

Les modules principaux sont :

- `src/ingestion` : extraction, nettoyage et découpage ;
- `src/catalog/gutendex.py` : recherche et téléchargement via Gutendex ;
- `src/catalog/library.py` : installation et indexation de la bibliothèque locale ;
- `src/rag/local_index.py` : index BM25 sans service externe ;
- `src/retrieval/semantic_retriever.py` : embeddings Ollama et cache incrémental ;
- `src/ingestion/loaders.py` : chargeurs standardisés EPUB, texte, Markdown, PDF et DOCX ;
- `src/llm/ollama_provider.py` : accès configurable au fournisseur Ollama ;
- `src/rag/config.py` : limites de contexte et réglages de performance ;
- `src/retrieval/query_analyzer.py` : classification et réécriture multilingue ;
- `src/retrieval/hybrid_retriever.py` : fusion BM25 + embeddings et orchestration Neo4j ;
- `src/retrieval/graph_retriever.py` : relations sourcées entre personnages ;
- `src/rag/engine.py` : génération, validation et visualisation ;
- `app/api.py` : API locale FastAPI, façades vers les services existants ;
- `web/src/ui` : interface React TypeScript (chat, citations, bibliothèque et graphe) ;
- `app/streamlit_app.py` : ancienne interface, optionnelle pour la transition ;
- `src/extraction` : extraction structurée destinée au graphe permanent ;
- `src/graph` : chargement Neo4j optionnel.

Les choix d'architecture, les frontières entre composants et les compromis
sont détaillés dans [`docs/architecture.md`](docs/architecture.md).

## Performance

Le chemin standard n'effectue plus deux appels LLM par question. L'analyse
multilingue est déterministe par défaut, le modèle reste chargé pendant 15
minutes, quatre passages sont envoyés par défaut et le contexte est plafonné.

Le modèle 4B quantifié réduit la mémoire et la génération par rapport au 7B.
Le mode de raisonnement est explicitement désactivé pour ce chat factuel afin
de retourner directement le JSON attendu. Après la pré-indexation, une question
ne calcule qu'un seul embedding court, puis effectue un unique appel au modèle
de réponse.

- analyse de la question : environ 1 ms, contre environ 16 s auparavant ;
- recherche BM25 : quelques millisecondes ;
- recherche sémantique avec modèle déjà chargé : un embedding de question ;
- la première exécution reste plus lente car Ollama charge les modèles en mémoire.

Les temps exacts dépendent du CPU, du GPU et de la longueur de la réponse.
L'interface affiche désormais le temps total et le temps de recherche.

Les réglages sont centralisés dans `.env` :

```text
OLLAMA_MODEL=qwen3.5:4b-q4_K_M
OLLAMA_EMBED_MODEL=qwen3-embedding:0.6b
OLLAMA_KEEP_ALIVE=15m
RAG_EMBED_BATCH_SIZE=16
RAG_MIN_SEMANTIC_SCORE=0.40
RAG_MIN_LEXICAL_COVERAGE=0.30
RAG_TEMPERATURE=0
LLM_PROVIDER=ollama
OLLAMA_HOST=http://127.0.0.1:11434
OLLAMA_TIMEOUT_SECONDS=120
RAG_QUERY_ANALYSIS=heuristic
RAG_NUM_CTX=8192
RAG_NUM_PREDICT=700
RAG_MAX_CONTEXT_CHARS=18000
RAG_CHUNK_SIZE=240
RAG_CHUNK_OVERLAP=40
RAG_RERANK=true
```

`RAG_QUERY_ANALYSIS=llm` réactive la reformulation par le modèle pour les cas
complexes, au prix d'un second appel et d'une latence supérieure.

## Tests

```powershell
python -m pytest -q
```

Les tests n'appellent pas Ollama : ils utilisent des réponses simulées pour
valider la recherche, les citations, les graphes et le mode de secours.

Lancer l'évaluation locale. Les extraits annotés sont peu nombreux et les
métriques de sources représentent un jeu de régression, pas un score académique :

```powershell
python -m rag.evaluate --output data/library/evaluation-latest.json
python -m rag.evaluate --generate
```

## Activer Neo4j

Les extractions de graphe sourcées sont conservées localement dans
`data/library/graphs`. Neo4j ajoute une projection facultative. Pour le démarrer
avec Docker, définissez `NEO4J_PASSWORD` dans `.env`, puis lancez :

```powershell
docker compose -p literary-chat up -d neo4j
```

Les ports sont accessibles uniquement depuis la machine locale. Sans Neo4j,
le chat et le graphe local restent utilisables. Pour configurer le pilote :

```text
NEO4J_URI=neo4j://127.0.0.1:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=votre_mot_de_passe
NEO4J_ENABLED=true
```

Construire progressivement un lot de relations, puis le synchroniser si besoin :

```powershell
python -m literary_chat graph --work metamorphosis --limit 10 --sync
```

Les personnages sont séparés par œuvre. Chaque relation conserve son type,
son `work_id`, son passage de preuve et la citation extraite.
Dans l’interface, le graphe se construit par lots afin de garder l’extraction
locale contrôlable ; les relations ne sont ajoutées que si les deux entités et
leur citation sont présentes dans le passage source. Clique sur une arête pour
afficher le passage complet.

## Limites actuelles

Le projet utilise des passages d'environ 240 mots avec un chevauchement de 40 mots :
BM25 reste précis sur les noms et citations, tandis que les embeddings
multilingues retrouvent les paraphrases et les questions posées dans une autre
langue. Un réordonnancement local réduit les passages presque identiques ; aucun cross-encoder lourd n'est requis.
Le catalogue distant est limité aux ebooks disponibles légalement dans Project
Gutenberg ; les titres encore protégés peuvent apparaître dans d'autres
catalogues, mais leur texte intégral ne peut pas être téléchargé automatiquement.
Ils doivent être importés depuis un fichier obtenu légalement. Les PDF scannés
sans texte nécessitent une étape OCR qui n'est pas incluse.
