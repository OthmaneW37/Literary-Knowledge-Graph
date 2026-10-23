# NarrativeLens

**Assistant intelligent multimodal et agentique pour l’analyse de livres**

NarrativeLens fait évoluer Literary Chat sans remplacer son moteur de lecture : questions en français ou en anglais, recherche hybride locale BM25 + embeddings, citations vérifiées, progression de lecture et graphe de personnages restent disponibles. Une image facultative peut maintenant accompagner une question sur une page, une illustration, une carte ou une planche.

Les observations d’image servent à guider la recherche. Elles ne deviennent pas des faits sans preuve trouvée dans les passages du livre. La génération conserve les citations textuelles vérifiées et la limite de chapitre.

## Fonctionnalités

- **Accès direct à plus de 75 000 livres du domaine public** : l’onglet **Explorer** cherche dans le catalogue officiel de Project Gutenberg (titre, auteur, en français ou en anglais, tolérant aux accents et translittérations : « les frères Karamazov », « Dostoïevski »). « Ouvrir le livre » le télécharge, l’indexe et l’ouvre dans la discussion en quelques secondes ; la recherche sémantique se termine en arrière-plan.
- Import de fichiers personnels EPUB, TXT, Markdown, PDF avec texte et DOCX.
- Extraction locale d’images intégrées EPUB, PDF et DOCX, avec identifiant, livre, repère de chapitre/page, légende, contexte, OCR disponible et chemin local.
- Chat texte historique conservé, avec BM25, embeddings Ollama, fusion RRF, graphe et réponse extractive de secours.
- Question texte + image JPG, PNG ou WebP, prévisualisation, analyse VLM optionnelle et observations marquées comme incertaines.
- Recherche visuelle par embeddings image/texte CLIP multilingues en option. Les vecteurs sont mis en cache localement. Sans cette option, les questions texte continuent de fonctionner; les citations visuelles ne sont pas produites.
- Règle centrale anti-spoiler appliquée aux passages, relations et visuels récupérés.
- Progression, conversations et annotations locales; authentification JWT activable, stockage séparé par utilisateur pour ces données.
- Annotation préparée puis affichée exactement avant approbation; jeton lié au texte et aux preuves, sauvegarde idempotente.
- Serveur MCP optionnel pour la progression, les métadonnées et les annotations avec confirmation.
- Workflow LangGraph optionnel; chemin Python direct compatible si LangGraph n’est pas installé.
- Cross-encoder multilingue optionnel et scripts pour construire un jeu réel, entraîner et évaluer un reranker.
- **Banc d’essai intégré** : 76 scénarios annotés (faits, personnages, anti-spoiler, hors corpus, fausse prémisse, injection, comparaison, image), jouables un par un en direct dans l’onglet **Évaluation** ou en série, avec verdicts expliqués et historique des scores.
- Recherche renforcée par le modèle local : reformulation de la question dans la langue de l’édition (mots-clés + phrase hypothétique) et reclassement des 20 meilleurs candidats. Ces étapes ne servent qu’à trouver les passages ; la réponse reste vérifiée sur le texte.
- Garde anti-spoiler explicite pour les questions sur la fin du livre et correction des questions à fausse prémisse.

## Architecture

```mermaid
flowchart LR
  UI[React: question + image facultative] --> API[FastAPI: validation et limites]
  API --> VLM[VLM Ollama optionnel: observations]
  API --> WF[Workflow LangGraph optionnel]
  WF --> TXT[BM25 + embeddings Ollama]
  WF --> VIS[Embeddings visuels CLIP optionnels]
  TXT --> FUSION[Fusion RRF / reranking]
  VIS --> FUSION
  FUSION --> CHECK[Evidence checker + SpoilerPolicy]
  CHECK --> ANSWER[Réponse et citations vérifiées]
  API --> MCP[Outils MCP optionnels: données utilisateur]
  MCP --> APPROVAL[Approbation explicite avant annotation]
  TXT --> GRAPH[Neo4j facultatif + graphe local]
```

Le détail des chemins exécutables et de leurs limites est dans [docs/architecture.md](docs/architecture.md), [docs/multimodal-rag.md](docs/multimodal-rag.md), [docs/langgraph-workflow.md](docs/langgraph-workflow.md), [docs/mcp.md](docs/mcp.md), [docs/security.md](docs/security.md), [docs/reranker-training.md](docs/reranker-training.md) et [docs/evaluation.md](docs/evaluation.md).

## Installation locale

Prérequis : Python 3.11+, Node.js 20+, npm et Ollama. Sous PowerShell :

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,legacy-ui]"
Copy-Item .env.example .env
ollama pull qwen3.5:9b
ollama pull qwen3-embedding:0.6b
literary-library embeddings
```

Le modèle par défaut est **Qwen 3.5 9B** (6,6 Go, texte et images) : il tient avec le modèle d’embeddings dans une carte graphique de 12 à 16 Go. Sur une machine plus modeste, `OLLAMA_MODEL=qwen3.5:4b-q4_K_M` reste possible, avec des réponses nettement moins fiables (voir [docs/evaluation-results.md](docs/evaluation-results.md)). Le même modèle sert à l’analyse d’images (`VLM_MODEL`).

Pour les embeddings visuels et OCR local, installer seulement les extras nécessaires :

```powershell
python -m pip install -e ".[visual]"
$env:VISUAL_EMBEDDINGS = "clip"
python -m pip install -e ".[ocr]"
$env:OCR_ENABLED = "true"
```

L’extra OCR installe le client Python. Tesseract OCR doit également être installé sur le système et accessible dans `PATH`. L’OCR n’est lancé que sur des pages PDF sans texte exploitable, et seulement si `OCR_ENABLED=true`.

## Lancer l’application

Pour démarrer l’API et l’interface ensemble, depuis la racine :

```powershell
python scripts/dev.py
```

Le lanceur réutilise les serveurs déjà actifs. `Ctrl+C` arrête uniquement ceux qu’il a démarrés. Cela évite de lancer Vite seul et d’obtenir une erreur de proxy sur le port 8000.

Pour les lancer séparément :

Terminal 1, racine du dépôt :

```powershell
python -m uvicorn app.api:app --host 127.0.0.1 --port 8000
```

Terminal 2 :

```powershell
cd web
npm install
npm run dev
```

Ouvre `http://127.0.0.1:5173`. L’API et ses schémas sont sur `http://127.0.0.1:8000/docs`. Pour la version servie par FastAPI : `cd web; npm run build; cd ..; python -m uvicorn app.api:app --host 127.0.0.1 --port 8000`.

Active l’authentification dans `.env` avec `AUTH_ENABLED=true` et un `JWT_SECRET` aléatoire d’au moins 32 caractères; `APPROVAL_SECRET` permet d’utiliser une clé distincte pour les approbations. Génère une clé temporaire de session PowerShell avec `$env:JWT_SECRET = [Convert]::ToBase64String([Security.Cryptography.RandomNumberGenerator]::GetBytes(32))`. Les comptes locaux se créent depuis l’écran de connexion; l’API expose aussi `/api/auth/register` et `/api/auth/login`. Sans cette option, le mode local historique reste actif.

## Trouver un livre

Onglet **Explorer** (ou le bouton ＋ de la bibliothèque) : tape un titre ou un auteur, puis **Ouvrir le livre**. Le catalogue officiel de Project Gutenberg (`pg_catalog.csv.gz`, environ 80 000 entrées) est téléchargé une fois, mis en cache dans `data/library/catalog/` et rafraîchi chaque semaine ; la recherche fonctionne ensuite même si le miroir Gutendex est hors service. Le livre est téléchargé depuis gutenberg.org, découpé et indexé, puis s’ouvre dans la discussion. Les embeddings sont calculés en arrière-plan (progression dans la barre du haut) ; pendant ce temps la recherche par mots-clés et la reformulation par le modèle local répondent déjà.

**En français** : pour une recherche en français, Explorer interroge aussi **Wikisource** (API MediaWiki) et place ses textes en premier. Une œuvre est assemblée depuis sa page et ses sous-pages (« Livre I », « Livre II »…), par exemple *Les Frères Karamazov* dans la traduction Halpérine-Kaminsky et Morice (162 000 mots, environ 15 s). Une page qui ne liste que des éditions est signalée : il suffit de choisir une édition précise. Quand seul un texte anglais existe, chaque passage (sources d’une réponse, lecture par chapitre) propose **Traduire en français** : traduction automatique par le modèle local, clairement signalée, le texte original restant la preuve citée.

Les œuvres encore protégées par le droit d’auteur apparaissent comme fiches Open Library sans texte : si tu possèdes le fichier, **Importer un fichier personnel**.

## Import et questions multimodales

Depuis **Ajouter un livre**, importe un EPUB/PDF/DOCX contenant texte et images. Les images extraites restent dans `data/library/visuals`; leurs métadonnées sont dans `data/processed/*.visuals.json`. Les pages PDF gardent leurs numéros physiques. Pour une BD, chaque page image est l’unité visuelle; la segmentation des cases n’est pas faite.

Dans la discussion, joins un JPG, PNG ou WebP (10 Mo par défaut) et pose une question, par exemple :

- « Explique cette page sans me spoiler après le chapitre 5. »
- « Cette illustration peut-elle correspondre à une scène que j’ai déjà lue ? »
- « Quels indices du passage accompagnant cette carte aident à la lire ? »

Le VLM local par défaut est configuré par `VLM_PROVIDER` et `VLM_MODEL`. Sans VLM, la question textuelle est quand même traitée et l’interface indique l’absence d’analyse visuelle. Pour chercher une image dans les visuels du livre par similarité, active `[visual]` et `VISUAL_EMBEDDINGS=clip`; le premier passage calcule les vecteurs puis les conserve dans `data/library/visual_embeddings`. Ce modèle joint image/texte est séparé des observations du VLM.

## Services optionnels

Neo4j reste facultatif; lance-le avec `docker compose up -d neo4j` après avoir défini `NEO4J_PASSWORD`. Le graphe local continue sans Neo4j.

`QDRANT_*`, `POSTGRES_*` et `MLFLOW_*` sont réservés pour une évolution future : ces services ne sont pas encore raccordés au runtime et ne sont pas ajoutés au Compose. Le stockage actuel reste le système de fichiers local. Ollama s’exécute sur l’hôte, ce qui évite les contraintes GPU de Docker sous Windows.

Extras optionnels supplémentaires : `.[orchestration]` pour LangGraph, `.[mcp]` pour le SDK MCP et `.[training]` pour les outils d’entraînement. Les absences de ces paquets ne bloquent pas le serveur.

## MCP et annotations

Le serveur stdio s’exécute avec `narrativelens-mcp` après `python -m pip install -e ".[mcp]"`. Il propose les outils de lecture de progression, annotations, métadonnées, préparation et sauvegarde d’annotations. Définis `NARRATIVELENS_DATA_DIR` et `NARRATIVELENS_USER_ID` pour le profil local du serveur. La sauvegarde exige le jeton correspondant au contenu exact; un changement impose une nouvelle préparation. Le frontend utilise les routes locales FastAPI qui appellent le même service d’annotation.

## Reranker et évaluation

Le reranker par défaut est la diversification déterministe déjà présente. Pour essayer le cross-encoder pré-entraîné, installe l’extra `.[training]`, puis configure `RERANKER_PROVIDER=cross_encoder`. Pour un modèle entraîné, configure `RERANKER_PROVIDER=trained` et `RERANKER_PATH`.

Le jeu JSONL contient une question, un passage positif et des hard negatives :

```powershell
python training/build_dataset.py .\training\examples.jsonl .\data\library\reranker-train.jsonl
python training/train_reranker.py .\data\library\reranker-train.jsonl --base-model cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 --output .\models\reranker --epochs 2
python training/evaluate_reranker.py .\data\library\ranking-candidates.jsonl --output .\data\library\reranker-evaluation.json
python -m rag.evaluate --output data/library/evaluation-latest.json
```

Aucun entraînement ni score de benchmark n’est inclus ou prétendu. Les scripts écrivent uniquement les résultats mesurés sur les données fournies.

## Banc d’essai

L’onglet **Évaluation** de l’interface rejoue des scénarios annotés par le même chemin que la discussion et affiche, pour chacun, la réponse, ses citations et chaque contrôle réussi ou échoué. En ligne de commande :

```powershell
python -m evaluation validate   # chaque citation de référence existe mot pour mot et respecte la limite de chapitre
python -m evaluation run        # joue les 76 scénarios et enregistre le résultat
```

Méthode, catégories et limites : [docs/evaluation.md](docs/evaluation.md). Résultats mesurés avant et après les améliorations : [docs/evaluation-results.md](docs/evaluation-results.md).

## Tests et vérifications

```powershell
python -m pytest -q
cd web
npm run build
```

Les tests mockent les providers d’image et n’ont besoin ni de VLM, ni de CLIP, ni de Neo4j. Les appels réels aux modèles nécessitent les modèles correspondants dans Ollama; l’index visuel requiert son extra et son premier calcul d’embeddings.

## Structure

- `src/ingestion` : chargeurs, extraction visuelle, OCR optionnel et découpage.
- `src/retrieval` : BM25, embeddings, graphe, SpoilerPolicy, retrieval visuel et fusion.
- `src/vision` : provider VLM et sorties structurées d’observations.
- `src/reranking`, `training` : cross-encoder optionnel, données et entraînement.
- `src/workflows` : graphe de conversation LangGraph optionnel et fallback direct.
- `src/mcp_server`, `src/security` : outils MCP, approbation, comptes et jetons.
- `src/rag`, `src/catalog`, `src/graph`, `src/storage` : moteur existant, bibliothèque, graphe et persistance locale.
- `app/api.py` : API FastAPI.
- `web/src` : interface React/TypeScript.
- `tests` : suite pytest.

## Limites connues

Qdrant, PostgreSQL et MLflow sont des variables de configuration réservées et ne participent pas au runtime actuel. Le client FastAPI appelle le serveur MCP stdio lorsque MCP_TRANSPORT=stdio ; le mode direct reste disponible. L’authentification est un prototype local avec création de compte ouverte; elle n’a pas de limitation de débit. Les routes de conversation limitent les appels Ollama/MCP, le temps restant des appels réseau et la concurrence. Les traitements locaux bloquants ne sont pas interrompus de force. Le reranker entraîné nécessite un jeu réel construit par l’utilisateur et un entraînement effectif. Les relations du graphe sont encore principalement des relations de personnages issues du texte.

## Correspondance au sujet pédagogique

Consulter [l’audit des 14 rubriques](docs/project-alignment.md), le [profil de démonstration](docs/demo-profile.md) et le PDF `output/pdf/NarrativeLens_Projet_IA_Generative_Multimodale.pdf`.
