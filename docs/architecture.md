# Architecture technique

## Objectif

Literary Chat répond à des questions en français ou en anglais sur un corpus
de romans choisi par l'utilisateur. Une réponse doit rester rattachée aux
passages réellement indexés et ne doit pas dépendre d'une API de génération.

## Parcours des données

```text
Gutendex / fichier utilisateur
        │
        ▼
validation du format et de la taille
        │
        ▼
extraction EPUB, texte ou PDF
        │
        ▼
nettoyage → chapitres → passages JSON
        │
        ▼
index BM25 en mémoire + embeddings persistés

question FR/EN
        │
        ▼
analyse heuristique + expansion bilingue
        │
        ├── recherche BM25
        ├── recherche sémantique multilingue
        └── Neo4j à la demande (optionnel)
                    │
                    ▼
     fusion réciproque des classements
                    │
                    ▼
           contexte borné et sourcé
                    │
                    ▼
             génération Ollama
                    │
                    ▼
       validation citations / diagramme
```

## Responsabilités

- `catalog` obtient ou reçoit les fichiers et maintient le catalogue local.
- `ingestion` transforme un document en passages stables et citables.
- `retrieval` analyse la question et rassemble les preuves textuelles ou
  structurées.
- `rag` construit une réponse contrainte, valide les identifiants de source et
  expose des mesures de latence.
- `graph` stocke les entités et relations optionnelles sans être nécessaire au
  fonctionnement du chat.
- `app/api.py` expose les opérations du moteur par HTTP local ; `web/` fournit
  l'interface React/TypeScript. Ni l'API ni le navigateur ne réimplémentent
  l'indexation ou la validation des sources.

## Décisions importantes

### Un appel LLM sur le chemin standard

La classification précédente utilisait le même modèle 7B que la génération,
ce qui ajoutait un chargement et une génération avant même la recherche. Le
mode par défaut est maintenant heuristique et bilingue. Le mode LLM reste
activable par configuration lorsqu'une meilleure résolution conversationnelle
est plus importante que la latence.

### Recherche hybride locale

BM25 avec index inversé est rapide et précis sur les noms propres. Les vecteurs
de `qwen3-embedding:0.6b` améliorent le rappel pour les paraphrases et les
questions dont la langue diffère de celle du roman. Une fusion RRF combine les
deux classements sans comparer directement leurs échelles de score. Les
vecteurs des passages sont calculés par lots, persistés sous `data/library` et
seuls les passages nouveaux ou modifiés sont recalculés.

### Neo4j facultatif et paresseux

La base graphe ne se connecte qu'à la première question qui en a besoin et
seulement lorsque `NEO4J_ENABLED=true`. Une
base arrêtée ne bloque donc ni le démarrage, ni les questions textuelles. Les
diagrammes peuvent aussi être produits directement depuis les passages, puis
sont filtrés pour ne conserver que les arêtes associées à une preuve valide.

### Couverture des livres

Gutendex fournit légalement les textes du domaine public. Les titres qui ne
peuvent pas être téléchargés sont couverts par l'import d'un exemplaire local
EPUB, TXT, Markdown ou PDF textuel. Le contenu d'un livre importé reste sous
`data/` et est ignoré par Git.

## Garanties et limites

- Une citation inventée par le modèle est rejetée.
- Une arête de diagramme sans passage de preuve est rejetée.
- La taille des fichiers et du contexte envoyé au modèle est bornée.
- Une réponse extractive reste disponible quand Ollama est arrêté.
- Un PDF image nécessite de l'OCR avant import.
- Le moteur ne peut pas garantir une réponse correcte si l'information n'est
  pas présente dans le texte ou si aucun passage pertinent n'est récupéré.

## Stratégie de test

Les tests unitaires couvrent le découpage, BM25, l'expansion bilingue, les
citations, les diagrammes, Neo4j dégradé, Gutendex et les imports locaux. Les
routes FastAPI de bibliothèque, de passages et de chat ont aussi des tests
d'intégration avec un moteur simulé. Les appels externes et Ollama sont
simulés dans la suite automatisée.
