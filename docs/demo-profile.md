# Profil de démonstration intégré

## Installation et exécution

```powershell
python -m pip install -r requirements.lock
python -m pip install --no-deps -e .
# .env : MCP_TRANSPORT=stdio ; VLM_MODEL=qwen3.5:4b-q4_K_M
python scripts/dev.py
```

LangGraph et LangChain Core sont installés dans ce profil. Le nœud de récupération utilise RunnableLambda ; l'état circule dans un StateGraph borné à 8 étapes. La validation humaine des annotations utilise deux requêtes séparées, un jeton signé expirant et une écriture idempotente. Ce n'est pas un checkpoint LangGraph durable.

MCP_TRANSPORT=stdio active le client MCP réel de FastAPI pour préparer, sauvegarder et consulter les annotations d'un livre. Le serveur enfant reçoit une identité fixée par l'API, le dossier de données et la clé d'approbation. Le nom de l'outil et l'exécutable sont contrôlés par le code. Aucun outil d'envoi de documents n'existe. Les réponses MCP sont des données structurées et ne sont pas interprétées comme des instructions par un modèle.

## Contrôle d'accès

AUTH_ENABLED=true et JWT_SECRET (au moins 32 caractères) activent l'authentification. Les images sont chargées avec l'en-tête Authorization ; aucun jeton n'est placé dans une URL. La bibliothèque est partagée par défaut. Pour une démonstration avec groupes, créer `data/library/access_policy.json` :

```json
{
  "default": "private",
  "user_groups": {"alice": ["atelier_a"], "bob": ["atelier_b"]},
  "books": {
    "metamorphosis": {"groups": ["atelier_a"]},
    "the_trial": {"groups": ["atelier_b"]}
  }
}
```

Cette politique filtre bibliothèque, passages, recherche, graphe, images, conversations, progression et annotations MCP. Les fichiers restent accessibles à l'opérateur du système : ce contrôle applicatif ne remplace pas les permissions du disque. Un livre non mentionné suit `default` ; ne pas laisser la valeur `shared` si le corpus doit être fermé.

## Limites et journalisation

Le profil utilise 6 appels de modèle (embeddings compris), 3 appels MCP et 60 secondes de budget, avec une requête d'inférence concurrente. Les appels réseau Ollama reçoivent le temps restant ; un budget épuisé empêche les appels suivants et retourne une erreur ou une réponse extractive. Ce délai est coopératif : il ne tue pas un calcul local bloquant ni un calcul déjà lancé sur le GPU distant. Le plafond physique GPU et les quotas cumulés par utilisateur restent à instrumenter. Les tâches d'import et de graphe sont hors de ce budget conversationnel.

Les réponses portent `X-Model-Calls`, `X-MCP-Calls` et `Server-Timing`. Les événements techniques ne journalisent ni question, ni contenu du livre, ni jeton. Les conversations, elles, sont conservées explicitement dans le stockage utilisateur ; définir une durée de conservation avant un usage collectif.

## Docker et retour arrière

```powershell
docker compose --env-file .env -f compose.demo.yml up --build -d
```

Définir JWT_SECRET et APPROVAL_SECRET avant l'exécution. Le corpus est monté depuis `./data` ; conserver les permissions de lecture/écriture de l'utilisateur du conteneur (UID 10001 sous Linux). Ollama tourne sur l'hôte. Sauvegarder `data` avant migration. En cas de régression, arrêter le profil, redéployer l'image précédente et restaurer la sauvegarde compatible. Le compose historique Neo4j reste facultatif.

Le fichier requirements.lock fige le profil Python installé et package-lock.json le frontend. Les images Docker sont identifiées par digest dans le journal de build. La CI exécute les tests et la compilation ; elle n'a pas encore été exécutée sur GitHub dans cette session.
