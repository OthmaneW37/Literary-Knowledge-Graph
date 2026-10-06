# Workflow de conversation

`src/workflows/state.py` décrit l’état de requête (propriétaire, question, livres, chapitre, observations et réponse). `NarrativeWorkflow` appelle le moteur LiteraryAssistant existant, contrôle les citations et le spoiler boundary, puis route vers réponse ou clarification.

Avec `python -m pip install -e ".[orchestration]"`, le flux est compilé via LangGraph :

```text
START -> retrieve -> evidence_check -> réponse
                                 \-> clarification
```

Sans LangGraph, `invoke` exécute les mêmes fonctions séquentiellement. Le workflow est borné; il n’a pas de boucle ni d’agent autonome. L’annotation constitue une action distincte et requiert une confirmation du lecteur avant écriture.
