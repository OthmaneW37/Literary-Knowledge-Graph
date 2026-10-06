# Sécurité locale

- Les uploads d’image acceptent uniquement `image/jpeg`, `image/png` et `image/webp`, vérifient la signature binaire et sont plafonnés par `MAX_IMAGE_BYTES` (10 Mio par défaut).
- Les uploads de livre conservent leur limite existante de 50 Mio et leurs extensions supportées. Les noms enregistrés sont dérivés d’IDs ou de noms sans chemin.
- Les prompts demandent de traiter les pages, le texte OCR et l’historique comme des données, jamais comme des instructions. Les observations visuelles ne sont pas ajoutées au prompt de réponse : elles servent à la recherche.
- `AUTH_ENABLED=true` active les comptes locaux, les mots de passe PBKDF2 et les JWT HS256. Il faut définir `JWT_SECRET`; sans lui, les routes d’auth expliquent la configuration manquante. Le compte se crée localement depuis l’interface. La création est ouverte et ne fournit pas de rôles admin.
- Conversations, progression et annotations sont séparées par compte quand l’auth est active. Les livres restent partagés par défaut ; access_policy.json peut restreindre chacun par utilisateur ou groupe, avant recherche et ouverture des images.
- Les mutations annotation requièrent un jeton d’approbation à usage de contenu exact et se sauvegardent de façon idempotente.
- Le serveur MCP stdio utilise un `NARRATIVELENS_USER_ID` fixe fourni par son opérateur. Ne l’exposez pas à des clients non fiables.
- Les secrets restent dans `.env`; `.env` est ignoré par Git. Les budgets limitent les appels des adaptateurs Ollama et MCP dans les routes conversation/annotations ; les tâches d’ingestion et de construction du graphe restent séparées.

Le service est prévu pour un prototype local. Il n’implémente pas encore de limitation de débit ni de gestion multi-instance/révocation JWT. Ne l’exposez pas sur un réseau public sans durcir l’authentification et l’exploitation.
