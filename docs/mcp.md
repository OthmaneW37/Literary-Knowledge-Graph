# MCP

L’extra `.[mcp]` installe le SDK MCP. La commande `narrativelens-mcp` démarre un serveur stdio local avec les outils `get_reading_progress`, `get_user_annotations`, `get_book_metadata`, `prepare_annotation` et `save_annotation`.

Le profil du serveur est défini par `NARRATIVELENS_DATA_DIR` et `NARRATIVELENS_USER_ID`. Préparation et sauvegarde valident les IDs de preuves contre les passages locaux et la progression du profil. L’outil de préparation ne modifie rien; il produit un jeton court, lié au texte exact, propriétaire, chapitre et preuves. Une mutation avec contenu changé ou jeton expiré est rejetée. Une empreinte stable rend la sauvegarde idempotente.

L’interface React présente un aperçu modifiable avant préparation, puis bloque son texte et affiche exactement le texte associé au jeton avant le clic final. Le frontend appelle les routes FastAPI locales et celles-ci utilisent le même `AnnotationService`; elles ne créent pas de client vers un serveur MCP distant.
