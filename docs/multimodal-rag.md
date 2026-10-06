# Retrieval multimodal

Les chargeurs lisent le texte comme avant et collectent les images incorporées lorsqu’ils sont accessibles. Chaque visuel a une empreinte stable, un chemin local, un type, le livre, une position connue, une légende, du texte voisin et un champ OCR. Les PDF conservent la page physique. L’EPUB associe l’image au document de lecture qui la référence; DOCX fournit le média intégré et un contexte de document.

`LocalVisualIndex` lit les manifestes visuels. Avec `VISUAL_EMBEDDINGS=clip` et l’extra `.[visual]`, il utilise l’encodeur d’image CLIP et son encodeur de texte multilingue aligné, calcule les embeddings au premier besoin et les met en cache sous `data/library/visual_embeddings`. Cela permet texte-vers-image et image-vers-image. Le modèle multilingue couvre le français; sa carte de modèle explique que l’encodeur image CLIP original est inchangé et doit être associé au modèle textuel multilingue : [carte du modèle](https://huggingface.co/sentence-transformers/clip-ViT-B-32-multilingual-v1). Sans le provider, la branche image ne retourne pas de faux résultats; le chat textuel reste disponible.

Le VLM est indépendant du retriever. `VisionAnalyzer` retourne un modèle Pydantic fermé (OCR observé, description, entités visibles, objets, scène possible et incertitudes). La question envoyée au modèle de réponse demeure celle de l’utilisateur; les observations servent uniquement de termes de recherche. Le moteur de réponse doit citer ses passages réels.

Les visuels trouvés sont exposés sur `/api/visuals/{visual_id}` après validation du chemin sous le répertoire visuel local. L’API `POST /api/chat/multimodal` reçoit la question, les IDs de livres et éventuellement un fichier. L’image est supprimée du fichier temporaire à la fin de la requête.

L’OCR PDF est désactivé par défaut. `OCR_ENABLED=true` déclenche pytesseract seulement pour des pages sans couche texte et des images embarquées reconnues par pypdf. Il faut installer Tesseract système. L’extraction/segmentation des cases BD n’est pas faite; une page demeure l’unité.
