# Recherche et ajout de livres

Depuis **Ajouter un livre**, saisir un titre, un auteur, un ISBN ou une description. Le modèle local reformule les demandes longues ; si le modèle est indisponible, la recherche conserve les mots saisis.

Les recherches interrogent Open Library (fiches, couvertures, sujets et descriptions) et Gutendex / Project Gutenberg (EPUB ou texte téléchargeable). Choisir un résultat permet de consulter sa provenance avant de télécharger ou d'associer un fichier personnel. Une panne de catalogue affiche un avertissement sans masquer les résultats de l'autre catalogue.

## Anna’s Archive

Le bouton ouvre une recherche EPUB dans le navigateur. Le téléchargement automatique depuis Anna’s Archive n'est pas intégré : les requêtes anonymes rencontrent une vérification anti-bot et son API de téléchargement rapide nécessite une clé membre. Le fichier obtenu peut ensuite être importé avec la fiche sélectionnée.

## Métadonnées et sources

- L'import EPUB récupère son titre, son auteur, sa langue, sa description, ses sujets et sa couverture intégrée.
- « Compléter la fiche » enrichit un livre existant sans remplacer son texte.
- Les métadonnées externes conservent leur catalogue, leur URL et leur date de récupération.
- Les résumés constituent un contexte complémentaire. Ils ne sont pas indexés comme passages du livre et ne peuvent pas servir de citations.
- Les résumés complets sont exclus du contexte des réponses lorsqu'une limite de chapitre est active.
- Les noms de personnages issus des fiches restent des candidats : ils doivent être retrouvés et validés dans le texte avant d'enrichir le graphe.

## Vérifications effectuées

106 tests Python et 12 tests JavaScript réussis ; compilation TypeScript/Vite réussie. Import réel de l'EPUB Gutenberg n°11 dans un dossier temporaire : couverture extraite, résumé conservé, passages indexés. Recherche Open Library vérifiée en ligne. La recherche Gutendex a rencontré des délais dépassés, même si l'accès direct à la fiche et au fichier n°11 fonctionnait.

La vérification visuelle du nouveau formulaire reste à faire : l'outil de navigateur était indisponible lors de cette session.
