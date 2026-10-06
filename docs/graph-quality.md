# Graphe : couverture, sources et limites

Le graphe est construit à partir des passages locaux, et non à partir des connaissances générales du modèle. La couverture indique combien de passages accessibles ont été analysés. Une couverture de 100 % signifie que chaque passage a été traité ; elle ne garantit pas que le modèle a identifié toutes les relations.

Dans **Graphe** ou **Personnages**, **Analyser tous les passages** traite les passages manquants successivement. La couverture et les résultats sont actualisés après chaque passage. **Mettre en pause** termine le passage en cours puis arrête la boucle. La reprise saute les passages déjà traités. Garder la page ouverte pendant l’analyse ; une interruption conserve les résultats enregistrés sur disque.

Les personnages isolés sont conservés, avec les passages où ils apparaissent. Ils sont présentés sous le dessin pour ne pas encombrer les relations. Les groupes récurrents ont une couleur distincte et ne sont pas comptés comme des individus. La vue Personnages donne accès à toutes les relations et mentions disponibles. Les lieux, thèmes et événements sont facultatifs dans le graphe. Les relations multiples entre deux nœuds sont regroupées sur une ligne, avec leurs citations sélectionnables.

Les identifiants des nœuds incluent le livre : deux personnages appelés Alice dans deux livres différents restent distincts. La résolution associe maintenant chaque **occurrence** (livre, passage, appellation) à une identité. Une même appellation peut désigner deux personnes différentes selon le passage. Les citations originales sont conservées ; seuls les nœuds et les extrémités affichées sont regroupés. La recherche accepte les appellations anciennes.

Pour la traduction anglaise de **La Métamorphose** du corpus, `data/annotations/identities/metamorphosis.json` fournit des attributions relues et des relations sourcées : Gregor, Grete, leurs parents, le patron, le fondé de pouvoir, etc. Ce sont des annotations explicites du corpus, et non une règle universelle qui remplacerait « his mother » dans tous les livres. « Mr. Samsa » désigne Gregor dans les occurrences du chapitre 1 et son père dans celles du chapitre 3. Les domestiques que les extraits ne permettent pas d’identifier avec certitude, les ensembles provisoires et les références génériques restent sans attribution. Le locataire du milieu est distinct du groupe des trois locataires.

Chaque annotation vérifie l’empreinte du passage et un ancrage accessible de l’identité. Une modification des sources invalide les attributions concernées. Les chapitres futurs ne servent pas à nommer les personnages de la vue courante.

Pour les autres textes, **Regrouper les personnages** effectue une résolution locale en lots reprenables, suivie d’une seconde vérification des attributions par le modèle. Le résultat complet est enregistré par livre, limite de chapitre et révision du texte. Les appels de lecture n’exécutent pas le modèle. Une référence incertaine ne crée pas un personnage. Cette voie automatique reste dépendante de la qualité du modèle et n’a pas la même garantie de relecture que les annotations du corpus. Une analyse terminée n’est pas une preuve d’exhaustivité.

Chaque relation affichée doit posséder une citation continue retrouvée dans son passage. L’extraction brute exige les appellations littérales des deux extrémités. La vue des identités peut afficher leurs noms canoniques, ou une relation relue du corpus, tout en conservant sa citation exacte et son passage. Une seconde passe du modèle vérifie aussi l’attribution et le sens de la relation, et rejette les liens insuffisamment appuyés. Ce contrôle réduit les erreurs sans garantir une interprétation parfaite ; consulter la citation reste nécessaire.

À la fin de l’analyse intégrale, une passe consacrée aux noms propres recherche les personnages oubliés pendant l’extraction des relations, à partir de titres et d’appels directs répétés. Les majuscules seules ne suffisent pas. Le modèle doit reconnaître une personne à partir des extraits. Les mentions supplémentaires sont conservées avec leur passage et son empreinte, sans fabriquer de liens.

Le filtre anti-spoiler s’applique aux relations, aux mentions des personnages et à la couverture. Une modification du texte invalide les extractions associées à l’ancienne version. Les requêtes de lecture ne déclenchent pas d’extraction coûteuse.

Tests de régression : `python -m pytest tests/test_graph_regressions.py -q`.


La projection Neo4j conserve les extractions littérales pour la recherche et leur validation. La résolution des identités décrite ici concerne les vues Graphe et Personnages de l’API ; elle ne réécrit pas les preuves brutes.

Vérifications : `python -m pytest -q`, puis `npm test` et `npm run build` dans `web/`. Les tests des identités couvrent les doublons, les titres ambigus, les changements de sources, les citations inventées, les attributions incertaines et les frontières entre livres et chapitres.
