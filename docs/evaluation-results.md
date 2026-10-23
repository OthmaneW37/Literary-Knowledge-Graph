# Résultats mesurés : avant / après

Mesures du 22 octobre 2026 sur les 76 scénarios de `data/annotations/scenarios.json` (méthode : [evaluation.md](evaluation.md)). Modèle local `qwen3.5:4b-q4_K_M`, embeddings `qwen3-embedding:0.6b`, Ollama sur la même machine. Chaque ligne est une exécution complète réelle, enregistrée dans `data/library/evaluation/`.

**Résultat actuel : 61/76 (80 %) avec Qwen 3.5 9B**, contre 34/76 au départ. Le détail par étape suit.

## Synthèse (première phase, modèle 4B)

| Catégorie | Avant | Après |
|---|---:|---:|
| Faits du texte | 3/20 | **12/20** |
| Personnages et relations | 2/14 | **4/14** |
| Anti-spoiler | 7/11 | **10/11** |
| Hors corpus | 8/8 | **8/8** |
| Fausse prémisse | 4/8 | 3/8 |
| Injection de consignes | 8/8 | **8/8** |
| Comparaison de livres | 1/4 | **2/4** |
| Question avec image | 1/3 | **2/3** |
| **Total** | **34/76 (45 %)** | **49/76 (64 %)** |

Temps médian par scénario : 3,3 s avant, 6,5 s après (étapes de recherche supplémentaires par le modèle local).

## Ce que le banc d’essai a révélé, et les corrections

| Exécution | Score | Changement mesuré |
|---|---:|---|
| Référence | 34 | Recherche : le passage qui répond n’est dans les 4 premiers que pour 17 questions factuelles sur 41 ; questions françaises sur texte anglais quasi introuvables. Fin du livre demandée avant le chapitre lu : extraits bruts au lieu d’un refus clair. |
| 2 | 39 | **Garde anti-spoiler explicite** (refus motivé des questions sur la fin), **détection des fausses prémisses**, **reformulation de la question dans la langue de l’édition** (mots-clés + phrase hypothétique) et **reclassement des 20 meilleurs candidats** par le modèle local : 25 questions sur 41 trouvent leur passage dans les 4 premiers. |
| 3-4 | 40-41 | Vérification des citations à l’identique **mot pour mot** (ponctuation, guillemets typographiques et coupures « … » tolérés, aucun mot changé) ; citations rattachées au bon passage ; identifiants « [SOURCE …] » nettoyés. |
| 5 | 44 | Citations demandées **avant** la réponse et dans la langue du livre (le modèle les traduisait) ; une citation fausse est retirée seule et l’audit des faits porte alors sur les sources restantes. |
| 6 | **49** | Quand l’audit refuse une reformulation, réponse par **citations exactes déjà vérifiées** plutôt que par extraits bruts. |

## Passage au modèle Qwen 3.5 9B (23 octobre 2026)

Constat d’usage : avec le 4B, une question large (« Montre-moi les liens entre les personnages ») recevait des citations sans rapport, et le repli « citation exacte » montrait des passages hors sujet. Changements : modèle **Qwen 3.5 9B** (16 Go de mémoire graphique disponibles), questions sur l’ensemble du livre orientées vers un chemin dédié (personnages principaux détectés dans le texte, relations exigeant une citation exacte), plus aucune citation non vérifiée présentée comme réponse, correction neutre conservée quand une fausse prémisse est détectée.

| Catégorie | 4B (exécution 6) | **9B** |
|---|---:|---:|
| Faits du texte | 12/20 | **13/20** |
| Personnages et relations | 4/14 | **9/14** |
| Anti-spoiler | 10/11 | **11/11** |
| Hors corpus | 8/8 | **8/8** |
| Fausse prémisse | 3/8 | **8/8** |
| Injection de consignes | 8/8 | **8/8** |
| Comparaison de livres | 2/4 | 2/4 |
| Question avec image | 2/3 | 2/3 |
| **Total** | 49/76 | **61/76 (80 %)** |

Corrections ajoutées avec le 9B : recherche équilibrée quand plusieurs livres sont sélectionnés (chaque livre fournit ses passages), une seconde tentative quand l’audit refuse un détail (le modèle reçoit la raison et ne garde que ce que le texte affirme), citations limitées en longueur (les citations trop longues tronquaient la réponse), prénoms et surnoms donnés ensemble à la recherche (Aliocha / Alexéi). Trois annotations ont reçu un second passage de référence, vérifié mot pour mot, quand la réponse juste citait un autre passage tout aussi probant.

Le score du 4B comptait des réponses « citation exacte » qui ne sont plus produites (elles pouvaient être hors sujet) ; le 9B atteint 52/76 sans elles. Temps médian : 11 s par question (6,5 s avec le 4B).

## Lecture honnête des échecs restants

- **Personnages (4/14)** : le modèle de 4 milliards de paramètres cite souvent un passage voisin de celui qui prouve le fait ; le contrôle « passage de référence cité » le signale. Un modèle plus grand ou un reranker entraîné (`training/`) sont les pistes suivantes.
- **Fausse prémisse (3/8)** : le modèle confirme parfois l’hypothèse fausse ou répond par des citations sans la contester. Aucune réponse n’invente pour autant le fait supposé.
- **Audit des faits** : il rejette encore quelques reformulations justes (par excès de prudence). Le moteur répond alors par la citation exacte, ce qui reste fidèle au texte.
- Les contrôles textuels sont volontairement simples et lisibles ; la fidélité fine se vérifie en ouvrant les citations dans l’onglet Évaluation.

## Accès direct aux livres (vérifié en réel)

Dans l’onglet **Explorer**, « Les Frères Karamazov » trouve *The Brothers Karamazov* (Project Gutenberg n° 28054). « Ouvrir le livre » l’a téléchargé, découpé en 2 052 passages et ouvert dans la discussion en une quinzaine de secondes, embeddings terminés en arrière-plan. À la question « Qui est Aliocha et quel est son lien avec Fiodor Pavlovitch ? », la réponse vérifiée a été : « Aliocha est le troisième fils de Fiodor Pavlovitch Karamazov », appuyée sur « Alexey Fyodorovitch Karamazov was the third son of Fyodor Pavlovitch Karamazov » (les noms français sont rapprochés de l’orthographe du livre). « Orgueil et préjugés » trouve *Pride and Prejudice* grâce à la traduction du titre par le modèle local.

## Reproduire

```powershell
python -m evaluation validate
python -m evaluation run
```

Ou, dans l’application, onglet **Évaluation** → **Tout jouer**. Les résultats varient légèrement d’une exécution à l’autre (génération par un modèle local).
