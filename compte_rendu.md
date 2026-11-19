# Compte rendu : visualisation du Spotify Tracks Dataset

**UE Visualisation des données** · Membres : DI FEDE Santiago, SELMANE Ilyane, TOUPIN Nicolas

## Contexte

- **Données** : 114 000 morceaux, 114 genres (1 000 par genre), 31 437 artistes, 7 caractéristiques audio comprises entre 0 et 1 (*danceability, energy, valence, acousticness, speechiness, instrumentalness, liveness*), plus `tempo`, `loudness`, `duration`, `popularity` et `explicit`.
- **Préparation** : suppression des 3 valeurs manquantes et des doublons (`track_id`, `track_genre`), durée convertie en minutes, profils moyens calculés par genre.
- **Public** : curateurs de playlists, programmateurs et DJs, qui sont des connaisseurs de musique mais pas des statisticiens.
- **Livrables** : `presentation_spotify.html`, une interface autonome qui fonctionne hors ligne, et `projet_spotify_visualisation.ipynb`, le notebook de vérification.

| Objectif | Visualisation | Tâches |
|---|---|---|
| O1 · Caractériser les genres | Carte ACP + biplot | T1, T2 |
| O2 · Comprendre la popularité | Heatmap de corrélations par genre | T3, T4 |
| O3 · Construire une sélection | Coordonnées parallèles + brushing | T6, T7 |
| O4 · Repérer l'atypique | Strip plot des écarts | T8 |

## Interactions : vues coordonnées (*overview + detail*)

Les 4 vues partagent **une seule sélection : le genre**. L'interface suit un parcours en trois niveaux :

1. **Vue globale** : toutes les vues montrent l'ensemble du catalogue.
2. **Sélection d'un genre**, de trois façons : barre de recherche, clic sur un point de la carte (O1) ou clic sur une colonne de la heatmap (O2). Les 4 vues se mettent alors à jour **en même temps**.
3. **Détail d'un morceau** : un clic sur un morceau (tableau O3 ou point O4) ouvre une **fiche**. Elle donne la popularité, le tempo, la durée et l'écart au genre, ainsi qu'un **radar qui compare le morceau à la moyenne de son genre**.

| Vue | Vue globale | Genre sélectionné |
|---|---|---|
| O1 Carte | 114 genres, couleur = popularité | genre mis en avant et relié à ses 5 plus proches voisins, les autres estompés ; radar du profil du genre comparé au catalogue |
| O2 Heatmap | 30 genres aux liens les plus forts | colonne du genre encadrée (et ajoutée si elle n'y était pas) ; phrase qui résume ce qui fait monter ou baisser la popularité dans ce genre |
| O3 Coord. parallèles | échantillon de 8 000 morceaux ; tableau calculé sur tout le catalogue | **tous** les morceaux du genre ; le tableau des morceaux filtrés se met à jour à chaque déplacement de filtre |
| O4 Strip plot | 12 genres les plus hétérogènes | strip plot du genre seul, ses morceaux les plus atypiques et la caractéristique qui les rend atypiques |

Le bouton **« Vue globale »** annule la sélection. Les filtres posés en O3 sont conservés quand on change de genre.

---

## O1 · Carte des genres (ACP + biplot)

**Question** : quel est le profil sonore de chaque genre, et quels genres se ressemblent ?

**Ce qui est calculé**
1. Le profil moyen de chaque genre sur les 7 caractéristiques, ce qui donne une matrice de 114 × 7.
2. Une standardisation en z-scores, pour que chaque caractéristique pèse autant que les autres.
3. Une **analyse en composantes principales** (décomposition SVD) qui projette les genres en 2D. Les deux axes conservent **58 %** de la variance (35 % + 23 %).
4. Les **flèches** (biplot), qui donnent la contribution de chaque caractéristique aux deux axes.
5. La **couleur**, qui encode la popularité médiane du genre.

**Tâches** : T1 (*comparer* les profils, en lisant la position d'un genre par rapport aux flèches) et T2 (*identifier* les genres similaires grâce à la proximité sur la carte).

**Pourquoi cette technique** : elle résume 7 dimensions en une seule image où la distance entre deux points représente la ressemblance musicale. Le biplot évite au lecteur d'avoir à interpréter lui-même les axes.

**Résultat** : « house » est proche de *edm*, *dance* et *electro*. Les genres les plus singuliers sont *sleep*, *comedy*, *new-age* et *ambient*.

**Limite** : 42 % de l'information n'est pas représentée, donc deux genres proches sur la carte peuvent quand même différer sur un axe masqué.

---

## O2 · Heatmap des corrélations par genre

**Question** : quelles caractéristiques audio sont associées aux morceaux les plus écoutés ?

**Ce qui est calculé**
1. Les morceaux dont la popularité vaut 0 sont exclus, car il s'agit de morceaux sans score (constat fait en vérification T3).
2. **Pour chaque genre**, la corrélation de Spearman ρ entre la popularité et chacune des 10 variables (7 caractéristiques + tempo, loudness, durée).
3. Les 30 genres dont le lien le plus fort est le plus marqué sont retenus, puis ordonnés par clustering hiérarchique (méthode de Ward) pour rapprocher les genres qui se comportent de la même façon.
4. Les couleurs suivent une échelle divergente : orange quand ρ > 0, gris quand ρ ≈ 0, bleu quand ρ < 0.

**Tâches** : T4 (*corréler* les caractéristiques et la popularité) et T3 (*résumer* la popularité par genre, avec le traitement des valeurs 0).

**Pourquoi cette technique** : une matrice genre × caractéristique montre 300 corrélations d'un seul coup d'œil et fait apparaître des blocs de genres qui réagissent de la même manière.

**Résultat** : sur tout le catalogue, les liens sont faibles (|ρ| ≤ 0,17). Calculés à l'intérieur de chaque genre, ils sont bien plus forts : 30 couples genre × caractéristique dépassent |ρ| = 0,3, et le maximum est ρ = 0,55 entre *loudness* et la popularité en *k-pop*. **Il n'existe pas de recette universelle de la popularité, elle dépend du genre.**

**Limite** : une corrélation ne prouve pas une causalité. Spearman ne mesure par ailleurs que les relations monotones.

---

## O3 · Coordonnées parallèles avec brushing

**Question** : comment trouver rapidement des morceaux qui correspondent à une ambiance donnée ?

**Ce qui est calculé**
1. Un échantillon de 8 000 morceaux distincts, auquel on ajoute tous les morceaux qui correspondent au préréglage.
2. Chaque morceau est tracé comme une **ligne** qui traverse 7 axes : tempo, énergie, danceability, valence, acousticness, durée (plafonnée à 10 min) et popularité.
3. Le **brushing** fonctionne ainsi : l'utilisateur sélectionne une plage sur un axe, les filtres se combinent (ET logique), et seules les lignes qui satisfont tous les filtres restent colorées.
4. La couleur encode la popularité.

**Tâches** : T6 (*filtrer* selon des plages de tempo, d'énergie et de valence) et T7 (*consulter* le détail : le tableau des morceaux retenus, puis la fiche d'un morceau au clic).

**Pourquoi cette technique** : c'est la technique de référence pour filtrer de façon interactive selon plusieurs critères numériques. Elle reproduit le geste d'un curateur (« 120–130 BPM, très énergique… ») tout en gardant la vue d'ensemble.

**Résultat** : avec le préréglage « soirée dansante » (120–130 BPM, énergie > 0,8, danceability > 0,7), **1 850 morceaux** correspondent sur l'ensemble du catalogue. Le plus populaire est *One Kiss* de Calvin Harris et Dua Lipa (89/100).

**Limite** : au-delà de quelques milliers de lignes le graphique devient surchargé, d'où l'échantillon de 8 000 morceaux en vue globale (le tableau, lui, porte sur tout le catalogue).

---

## O4 · Strip plot des écarts au profil du genre

**Question** : quels morceaux sortent du profil habituel de leur genre ?

**Ce qui est calculé**
1. Pour chaque morceau et chaque caractéristique, un z-score par rapport au genre : (valeur − moyenne du genre) / écart-type du genre.
2. L'**écart** du morceau, égal à √(Σ z²), c'est-à-dire sa distance euclidienne au profil moyen de son genre.
3. Un seuil fixé au **99ᵉ percentile** de cet écart, soit 6,2 (ligne pointillée).
4. Sont affichés les 12 genres les plus hétérogènes, avec un point par morceau, et le morceau le plus atypique de chaque genre mis en évidence.

**Tâche** : T8 (*localiser* les valeurs aberrantes).

**Pourquoi cette technique** : en affichant **chaque morceau** plutôt qu'un résumé (comme une boîte à moustaches), on voit d'un côté la norme du genre (la masse de points) et de l'autre les exceptions, qu'on identifie au survol.

**Résultat** : **1 136 morceaux** dépassent le seuil. Le record est *Alvará – Ao vivo* de Jorge Aragão (pagode), avec un écart de 31. Son *instrumentalness* se situe à **+31 écarts-types** de la moyenne du genre, alors que la quasi-totalité des morceaux de pagode sont à 0.

**Limite** : quand une caractéristique varie très peu dans un genre (écart-type proche de 0), un petit écart en valeur absolue produit un z-score énorme. Un « atypique » peut donc être dû à une seule variable.

---

## Conclusion

Les quatre vues forment un parcours *overview → detail*. On commence par la carte des genres (O1), on regarde ensuite ce qui fait la popularité dans chacun d'eux (O2), on passe à la sélection concrète de morceaux (O3), puis on termine sur les exceptions (O4). Le principal enseignement est que **le genre structure tout** : les profils sonores sont très distincts, et même les liens avec la popularité changent d'un genre à l'autre.

_Pour régénérer l'interface : `.venv/bin/python build_presentation.py`._
