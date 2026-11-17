# Visualisation du Spotify Tracks Dataset

Projet de l'UE **Visualisation des données** · DI FEDE Santiago, SELMANE Ilyane, TOUPIN Nicolas

Quatre visualisations interactives pour comprendre ce qui distingue les genres musicaux et ce qui rend un morceau populaire.

## Contenu

| Fichier | Rôle |
|---|---|
| `presentation_spotify.html` | **Interface de présentation** (s'ouvre dans un navigateur et fonctionne hors ligne) |
| `compte_rendu.md` | Compte rendu : calculs, questions, tâches et résultats de chaque visualisation |
| `projet_spotify_visualisation.ipynb` | Notebook : public, objectifs, tâches et vérifications sur les données |
| `build_presentation.py` | Script qui génère `presentation_spotify.html` |

## Installation

```bash
python3.11 -m venv .venv
.venv/bin/pip install ipykernel pandas plotly scipy nbformat
```

Il faut ensuite placer `dataset.csv` à la racine du projet. Ce fichier ([Spotify Tracks Dataset, Kaggle](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset)) n'est pas versionné.

## Utilisation

- **Régénérer l'interface** : `.venv/bin/python build_presentation.py`
- **Notebook** : ouvrir le notebook dans VS Code, choisir le kernel `.venv`, puis cliquer sur *Run All*.
