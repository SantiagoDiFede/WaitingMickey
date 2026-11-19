"""Génère presentation_spotify.html : interface interactive (thème Spotify) qui relie
les 4 visualisations du projet autour d'une sélection de genre (overview + detail).

  Vue globale  →  sélection d'un genre (recherche, clic sur la carte ou la heatmap)
               →  les 4 vues se recentrent sur ce genre
               →  clic sur un morceau (tableau O3, point O4) : fiche détaillée

Python prépare les données ; JavaScript (plotly.js, embarqué) dessine et relie les vues.
Usage : .venv/bin/python build_presentation.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from plotly.offline import get_plotlyjs
from scipy.cluster.hierarchy import linkage, leaves_list

ICI = Path(__file__).parent

# ----------------------------------------------------------------------------
# Préparation (identique au notebook)
# ----------------------------------------------------------------------------
df = pd.read_csv(ICI / "dataset.csv", index_col=0)
features = ["danceability", "energy", "valence", "acousticness",
            "speechiness", "instrumentalness", "liveness"]
clean = (df.dropna().drop_duplicates(subset=["track_id", "track_genre"])
           .reset_index(drop=True))
clean["duration_min"] = clean["duration_ms"] / 60000
clean["artists"] = clean["artists"].str.replace(";", ", ")
profils = clean.groupby("track_genre")[features].mean()
genres = list(profils.index)                       # ordre alphabétique
gi = {g: i for i, g in enumerate(genres)}
z = (profils - profils.mean()) / profils.std()
premier = ~clean.duplicated("track_id")            # 1re occurrence d'un morceau

# --- O1 : ACP des profils de genres ---
X = z.values - z.values.mean(axis=0)
U, S, Vt = np.linalg.svd(X, full_matrices=False)
pcx, pcy = U[:, 0] * S[0], U[:, 1] * S[1]
var_exp = S**2 / (S**2).sum()
ech = 0.6 * np.abs(np.r_[pcx, pcy]).max() / np.abs(Vt[:2]).max()
pop_med = clean.groupby("track_genre")["popularity"].median().reindex(genres)
dist_house = ((z - z.loc["house"]) ** 2).sum(axis=1) ** 0.5
singuliers = pd.Series(np.hypot(pcx, pcy), index=genres).nlargest(4).index

# --- O2 : corrélations popularité × caractéristique, par genre ---
cols_corr = features + ["tempo", "loudness", "duration_min"]
pos = clean[clean["popularity"] > 0]
rho_global = pos[cols_corr].corrwith(pos["popularity"], method="spearman")
corr_g = (pos.groupby("track_genre")[cols_corr + ["popularity"]]
          .apply(lambda g: g[cols_corr].corrwith(g["popularity"], method="spearman"))
          .reindex(genres))
top30 = corr_g.abs().max(axis=1).nlargest(30).index
c30 = corr_g.loc[top30]
top30 = list(c30.index[leaves_list(linkage(c30.fillna(0).values, "ward"))])
paire = corr_g.stack().abs().idxmax()

# --- O3 : coordonnées parallèles ---
filtre = (clean["tempo"].between(120, 130) & (clean["energy"] > 0.8)
          & (clean["danceability"] > 0.7))
sel = clean[filtre & premier]
best = sel.sort_values("popularity", ascending=False).iloc[0]
echantillon = sorted(set(clean[premier].sample(8000, random_state=1).index) | set(sel.index))

# --- O4 : écart de chaque morceau au profil de son genre ---
moy = clean.groupby("track_genre")[features].transform("mean")
std = clean.groupby("track_genre")[features].transform("std")
zt = (clean[features] - moy) / std
clean["ecart"] = np.sqrt((zt ** 2).sum(axis=1))
seuil = clean["ecart"].quantile(0.99)
het12 = clean.groupby("track_genre")["ecart"].max().nlargest(12).index
i_atyp = clean["ecart"].idxmax()
atyp, cause = clean.loc[i_atyp], zt.loc[i_atyp].abs().idxmax()

# ----------------------------------------------------------------------------
# Données envoyées au navigateur (format colonnes, entiers pour alléger)
# ----------------------------------------------------------------------------
r2 = lambda a: [round(float(v), 3) for v in a]
DATA = {
    "genres": genres, "features": features, "corrCols": cols_corr,
    "pca": {"x": r2(pcx), "y": r2(pcy), "r": r2(np.hypot(pcx, pcy)),
            "load": [[round(Vt[0, j] * ech, 3), round(Vt[1, j] * ech, 3)] for j in range(len(features))],
            "var": r2(var_exp[:2])},
    "z": [r2(row) for row in z.values],
    "prof": [r2(row) for row in profils.values],
    "std": [r2(row) for row in clean.groupby("track_genre")[features].std().reindex(genres).values],
    "mean": r2(clean[premier][features].mean()),
    "popMed": [int(v) for v in pop_med],
    "corr": [[None if pd.isna(v) else round(float(v), 3) for v in row] for row in corr_g.values],
    "top30": [gi[g] for g in top30],
    "het12": [gi[g] for g in het12],
    "seuil": round(float(seuil), 2),
    "sample": [int(i) for i in echantillon],
    "tracks": {
        "g": clean["track_genre"].map(gi).tolist(),
        "n": clean["track_name"].tolist(), "a": clean["artists"].tolist(),
        "t": clean["tempo"].round().astype(int).tolist(),
        **{k: (clean[f] * 100).round().astype(int).tolist()
           for k, f in zip(["d", "e", "v", "ac", "sp", "ins", "li"], features)},
        "du": (clean["duration_min"] * 10).round().astype(int).tolist(),
        "p": clean["popularity"].astype(int).tolist(),
        "ec": (clean["ecart"] * 10).round().astype(int).tolist(),
        "u": premier.astype(int).tolist(),
    },
}
data_json = json.dumps(DATA, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")


def fr(x, d=0):
    """Nombre au format français : 114 000 ; 0,17 (espace insécable)."""
    return f"{x:,.{d}f}".replace(",", " ").replace(".", ",")


VALS = {
    "N_MORCEAUX": fr(len(df)), "N_GENRES": fr(len(genres)),
    "N_ARTISTES": fr(df["artists"].nunique()), "N_UNIQUES": fr(int(premier.sum())),
    "VAR_EXP": f"{var_exp[:2].sum():.0%}".replace("%", " %"),
    "VOISINS": ", ".join(dist_house.drop("house").nsmallest(3).index),
    "SINGULIERS": ", ".join(singuliers),
    "RHO_GLOBAL": fr(rho_global.abs().max(), 2),
    "RHO_PAIRE": fr(corr_g.loc[paire], 2), "PAIRE_G": paire[0], "PAIRE_F": paire[1],
    "N_FORT": str(int((corr_g.abs() > 0.3).sum().sum())),
    "N_SEL": fr(len(sel)), "BEST": f"{best['track_name']} — {best['artists']}",
    "BEST_POP": str(int(best["popularity"])),
    "N_ATYP": fr(int((clean["ecart"] > seuil).sum())), "SEUIL": fr(seuil, 1),
    "ATYP": f"{atyp['track_name']} — {atyp['artists']}", "ATYP_G": atyp["track_genre"],
    "CAUSE": cause, "Z_CAUSE": fr(zt.loc[i_atyp, cause], 0),
}

TEMPLATE = r"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Spotify Tracks — Visualisation des données</title>
<script>{{PLOTLYJS}}</script>
<style>
  :root { --bg:#121212; --card:#181818; --card-2:#232323; --line:#2a2a2a; --ink:#fff; --ink-2:#b3b3b3; --ink-3:#7a7a7a; --green:#1db954; }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  html { scroll-behavior: smooth; }
  body { background: var(--bg); color: var(--ink); font: 15px/1.55 -apple-system, "Helvetica Neue", Arial, sans-serif; }
  a { color: inherit; text-decoration: none; }
  button { font: inherit; cursor: pointer; border: 0; }

  /* ---------- Navigation + sélecteur de genre ---------- */
  nav { position: fixed; inset: 0 0 auto 0; z-index: 20; height: 64px; display: flex; align-items: center; gap: 22px;
        padding: 0 32px; background: rgba(18,18,18,.94); backdrop-filter: blur(10px); border-bottom: 1px solid var(--line); }
  .logo { display: flex; align-items: center; gap: 10px; font-weight: 700; white-space: nowrap; }
  .logo svg { width: 28px; height: 28px; }
  .links { display: flex; gap: 4px; }
  .links a { padding: 7px 12px; border-radius: 999px; color: var(--ink-2); font-size: 13px; font-weight: 600; transition: .15s; white-space: nowrap; }
  .links a:hover { color: var(--ink); background: var(--card-2); }
  .links a.active { color: #000; background: var(--green); }
  .picker { margin-left: auto; display: flex; align-items: center; gap: 8px; background: var(--card-2); border: 1.5px solid transparent;
            border-radius: 999px; padding: 4px 4px 4px 14px; transition: .2s; }
  .picker.on { border-color: var(--green); }
  .picker svg { width: 16px; height: 16px; flex: none; }
  .picker input { background: none; border: 0; outline: 0; color: var(--ink); font: 600 14px/1 inherit; width: 190px; }
  .picker input::placeholder { color: var(--ink-3); font-weight: 500; }
  .picker button { background: var(--ink); color: #000; border-radius: 999px; padding: 7px 13px; font-size: 12px; font-weight: 700; }
  .picker:not(.on) button { background: #3a3a3a; color: var(--ink-3); cursor: default; }

  section { min-height: 100vh; padding: 96px 40px 48px; max-width: 1280px; margin: 0 auto; scroll-margin-top: 0; }

  /* ---------- Accueil ---------- */
  .hero { display: flex; flex-direction: column; justify-content: center; }
  .eyebrow { color: var(--green); font-size: 13px; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; }
  h1 { font-size: clamp(40px, 6vw, 70px); line-height: 1.02; letter-spacing: -.035em; margin: 14px 0 16px; }
  .lead { color: var(--ink-2); font-size: 18px; max-width: 820px; }
  .lead b { color: var(--ink); }
  .kpis { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin: 30px 0 22px; }
  .kpi { background: var(--card); border-radius: 10px; padding: 18px 22px; }
  .kpi b { display: block; font-size: 32px; letter-spacing: -.02em; }
  .kpi span { color: var(--ink-2); font-size: 13px; }
  .pill { display: inline-block; background: var(--card-2); color: var(--ink); border-radius: 999px; padding: 5px 13px; font-size: 12px; font-weight: 600; white-space: nowrap; }
  .objs { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; }
  .obj { background: var(--card); border-radius: 10px; padding: 16px 20px; transition: .2s; display: block; }
  .obj:hover { background: var(--card-2); transform: translateY(-2px); }
  .obj .num { color: var(--green); font-weight: 800; font-size: 13px; }
  .obj h3 { font-size: 17px; margin: 4px 0; }
  .obj p { color: var(--ink-2); font-size: 13px; }
  .cta { margin-top: 22px; align-self: flex-start; background: var(--green); color: #000; font-weight: 700; border-radius: 999px; padding: 10px 22px; }

  /* ---------- Tableau de bord : les 4 vues sur un écran ---------- */
  #viz { height: 100vh; min-height: 760px; max-width: none; padding: 76px 18px 14px; display: flex; flex-direction: column; gap: 10px; }
  .vizbar { display: flex; align-items: center; gap: 12px; color: var(--ink-3); font-size: 12px; }
  .mode { font-size: 12px; font-weight: 700; padding: 5px 12px; border-radius: 999px; background: var(--card-2); color: var(--ink-2); }
  .has-sel .mode { background: var(--green); color: #000; }
  .board { flex: 1; min-height: 0; display: grid; grid-template-columns: 1fr 1fr; grid-template-rows: 1fr 1fr; gap: 12px; }
  .panel { background: var(--card); border-radius: 10px; padding: 10px 14px 10px; display: flex; flex-direction: column; min-height: 0; min-width: 0; }
  .ph { display: flex; align-items: baseline; gap: 10px; }
  .ph .num { color: var(--green); font-weight: 800; font-size: 13px; letter-spacing: .08em; }
  h2 { font-size: 16px; letter-spacing: -.01em; }
  .ph .tech { color: var(--ink-3); font-size: 12px; }
  .ph .hint { margin-left: auto; text-align: right; }
  .fig { flex: 1; min-height: 0; }
  .insight { font-size: 13px; color: var(--ink-2); border-left: 3px solid var(--green); padding: 1px 0 1px 10px; margin-top: 4px; }
  .insight b { color: var(--ink); }
  .fo { display: none; } .has-sel .fo { display: block; } .has-sel .ov { display: none; }
  .chips { display: inline-flex; flex-wrap: wrap; gap: 5px; vertical-align: middle; }
  .chip { background: var(--card-2); color: var(--ink); border-radius: 999px; padding: 3px 10px; font-size: 12px; font-weight: 600; transition: .15s; }
  .chip:hover { background: var(--green); color: #000; }
  .trk { color: var(--ink); font-weight: 700; border-bottom: 1px dashed var(--green); cursor: pointer; }
  .hint { color: var(--ink-3); font-size: 11.5px; }
  .hint b { color: var(--ink-2); }

  /* ---------- Tableau O3 ---------- */
  #fig3 { flex: 2; }
  .tbl { flex: 1; min-height: 0; overflow-y: auto; }
  .tbl-head { color: var(--ink-2); font-size: 12.5px; margin: 2px 0 4px; }
  .tbl-head b { color: var(--green); font-size: 15px; }
  table { width: 100%; border-collapse: collapse; font-size: 12px; }
  th { position: sticky; top: 0; background: var(--card); text-align: left; color: var(--ink-3); font-weight: 600; text-transform: uppercase; font-size: 11px; letter-spacing: .08em; padding: 4px 8px; border-bottom: 1px solid var(--line); }
  td { padding: 4px 8px; white-space: nowrap; max-width: 180px; overflow: hidden; text-overflow: ellipsis; color: var(--ink-2); border-bottom: 1px solid #202020; }
  td:first-child { color: var(--ink); font-weight: 600; }
  td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
  tbody tr { cursor: pointer; transition: .1s; }
  tbody tr:hover { background: var(--card-2); }
  tbody tr:hover td:first-child { color: var(--green); }

  /* ---------- Fiche morceau (detail) ---------- */
  .drawer { position: fixed; top: 64px; right: 0; bottom: 0; width: 400px; z-index: 30; background: #181818; border-left: 1px solid var(--line);
            box-shadow: -20px 0 40px rgba(0,0,0,.5); transform: translateX(105%); transition: transform .25s ease; padding: 22px 24px; overflow-y: auto; }
  .drawer.open { transform: none; }
  .drawer .close { position: absolute; top: 14px; right: 14px; width: 32px; height: 32px; border-radius: 50%; background: var(--card-2); color: var(--ink); font-size: 18px; }
  .drawer .kicker { color: var(--green); font-size: 11px; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; }
  .drawer h3 { font-size: 22px; letter-spacing: -.02em; line-height: 1.2; margin: 6px 40px 2px 0; }
  .drawer .artist { color: var(--ink-2); margin-bottom: 12px; }
  .stats { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin: 14px 0; }
  .stat { background: var(--card-2); border-radius: 8px; padding: 9px 12px; }
  .stat b { display: block; font-size: 18px; } .stat span { color: var(--ink-3); font-size: 11px; text-transform: uppercase; letter-spacing: .06em; }
  .flag { display: inline-block; background: #fff; color: #000; font-size: 10px; font-weight: 800; border-radius: 4px; padding: 1px 6px; margin-left: 6px; vertical-align: middle; }
  .why { font-size: 13px; color: var(--ink-2); background: var(--card-2); border-radius: 8px; padding: 10px 12px; }
  .why b { color: var(--ink); }

  footer { border-top: 1px solid var(--line); color: var(--ink-3); font-size: 12px; padding: 22px 40px; text-align: center; }
  kbd { background: var(--card-2); border-radius: 4px; padding: 1px 6px; color: var(--ink-2); }

  @media (max-width: 1100px) { .links { display: none; } }
  @media (max-width: 960px) {
    .kpis, .objs { grid-template-columns: repeat(2, 1fr); }
    section { padding: 88px 18px 32px; } .drawer { width: 100%; }
    #viz { height: auto; } .board { grid-template-columns: 1fr; grid-template-rows: none; grid-auto-rows: 480px; }
  }
</style>
</head>
<body>

<nav>
  <a class="logo" href="#intro">
    <svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="12" fill="#1db954"/>
      <path d="M6.5 9.3c3.8-1.1 8-.8 11.2 1.1M7.2 12.4c3.2-.9 6.6-.6 9.4 1M7.9 15.3c2.6-.7 5.2-.4 7.4.8" stroke="#000" stroke-width="1.6" stroke-linecap="round" fill="none"/></svg>
    Spotify Tracks · Dataviz
  </a>
  <div class="links">
    <a href="#intro">Projet</a><a href="#viz">Visualisations</a>
  </div>
  <div class="picker" id="picker">
    <svg viewBox="0 0 24 24" fill="none" stroke="#b3b3b3" stroke-width="2.4" stroke-linecap="round"><circle cx="11" cy="11" r="7"/><path d="M20 20l-4-4"/></svg>
    <input id="gsel" list="glist" placeholder="Choisir un genre…" autocomplete="off" aria-label="Sélectionner un genre">
    <button id="greset" title="Revenir à la vue globale">Vue globale</button>
  </div>
  <datalist id="glist"></datalist>
</nav>

<!-- ======================= ACCUEIL ======================= -->
<section id="intro" class="hero">
  <div class="eyebrow">UE Visualisation des données · DI FEDE Santiago · SELMANE Ilyane · TOUPIN Nicolas</div>
  <h1>Qu'est-ce qui distingue<br>les genres musicaux&nbsp;?</h1>
  <p class="lead">Quatre vues <b>reliées</b> sur un même écran. Choisissez un genre (recherche, carte ou heatmap) :
  toutes les vues se recentrent. Cliquez sur un morceau pour ouvrir sa fiche.</p>

  <div class="kpis">
    <div class="kpi"><b>{{N_MORCEAUX}}</b><span>morceaux (Kaggle)</span></div>
    <div class="kpi"><b>{{N_GENRES}}</b><span>genres · 1&nbsp;000 morceaux chacun</span></div>
    <div class="kpi"><b>{{N_ARTISTES}}</b><span>artistes</span></div>
    <div class="kpi"><b>7</b><span>caractéristiques audio (0 → 1)</span></div>
  </div>

  <div class="objs">
    <a class="obj" href="#viz"><div class="num">O1 · ACP</div><h3>Caractériser les genres</h3><p>Quels genres se ressemblent&nbsp;?</p></a>
    <a class="obj" href="#viz"><div class="num">O2 · Heatmap</div><h3>Comprendre la popularité</h3><p>Qu'est-ce qui va avec le succès&nbsp;?</p></a>
    <a class="obj" href="#viz"><div class="num">O3 · Coord. parallèles</div><h3>Construire une sélection</h3><p>Trouver des morceaux pour une ambiance.</p></a>
    <a class="obj" href="#viz"><div class="num">O4 · Strip plot</div><h3>Repérer l'atypique</h3><p>Qui sort du profil de son genre&nbsp;?</p></a>
  </div>
  <a class="cta" href="#viz">Explorer →</a>
</section>

<!-- ======================= TABLEAU DE BORD ======================= -->
<section id="viz">
  <div class="vizbar"><span class="mode"></span>
    <span>Public : curateurs de playlists, programmateurs, DJs · <kbd>Échap</kbd> ferme la fiche</span></div>
  <div class="board">

    <div class="panel">
      <div class="ph"><span class="num">O1</span><h2>La carte des genres</h2><span class="tech">ACP · {{VAR_EXP}} de la variance</span>
        <span class="hint"><b>Clic</b> : sélectionner · proches = similaires</span></div>
      <div class="fig" id="fig1"></div>
      <div class="insight"><span class="ov">Voisins de « house » : <b>{{VOISINS}}</b>. Genres les plus singuliers : <b>{{SINGULIERS}}</b>.</span>
        <span class="fo" id="fo1"></span></div>
    </div>

    <div class="panel">
      <div class="ph"><span class="num">O2</span><h2>Popularité × caractéristiques</h2><span class="tech">Spearman ρ par genre</span>
        <span class="hint"><b>Clic</b> sur une colonne : sélectionner</span></div>
      <div class="fig" id="fig2"></div>
      <div class="insight"><span class="ov">Global : |ρ| ≤ <b>{{RHO_GLOBAL}}</b>. Par genre : <b>{{N_FORT}}</b> liens &gt; 0,3
        (ρ = <b>{{RHO_PAIRE}}</b>, {{PAIRE_F}} en {{PAIRE_G}}). Pas de recette universelle.</span>
        <span class="fo" id="fo2"></span></div>
    </div>

    <div class="panel">
      <div class="ph"><span class="num">O3</span><h2>Construire une playlist</h2><span class="tech">préréglage « soirée dansante »</span>
        <span class="hint"><b>Glisser</b> sur un axe : filtrer · <b>clic</b> sur une ligne : fiche</span></div>
      <div class="fig" id="fig3"></div>
      <div class="tbl" id="tbl3"></div>
    </div>

    <div class="panel">
      <div class="ph"><span class="num">O4</span><h2>Les morceaux hors norme</h2><span class="tech">écart au profil du genre</span>
        <span class="hint"><b>Clic</b> sur un point : fiche</span></div>
      <div class="fig" id="fig4"></div>
      <div class="insight"><span class="ov"><b>{{N_ATYP}}</b> morceaux au-delà du seuil ({{SEUIL}}, 1 %). Record : <b>{{ATYP}}</b> ({{ATYP_G}}),
        {{CAUSE}} à +{{Z_CAUSE}} σ.</span><span class="fo" id="fo4"></span></div>
    </div>

  </div>
</section>

<footer>Données : <i>Spotify Tracks Dataset</i> (Kaggle, maharshipandya) · {{N_UNIQUES}} morceaux distincts après nettoyage · DI FEDE Santiago, SELMANE Ilyane, TOUPIN Nicolas</footer>

<!-- ======================= FICHE MORCEAU ======================= -->
<aside class="drawer" id="drawer" aria-label="Fiche du morceau">
  <button class="close" id="dclose" aria-label="Fermer">×</button>
  <div id="dbody"></div>
  <div id="dradar"></div>
</aside>

<script type="application/json" id="data">{{DATA}}</script>
<script>
"use strict";
const D = JSON.parse(document.getElementById("data").textContent);
const G = D.genres, F = D.features, T = D.tracks, N = T.g.length;
const FK = ["d", "e", "v", "ac", "sp", "ins", "li"];          // colonnes des 7 caractéristiques
const C = { bg: "#121212", ink: "#ffffff", ink2: "#b3b3b3", ink3: "#7a7a7a", grid: "#2a2a2a",
            green: "#1db954", blue: "#3987e5", orange: "#d95926", mid: "#3a3a3a" };
const SEQ = [[0, "#1d3a28"], [0.35, "#1a7a3c"], [0.7, "#1db954"], [1, "#b4f5c9"]];
const DIV = [[0, C.blue], [0.5, C.mid], [1, C.orange]];
const CFG = { responsive: true, displaylogo: false, modeBarButtonsToRemove: ["lasso2d", "select2d", "autoScale2d"] };

const byGenre = G.map(() => []);
for (let i = 0; i < N; i++) byGenre[T.g[i]].push(i);
const jit = i => ((i * 9301 + 49297) % 233280) / 233280 - 0.5;   // dispersion stable des points
const fx = (x, d = 2) => (x > 0 ? "+" : "") + x.toFixed(d).replace(".", ",").replace("-", "−");
const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const feat = (i, j) => T[FK[j]][i] / 100;
const zTrack = i => F.map((f, j) => (feat(i, j) - D.prof[T.g[i]][j]) / (D.std[T.g[i]][j] || 1));

const base = extra => Object.assign({
  font: { family: "-apple-system, 'Helvetica Neue', Arial, sans-serif", size: 12, color: C.ink2 },
  paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
  hoverlabel: { bgcolor: "#282828", bordercolor: "#3e3e3e", font: { color: C.ink, size: 12 } },
  margin: { l: 50, r: 20, t: 8, b: 40 }, autosize: true }, extra);
const ax = extra => Object.assign({ gridcolor: C.grid, zeroline: false, linecolor: C.grid, tickcolor: C.grid }, extra);
const cbar = title => ({ title: { text: title, font: { color: C.ink2, size: 11 } }, thickness: 8, outlinewidth: 0, len: 0.9 });

let sel = null;                       // genre sélectionné (index) ou null = vue globale

function voisins(g, k = 5) {
  return G.map((_, j) => [j, Math.hypot(...D.z[g].map((v, f) => v - D.z[j][f]))])
          .filter(([j]) => j !== g).sort((a, b) => a[1] - b[1]).slice(0, k).map(a => a[0]);
}

function radar(div, series, h) {
  const th = [...F, F[0]];
  Plotly.react(div, series.map(s => ({
    type: "scatterpolar", r: [...s.vals, s.vals[0]], theta: th, name: s.name, mode: "lines+markers",
    fill: s.fill ? "toself" : "none", line: { color: s.color, width: 2 }, marker: { size: 5 },
    hovertemplate: "%{theta} : %{r:.2f}<extra></extra>" })),
    base({ height: h, margin: { l: 70, r: 70, t: 16, b: 10 }, showlegend: true,
           legend: { orientation: "h", x: 0.5, xanchor: "center", y: -0.08, font: { size: 11, color: C.ink2 } },
           polar: { bgcolor: "rgba(0,0,0,0)",
                    radialaxis: { range: [0, 1], gridcolor: C.grid, linecolor: C.grid, showticklabels: false },
                    angularaxis: { gridcolor: C.grid, linecolor: C.grid, tickfont: { size: 10, color: C.ink } } } }),
    { displayModeBar: false, responsive: true });
}

/* ---------------- O1 : carte des genres ---------------- */
function renderO1() {
  const P = D.pca, nb = sel === null ? [] : voisins(sel);
  const on = i => sel === null || i === sel || nb.includes(i);
  const traces = nb.map(j => ({ type: "scatter", mode: "lines", x: [P.x[sel], P.x[j]], y: [P.y[sel], P.y[j]],
                                line: { color: C.ink, width: 1.5 }, hoverinfo: "skip", showlegend: false }));
  traces.push({
    type: "scatter", mode: "markers+text", x: P.x, y: P.y, hovertext: G, customdata: D.popMed,
    text: G.map((g, i) => sel === null ? (P.r[i] > 2.2 || g === "house" ? g : "") : (on(i) ? g : "")),
    textposition: "top center",
    textfont: { size: G.map((_, i) => i === sel ? 14 : 10), color: G.map((_, i) => i === sel ? C.ink : C.ink2) },
    marker: { size: G.map((_, i) => i === sel ? 20 : 11), color: D.popMed, colorscale: SEQ, cmin: 0, cmax: 70,
              opacity: G.map((_, i) => on(i) ? 1 : 0.18),
              line: { width: G.map((_, i) => i === sel ? 3 : 1.5), color: G.map((_, i) => i === sel ? C.ink : C.bg) },
              colorbar: cbar("Popularité<br>médiane") },
    hovertemplate: "<b>%{hovertext}</b><br>popularité médiane : %{customdata}<br><i>clic pour sélectionner</i><extra></extra>",
    showlegend: false });
  const ann = [];
  F.forEach((f, j) => {
    const [x, y] = P.load[j];
    ann.push({ x, y, ax: 0, ay: 0, xref: "x", yref: "y", axref: "x", ayref: "y", text: "", showarrow: true,
               arrowhead: 2, arrowwidth: 1.5, arrowcolor: "#5a5a5a" });
    ann.push({ x: x * 1.1, y: y * 1.1, text: "<b>" + f + "</b>", showarrow: false, font: { size: 11, color: C.ink },
               bgcolor: "rgba(18,18,18,.75)" });
  });
  Plotly.react("fig1", traces, base({ annotations: ann, uirevision: "o1",
    xaxis: ax({ title: { text: "CP1 · " + Math.round(P.var[0] * 100) + " %", standoff: 4 } }),
    yaxis: ax({ title: { text: "CP2 · " + Math.round(P.var[1] * 100) + " %", standoff: 4 } }) }), CFG);

  if (sel !== null) {
    const rang = D.popMed.filter(v => v > D.popMed[sel]).length + 1;
    document.getElementById("fo1").innerHTML =
      `<b>${esc(G[sel])}</b> · popularité ${D.popMed[sel]} (rang ${rang}/${G.length}) · voisins :
       <span class="chips">${nb.map(j => `<button class="chip" data-g="${j}">${esc(G[j])}</button>`).join("")}</span>`;
  }
}

/* ---------------- O2 : heatmap des corrélations ---------------- */
function renderO2() {
  let cols = D.top30.slice();
  if (sel !== null && !cols.includes(sel)) cols = [sel, ...cols];
  const zz = D.corrCols.map((_, r) => cols.map(g => D.corr[g][r]));
  const shapes = [];
  if (sel !== null) {
    const k = cols.indexOf(sel);
    shapes.push({ type: "rect", xref: "x", yref: "y", x0: k - 0.5, x1: k + 0.5, y0: -0.5, y1: D.corrCols.length - 0.5,
                  line: { color: C.ink, width: 2.5 } });
  }
  Plotly.react("fig2", [{
    type: "heatmap", z: zz, x: cols.map(g => G[g]), y: D.corrCols, colorscale: DIV, zmid: 0, zmin: -0.6, zmax: 0.6,
    xgap: 2, ygap: 2, hoverongaps: false,
    colorbar: Object.assign(cbar("ρ"), { tickvals: [-0.6, -0.3, 0, 0.3, 0.6] }),
    hovertemplate: "<b>%{x}</b><br>%{y} ↔ popularité<br>ρ = %{z:.2f}<br><i>clic pour sélectionner</i><extra></extra>" }],
    base({ shapes, margin: { l: 100, r: 10, t: 8, b: 80 },
           xaxis: ax({ tickangle: -55, showgrid: false, tickfont: { size: 10 } }),
           yaxis: ax({ showgrid: false, tickfont: { size: 11, color: C.ink } }) }), CFG);

  if (sel !== null) {
    const c = D.corr[sel].map((v, r) => [v, D.corrCols[r]]).filter(([v]) => v !== null).sort((a, b) => b[0] - a[0]);
    let txt;
    if (!c.length) txt = "pas assez de morceaux notés.";
    else {
      const [hi, lo] = [c[0], c[c.length - 1]];
      const parts = [];
      if (hi[0] >= 0.1) parts.push(`monte avec <b>${hi[1]}</b> (ρ = ${fx(hi[0])})`);
      if (lo[0] <= -0.1) parts.push(`baisse avec <b>${lo[1]}</b> (ρ = ${fx(lo[0])})`);
      txt = parts.length ? "la popularité " + parts.join(" et ") + "."
                         : "aucun lien net avec la popularité (|ρ| &lt; 0,1).";
    }
    document.getElementById("fo2").innerHTML = `<b>${esc(G[sel])}</b> (colonne encadrée) : ${txt}`;
  }
}

/* ---------------- O3 : coordonnées parallèles + tableau ---------------- */
const DIMS = [
  { k: "t", label: "Tempo (BPM)", range: [40, 220], s: 1 },
  { k: "e", label: "Énergie", range: [0, 1], s: 0.01 },
  { k: "d", label: "Danceability", range: [0, 1], s: 0.01 },
  { k: "v", label: "Valence", range: [0, 1], s: 0.01 },
  { k: "ac", label: "Acousticness", range: [0, 1], s: 0.01 },
  { k: "du", label: "Durée (min)", range: [0, 10], s: 0.1, max: 10 },
  { k: "p", label: "Popularité", range: [0, 100], s: 1 }];
const val = (d, i) => Math.min(T[d.k][i] * d.s, d.max ?? Infinity);
let cons = { 0: [120, 130], 1: [0.8, 1], 2: [0.7, 1] };           // préréglage « soirée dansante »
const inR = (v, cr) => (Array.isArray(cr[0]) ? cr : [cr]).some(([a, b]) => v >= a && v <= b);

function renderO3() {
  const rows = sel === null ? D.sample : byGenre[sel];
  Plotly.react("fig3", [{
    type: "parcoords",
    line: { color: rows.map(i => T.p[i]), colorscale: SEQ, cmin: 0, cmax: 100, showscale: true, colorbar: cbar("Popularité") },
    dimensions: DIMS.map((d, k) => Object.assign({ label: d.label, range: d.range, values: rows.map(i => val(d, i)) },
                                                  cons[k] ? { constraintrange: cons[k] } : {})),
    unselected: { line: { color: "#3a3a3a", opacity: sel === null ? 0.06 : 0.15 } },
    labelfont: { size: 11, color: C.ink }, tickfont: { size: 9, color: C.ink2 }, rangefont: { size: 9, color: "#6a6a6a" } }],
    base({ margin: { l: 50, r: 60, t: 46, b: 14 } }), CFG);
  table3();
}

function table3() {
  const pool = sel === null ? byGenre.flat().filter(i => T.u[i]) : byGenre[sel];
  const keys = Object.keys(cons);
  const ok = pool.filter(i => keys.every(k => inR(val(DIMS[k], i), cons[k]))).sort((a, b) => T.p[b] - T.p[a]);
  const filtres = keys.length ? keys.map(k => {
      const cr = Array.isArray(cons[k][0]) ? cons[k] : [cons[k]];
      return DIMS[k].label + " " + cr.map(([a, b]) => `${+a.toFixed(2)}–${+b.toFixed(2)}`).join(" ou ");
    }).join(" · ") : "aucun filtre";
  document.getElementById("tbl3").innerHTML =
    `<div class="tbl-head"><b>${ok.length.toLocaleString("fr-FR")}</b> morceaux correspondent
       ${sel === null ? "dans le catalogue" : "en <b style='font-size:13px'>" + esc(G[sel]) + "</b>"} · <span style="color:var(--ink-3)">${filtres}</span></div>
     <table><thead><tr><th>Titre</th><th>Artiste</th><th>Genre</th><th class="num">BPM</th><th class="num">Énergie</th><th class="num">Valence</th><th class="num">Popularité</th></tr></thead>
     <tbody>${ok.slice(0, 30).map(i => `<tr data-i="${i}"><td>${esc(T.n[i])}</td><td>${esc(T.a[i])}</td><td>${esc(G[T.g[i]])}</td>
       <td class="num">${T.t[i]}</td><td class="num">${(T.e[i] / 100).toFixed(2)}</td><td class="num">${(T.v[i] / 100).toFixed(2)}</td><td class="num">${T.p[i]}</td></tr>`).join("")}</tbody></table>
     ${ok.length > 30 ? `<div class="hint">30 plus populaires affichés</div>` : ""}`;
}

/* ---------------- O4 : strip plot des écarts ---------------- */
function renderO4() {
  const shown = sel === null ? D.het12 : [sel];
  const w = sel === null ? 0.7 : 0.9;
  const xs = [], ys = [], cd = [], best = [];
  shown.forEach((g, k) => {
    const rows = byGenre[g];
    rows.forEach(i => { xs.push(k + jit(i) * w); ys.push(T.ec[i] / 10); cd.push(i); });
    const tri = rows.slice().sort((a, b) => T.ec[b] - T.ec[a]);
    if (sel === null) best.push(tri[0]); else best.push(...tri.slice(0, 5));
  });
  const lab = sel === null ? best.slice().sort((a, b) => T.ec[b] - T.ec[a]).slice(0, 5) : best.slice(0, 3);
  const pos = i => shown.indexOf(T.g[i]) + jit(i) * w;
  Plotly.react("fig4", [
    { type: "scattergl", mode: "markers", x: xs, y: ys, customdata: cd,
      marker: { size: sel === null ? 5 : 7, color: C.green, opacity: sel === null ? 0.35 : 0.5 },
      text: cd.map(i => esc(T.n[i]) + "<br>" + esc(T.a[i])),
      hovertemplate: "<b>%{text}</b><br>écart : %{y:.1f}<br><i>clic : fiche du morceau</i><extra></extra>", showlegend: false },
    { type: "scatter", mode: "markers+text", x: best.map(pos), y: best.map(i => T.ec[i] / 10), customdata: best,
      text: best.map(i => lab.includes(i) ? esc(T.n[i]).slice(0, 20) : ""), textposition: "middle right",
      textfont: { size: 10, color: C.ink },
      marker: { size: 11, color: C.ink, line: { width: 2, color: C.bg } },
      hovertext: best.map(i => esc(T.n[i]) + "<br>" + esc(T.a[i])),
      hovertemplate: "<b>%{hovertext}</b><br>écart : %{y:.1f}<br><i>clic : fiche du morceau</i><extra></extra>", showlegend: false }],
    base({ margin: { l: 50, r: 110, t: 8, b: 56 },
      shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: D.seuil, y1: D.seuil, line: { color: "#6a6a6a", width: 1, dash: "dot" } }],
      annotations: [{ xref: "paper", x: 0, y: D.seuil, yanchor: "bottom", xanchor: "left", showarrow: false,
                      text: "seuil 99ᵉ percentile", font: { size: 11, color: C.ink2 } }],
      xaxis: ax({ tickvals: shown.map((_, k) => k), ticktext: shown.map(g => G[g]), range: [-0.6, shown.length - 0.4],
                  showgrid: false, tickangle: sel === null ? -30 : 0, tickfont: { size: sel === null ? 10 : 13, color: C.ink } }),
      yaxis: ax({ title: { text: "Écart (z)", standoff: 4 }, rangemode: "tozero" }) }), CFG);

  if (sel !== null) {
    const rows = byGenre[sel], n = rows.filter(i => T.ec[i] / 10 > D.seuil).length;
    const top = best[0], z = zTrack(top), j = z.reduce((m, v, k) => Math.abs(v) > Math.abs(z[m]) ? k : m, 0);
    document.getElementById("fo4").innerHTML = `<b>${esc(G[sel])}</b> : ${n}/${rows.length} au-delà du seuil. Record :
      <span class="trk" data-i="${top}">${esc(T.n[top])}</span>, ${F[j]} à ${fx(z[j], 1)} σ.`;
  }
}

/* ---------------- Fiche morceau (détail) ---------------- */
function card(i) {
  const g = T.g[i], z = zTrack(i), j = z.reduce((m, v, k) => Math.abs(v) > Math.abs(z[m]) ? k : m, 0);
  const atyp = T.ec[i] / 10 > D.seuil;
  document.getElementById("dbody").innerHTML = `
    <div class="kicker">Fiche morceau</div>
    <h3>${esc(T.n[i])}</h3><div class="artist">${esc(T.a[i])}</div>
    <button class="chip" data-g="${g}">Genre : ${esc(G[g])} →</button>
    <div class="stats">
      <div class="stat"><b>${T.p[i]}<small style="color:var(--ink-3);font-size:12px">/100</small></b><span>Popularité</span></div>
      <div class="stat"><b>${T.t[i]}</b><span>Tempo (BPM)</span></div>
      <div class="stat"><b>${(T.du[i] / 10).toFixed(1).replace(".", ",")} min</b><span>Durée</span></div>
      <div class="stat"><b>${(T.ec[i] / 10).toFixed(1).replace(".", ",")}${atyp ? '<span class="flag">ATYPIQUE</span>' : ""}</b><span>Écart au genre</span></div>
    </div>
    <div class="why">Trait le plus inhabituel pour un morceau de <b>${esc(G[g])}</b> :
      <b>${F[j]}</b> = ${feat(i, j).toFixed(2).replace(".", ",")} (moyenne du genre ${D.prof[g][j].toFixed(2).replace(".", ",")}, z = ${fx(z[j], 1)}).</div>`;
  document.getElementById("drawer").classList.add("open");
  radar("dradar", [{ name: "moyenne " + G[g], vals: D.prof[g], color: C.green, fill: true },
                   { name: "ce morceau", vals: F.map((_, k) => feat(i, k)), color: C.ink, fill: false }], 330);
}
const closeCard = () => document.getElementById("drawer").classList.remove("open");

/* ---------------- Sélection partagée ---------------- */
const input = document.getElementById("gsel");
function select(g) {
  sel = g;
  document.body.classList.toggle("has-sel", g !== null);
  document.getElementById("picker").classList.toggle("on", g !== null);
  input.value = g === null ? "" : G[g];
  document.querySelectorAll(".mode").forEach(m => m.textContent = g === null ? "Vue globale · tous les genres" : "Genre : " + G[g]);
  renderO1(); renderO2(); renderO3(); renderO4();
}

document.getElementById("glist").innerHTML = G.map(g => `<option value="${esc(g)}">`).join("");
input.addEventListener("input", () => { const k = G.indexOf(input.value.trim().toLowerCase()); if (k >= 0 && k !== sel) select(k); });
input.addEventListener("keydown", e => { if (e.key === "Enter") { const k = G.findIndex(g => g.startsWith(input.value.trim().toLowerCase())); if (k >= 0) select(k); input.blur(); } });
document.getElementById("greset").addEventListener("click", () => select(null));
document.getElementById("dclose").addEventListener("click", closeCard);
document.addEventListener("click", e => {
  const chip = e.target.closest("[data-g]"); if (chip) { select(+chip.dataset.g); return; }
  const trk = e.target.closest("[data-i]"); if (trk) card(+trk.dataset.i);
});

select(null);
// chaque graphique suit la taille de son panneau
const ro = new ResizeObserver(es => es.forEach(e => e.target.data && Plotly.Plots.resize(e.target)));
document.querySelectorAll(".fig").forEach(f => ro.observe(f));
document.getElementById("fig1").on("plotly_click", e => { const p = e.points[0]; if (p.data.mode === "markers+text") select(p.pointIndex); });
document.getElementById("fig2").on("plotly_click", e => select(G.indexOf(e.points[0].x)));
document.getElementById("fig4").on("plotly_click", e => card(e.points[0].customdata));
document.getElementById("fig3").on("plotly_restyle", () => {
  const dims = document.getElementById("fig3").data[0].dimensions; cons = {};
  dims.forEach((d, k) => { if (d.constraintrange && d.constraintrange.length) cons[k] = d.constraintrange; });
  table3();
});

/* ---------------- Navigation ---------------- */
const ids = ["intro", "viz"], links = [...document.querySelectorAll(".links a")];
const courant = () => { let i = 0; ids.forEach((id, k) => { if (document.getElementById(id).getBoundingClientRect().top < innerHeight * 0.4) i = k; }); return i; };
const maj = () => links.forEach((a, k) => a.classList.toggle("active", k === courant()));
addEventListener("scroll", maj, { passive: true }); maj();
addEventListener("keydown", e => {
  if (e.key === "Escape") { closeCard(); return; }
  if (e.target.tagName === "INPUT") return;
  const d = { ArrowRight: 1, ArrowDown: 1, PageDown: 1, ArrowLeft: -1, ArrowUp: -1, PageUp: -1 }[e.key];
  if (!d) return;
  e.preventDefault();
  document.getElementById(ids[Math.min(ids.length - 1, Math.max(0, courant() + d))]).scrollIntoView({ behavior: "smooth" });
});
</script>
</body>
</html>
"""

html = TEMPLATE
for k, v in VALS.items():
    html = html.replace("{{" + k + "}}", v)
assert html.count("{{") == 2, "placeholder non remplacé"   # restent DATA et PLOTLYJS
html = html.replace("{{DATA}}", data_json)
html = html.replace("{{PLOTLYJS}}", get_plotlyjs())     # en dernier : plotly.js embarqué (hors ligne)
(ICI / "presentation_spotify.html").write_text(html, encoding="utf-8")

print(f"presentation_spotify.html générée ({len(html) / 1e6:.1f} Mo)")
for k, v in VALS.items():
    print(f"  {k}: {v}")
