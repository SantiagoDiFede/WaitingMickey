"""Génère presentation_spotify.html : interface de présentation (thème Spotify)
à partir des 4 visualisations du notebook projet_spotify_visualisation.ipynb.

Usage : .venv/bin/python build_presentation.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.offline import get_plotlyjs
from scipy.cluster.hierarchy import linkage, leaves_list

ICI = Path(__file__).parent

# ----------------------------------------------------------------------------
# Thème Spotify
# ----------------------------------------------------------------------------
BG, CARD, INK, INK2, GRID = "#121212", "#181818", "#ffffff", "#b3b3b3", "#2a2a2a"
GREEN, BLUE, ORANGE, MID = "#1db954", "#3987e5", "#d95926", "#3a3a3a"
SEQ = [[0, "#1d3a28"], [0.35, "#1a7a3c"], [0.7, "#1db954"], [1, "#b4f5c9"]]
DIV = [[0, BLUE], [0.5, MID], [1, ORANGE]]

pio.templates["spotify"] = go.layout.Template(layout=dict(
    font=dict(family="-apple-system, 'Helvetica Neue', Arial, sans-serif", size=12, color=INK2),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(gridcolor=GRID, zeroline=False, linecolor=GRID, tickcolor=GRID),
    yaxis=dict(gridcolor=GRID, zeroline=False, linecolor=GRID, tickcolor=GRID),
    hoverlabel=dict(bgcolor="#282828", bordercolor="#3e3e3e", font=dict(color=INK, size=12)),
    margin=dict(l=60, r=30, t=20, b=50),
))
pio.templates.default = "spotify"

# ----------------------------------------------------------------------------
# Données (même préparation que le notebook)
# ----------------------------------------------------------------------------
df = pd.read_csv(ICI / "dataset.csv", index_col=0)
features = ["danceability", "energy", "valence", "acousticness",
            "speechiness", "instrumentalness", "liveness"]
clean = df.dropna().drop_duplicates(subset=["track_id", "track_genre"]).copy()
clean["duration_min"] = clean["duration_ms"] / 60000
clean["artists"] = clean["artists"].str.replace(";", ", ")
profils = clean.groupby("track_genre")[features + ["tempo", "popularity"]].mean()
z = (profils[features] - profils[features].mean()) / profils[features].std()
uniques = clean.drop_duplicates("track_id")

KPI = dict(morceaux=len(df), genres=df["track_genre"].nunique(),
           artistes=df["artists"].nunique(), uniques=len(uniques))

# ----------------------------------------------------------------------------
# O1 — Carte des genres (ACP + biplot)
# ----------------------------------------------------------------------------
X = z.values - z.values.mean(axis=0)
U, S, Vt = np.linalg.svd(X, full_matrices=False)
coords = pd.DataFrame(U[:, :2] * S[:2], index=z.index, columns=["PC1", "PC2"])
var_exp = S**2 / (S**2).sum()
pop_med = clean.groupby("track_genre")["popularity"].median()
ref = "house"
dist = ((z - z.loc[ref]) ** 2).sum(axis=1) ** 0.5
voisins = dist.drop(ref).nsmallest(5).index
rayon = np.hypot(coords["PC1"], coords["PC2"])
singuliers = rayon.nlargest(4).index

fig1 = go.Figure()
for v in voisins:
    fig1.add_trace(go.Scatter(x=coords.loc[[ref, v], "PC1"], y=coords.loc[[ref, v], "PC2"],
                              mode="lines", line=dict(color=INK, width=1.5),
                              hoverinfo="skip", showlegend=False))
etiq = [g if (r > 2.2 or g == ref or g in voisins) else "" for g, r in rayon.items()]
fig1.add_trace(go.Scatter(
    x=coords["PC1"], y=coords["PC2"], mode="markers+text", text=etiq,
    textposition="top center", textfont=dict(size=10, color=INK2),
    marker=dict(size=11, color=pop_med.loc[coords.index], colorscale=SEQ,
                line=dict(width=1.5, color=BG),
                colorbar=dict(title=dict(text="Popularité<br>médiane", font_color=INK2),
                              thickness=10, outlinewidth=0)),
    hovertext=coords.index, customdata=pop_med.loc[coords.index],
    hovertemplate="<b>%{hovertext}</b><br>popularité médiane : %{customdata}<extra></extra>",
    showlegend=False,
))
ech = 0.6 * np.abs(coords.values).max() / np.abs(Vt[:2]).max()
for j, f in enumerate(features):
    fx, fy = Vt[0, j] * ech, Vt[1, j] * ech
    fig1.add_annotation(x=fx, y=fy, ax=0, ay=0, xref="x", yref="y", axref="x", ayref="y",
                        text="", showarrow=True, arrowhead=2, arrowwidth=1.5, arrowcolor="#6a6a6a")
    fig1.add_annotation(x=fx * 1.1, y=fy * 1.1, text=f"<b>{f}</b>", showarrow=False,
                        font=dict(size=11, color=INK), bgcolor="rgba(18,18,18,0.75)")
fig1.update_layout(height=620,
                   xaxis_title=f"Composante 1 · {var_exp[0]:.0%} de la variance",
                   yaxis_title=f"Composante 2 · {var_exp[1]:.0%} de la variance")

# ----------------------------------------------------------------------------
# O2 — Heatmap des corrélations popularité × caractéristique, par genre
# ----------------------------------------------------------------------------
cols_corr = features + ["tempo", "loudness", "duration_min"]
pos = clean[clean["popularity"] > 0]
rho_global = pos[cols_corr].corrwith(pos["popularity"], method="spearman")
corr_g = (pos.groupby("track_genre")[cols_corr + ["popularity"]]
          .apply(lambda g: g[cols_corr].corrwith(g["popularity"], method="spearman"))
          .dropna(how="all"))
top_g = corr_g.abs().max(axis=1).nlargest(30).index
cg = corr_g.loc[top_g]
cg = cg.loc[cg.index[leaves_list(linkage(cg.fillna(0).values, "ward"))]]
paire = corr_g.stack().abs().idxmax()
rho_paire = corr_g.loc[paire]
n_fort = int((corr_g.abs() > 0.3).sum().sum())

fig2 = go.Figure(go.Heatmap(
    z=cg.T.values, x=list(cg.index), y=cols_corr,
    colorscale=DIV, zmid=0, zmin=-0.6, zmax=0.6, xgap=2, ygap=2,
    colorbar=dict(title=dict(text="ρ", font_color=INK2), thickness=10, outlinewidth=0,
                  tickvals=[-0.6, -0.3, 0, 0.3, 0.6]),
    hovertemplate="<b>%{x}</b><br>%{y} ↔ popularité<br>ρ = %{z:.2f}<extra></extra>",
))
fig2.update_layout(height=470, margin=dict(l=120, b=110),
                   xaxis=dict(tickangle=-50, showgrid=False, tickfont=dict(size=11)),
                   yaxis=dict(showgrid=False, tickfont=dict(size=12, color=INK)))

# ----------------------------------------------------------------------------
# O3 — Coordonnées parallèles avec brushing
# ----------------------------------------------------------------------------
filtre = (clean["tempo"].between(120, 130) & (clean["energy"] > 0.8)
          & (clean["danceability"] > 0.7))
sel = clean[filtre].drop_duplicates("track_id")
best = sel.sort_values("popularity", ascending=False).iloc[0]
pc = pd.concat([uniques.sample(8000, random_state=1), sel]).drop_duplicates("track_id")

fig3 = go.Figure(go.Parcoords(
    line=dict(color=pc["popularity"], colorscale=SEQ, showscale=True,
              colorbar=dict(title=dict(text="Popularité", font_color=INK2),
                            thickness=10, outlinewidth=0)),
    dimensions=[
        dict(label="Tempo (BPM)", values=pc["tempo"], range=[40, 220], constraintrange=[120, 130]),
        dict(label="Énergie", values=pc["energy"], range=[0, 1], constraintrange=[0.8, 1]),
        dict(label="Danceability", values=pc["danceability"], range=[0, 1], constraintrange=[0.7, 1]),
        dict(label="Valence", values=pc["valence"], range=[0, 1]),
        dict(label="Acousticness", values=pc["acousticness"], range=[0, 1]),
        dict(label="Durée (min)", values=pc["duration_min"].clip(upper=10), range=[0, 10]),
        dict(label="Popularité", values=pc["popularity"], range=[0, 100]),
    ],
    unselected=dict(line=dict(color="#3a3a3a", opacity=0.06)),
    labelfont=dict(size=13, color=INK), tickfont=dict(size=10, color=INK2),
    rangefont=dict(size=9, color="#6a6a6a"),
))
fig3.update_layout(height=500, margin=dict(l=60, r=70, t=60, b=30))

# ----------------------------------------------------------------------------
# O4 — Strip plot des écarts au profil du genre
# ----------------------------------------------------------------------------
moy = clean.groupby("track_genre")[features].transform("mean")
std = clean.groupby("track_genre")[features].transform("std")
zt = (clean[features] - moy) / std
clean["ecart"] = (zt ** 2).sum(axis=1) ** 0.5
genres_t8 = clean.groupby("track_genre")["ecart"].max().nlargest(12).index
sub8 = clean[clean["track_genre"].isin(genres_t8)]
seuil = clean["ecart"].quantile(0.99)
n_atyp = int((clean["ecart"] > seuil).sum())
top8 = sub8.loc[sub8.groupby("track_genre")["ecart"].idxmax()]
atyp = clean.loc[clean["ecart"].idxmax()]
cause = zt.loc[clean["ecart"].idxmax()].abs().idxmax()
z_cause = zt.loc[clean["ecart"].idxmax(), cause]

fig4 = go.Figure()
fig4.add_trace(go.Box(
    x=sub8["track_genre"], y=sub8["ecart"], boxpoints="all", jitter=0.7, pointpos=0,
    fillcolor="rgba(0,0,0,0)", line=dict(width=0),
    marker=dict(size=4, color=GREEN, opacity=0.35),
    customdata=sub8[["track_name", "artists"]].values,
    hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}<br>écart : %{y:.1f}<extra></extra>",
    showlegend=False,
))
fig4.add_trace(go.Scatter(
    x=top8["track_genre"], y=top8["ecart"], mode="markers+text",
    text=[t[:24] if e >= top8["ecart"].nlargest(5).min() else ""
          for t, e in zip(top8["track_name"], top8["ecart"])],
    textposition="middle right",
    textfont=dict(size=10, color=INK),
    marker=dict(size=10, color=INK, line=dict(width=2, color=BG)),
    customdata=top8[["track_name", "artists"]].values,
    hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}<br>écart : %{y:.1f}<extra></extra>",
    showlegend=False,
))
fig4.add_hline(y=seuil, line=dict(color="#6a6a6a", width=1, dash="dot"),
               annotation_text="seuil 99ᵉ percentile", annotation_position="top left",
               annotation_font=dict(color=INK2, size=11))
fig4.update_layout(height=560, margin=dict(r=150),
                   xaxis=dict(categoryorder="array", categoryarray=list(genres_t8),
                              tickangle=-30, showgrid=False, tickfont=dict(color=INK, size=12)),
                   yaxis=dict(title="Écart au profil du genre (z)"))

# ----------------------------------------------------------------------------
# Assemblage HTML
# ----------------------------------------------------------------------------
CFG = dict(responsive=True, displaylogo=False,
           modeBarButtonsToRemove=["lasso2d", "select2d", "autoScale2d"])


def div(fig, id_):
    return pio.to_html(fig, full_html=False, include_plotlyjs=False, div_id=id_, config=CFG)


def fr(x, d=0):
    """Nombre au format français : 114 000 ; 0,17 (espace insécable)."""
    return f"{x:,.{d}f}".replace(",", "\u00a0").replace(".", ",")


VALS = {
    "N_MORCEAUX": fr(KPI["morceaux"]), "N_GENRES": fr(KPI["genres"]),
    "N_ARTISTES": fr(KPI["artistes"]), "N_UNIQUES": fr(KPI["uniques"]),
    "FIG1": div(fig1, "fig1"), "FIG2": div(fig2, "fig2"),
    "FIG3": div(fig3, "fig3"), "FIG4": div(fig4, "fig4"),
    "VAR_EXP": f"{var_exp[:2].sum():.0%}".replace("%", " %"),
    "VOISINS": ", ".join(voisins[:3]),
    "SINGULIERS": ", ".join(singuliers),
    "RHO_GLOBAL": fr(rho_global.abs().max(), 2),
    "RHO_PAIRE": fr(rho_paire, 2), "PAIRE_G": paire[0], "PAIRE_F": paire[1],
    "N_FORT": str(n_fort),
    "N_SEL": fr(len(sel)), "BEST": f"{best['track_name']} — {best['artists']}",
    "BEST_POP": str(int(best["popularity"])),
    "N_ATYP": fr(n_atyp), "SEUIL": fr(seuil, 1),
    "ATYP": f"{atyp['track_name']} — {atyp['artists']}", "ATYP_G": atyp["track_genre"],
    "CAUSE": cause, "Z_CAUSE": fr(z_cause, 0),
}

TEMPLATE = r"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Spotify Tracks — Visualisation des données</title>
<script>{{PLOTLYJS}}</script>
<style>
  :root {
    --bg: #121212; --card: #181818; --card-2: #232323; --line: #2a2a2a;
    --ink: #ffffff; --ink-2: #b3b3b3; --ink-3: #7a7a7a; --green: #1db954;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  html { scroll-behavior: smooth; scroll-snap-type: y proximity; }
  body { background: var(--bg); color: var(--ink); font: 15px/1.55 -apple-system, "Helvetica Neue", Arial, sans-serif; }
  a { color: inherit; text-decoration: none; }

  /* Navigation */
  nav { position: fixed; inset: 0 0 auto 0; z-index: 10; height: 60px; display: flex; align-items: center; gap: 28px;
        padding: 0 40px; background: rgba(18,18,18,.92); backdrop-filter: blur(10px); border-bottom: 1px solid var(--line); }
  .logo { display: flex; align-items: center; gap: 10px; font-weight: 700; letter-spacing: -.01em; }
  .logo svg { width: 28px; height: 28px; }
  nav .links { display: flex; gap: 6px; margin-left: auto; }
  nav .links a { padding: 7px 14px; border-radius: 999px; color: var(--ink-2); font-size: 13px; font-weight: 600; transition: .15s; }
  nav .links a:hover { color: var(--ink); background: var(--card-2); }
  nav .links a.active { color: #000; background: var(--green); }

  section { min-height: 100vh; padding: 92px 40px 48px; scroll-snap-align: start; max-width: 1280px; margin: 0 auto; }

  /* Hero */
  .hero { display: flex; flex-direction: column; justify-content: center; }
  .eyebrow { color: var(--green); font-size: 13px; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; }
  h1 { font-size: clamp(40px, 6vw, 72px); line-height: 1.02; letter-spacing: -.035em; margin: 14px 0 18px; }
  .lead { color: var(--ink-2); font-size: 19px; max-width: 760px; }
  .kpis { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin: 40px 0 34px; }
  .kpi { background: var(--card); border-radius: 10px; padding: 20px 22px; }
  .kpi b { display: block; font-size: 34px; letter-spacing: -.02em; }
  .kpi span { color: var(--ink-2); font-size: 13px; }
  .public { display: flex; gap: 14px; align-items: center; color: var(--ink-2); margin-bottom: 26px; }
  .public b { color: var(--ink); }
  .pill { display: inline-block; background: var(--card-2); color: var(--ink); border-radius: 999px; padding: 5px 13px; font-size: 12px; font-weight: 600; }
  .objs { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; }
  .obj { background: var(--card); border-radius: 10px; padding: 20px; transition: .2s; display: block; }
  .obj:hover { background: var(--card-2); transform: translateY(-2px); }
  .obj .num { color: var(--green); font-weight: 800; font-size: 13px; }
  .obj h3 { font-size: 17px; margin: 6px 0 6px; }
  .obj p { color: var(--ink-2); font-size: 13px; }
  .obj .tech { margin-top: 12px; font-size: 12px; color: var(--ink-3); }

  /* Sections objectif */
  .head { display: flex; align-items: baseline; gap: 18px; margin-bottom: 6px; }
  .head .num { color: var(--green); font-weight: 800; font-size: 15px; letter-spacing: .08em; }
  h2 { font-size: clamp(28px, 3.4vw, 40px); letter-spacing: -.03em; line-height: 1.1; }
  .question { color: var(--ink-2); font-size: 18px; margin: 8px 0 18px; }
  .tags { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 20px; }
  .tag-tech { background: var(--green); color: #000; }
  .grid3 { display: grid; grid-template-columns: 1fr 1fr 1.15fr; gap: 14px; margin-bottom: 14px; }
  .box { background: var(--card); border-radius: 10px; padding: 16px 18px; font-size: 14px; color: var(--ink-2); }
  .box h4 { color: var(--ink); font-size: 12px; text-transform: uppercase; letter-spacing: .1em; margin-bottom: 8px; }
  .box ul { padding-left: 16px; } .box li { margin: 3px 0; }
  .box b { color: var(--ink); }
  .box.key { background: linear-gradient(135deg, rgba(29,185,84,.22), rgba(29,185,84,.05)); border: 1px solid rgba(29,185,84,.35); }
  .box.key h4 { color: var(--green); }
  .chart { background: var(--card); border-radius: 10px; padding: 14px 10px 6px; }
  .hint { color: var(--ink-3); font-size: 12px; padding: 4px 10px 6px; }

  footer { border-top: 1px solid var(--line); color: var(--ink-3); font-size: 12px; padding: 22px 40px; text-align: center; }
  .keys { position: fixed; right: 20px; bottom: 16px; color: var(--ink-3); font-size: 11px; }
  kbd { background: var(--card-2); border-radius: 4px; padding: 1px 6px; color: var(--ink-2); }

  @media (max-width: 960px) {
    .kpis, .objs { grid-template-columns: repeat(2, 1fr); } .grid3 { grid-template-columns: 1fr; }
    nav .links a { padding: 6px 9px; } section { padding: 84px 18px 32px; }
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
    <a href="#intro">Projet</a><a href="#o1">O1 Genres</a><a href="#o2">O2 Popularité</a>
    <a href="#o3">O3 Sélection</a><a href="#o4">O4 Atypiques</a>
  </div>
</nav>

<!-- ======================= INTRO ======================= -->
<section id="intro" class="hero">
  <div class="eyebrow">UE Visualisation des données · DI FEDE Santiago · SELMANE Ilyane · TOUPIN Nicolas</div>
  <h1>Qu'est-ce qui distingue<br>les genres musicaux&nbsp;?</h1>
  <p class="lead">Quatre visualisations interactives pour comprendre le profil sonore des genres, ce qui rend un morceau populaire,
  et pour construire une sélection à partir des caractéristiques audio calculées par Spotify.</p>

  <div class="kpis">
    <div class="kpi"><b>{{N_MORCEAUX}}</b><span>morceaux (Kaggle)</span></div>
    <div class="kpi"><b>{{N_GENRES}}</b><span>genres · 1&nbsp;000 morceaux chacun</span></div>
    <div class="kpi"><b>{{N_ARTISTES}}</b><span>artistes</span></div>
    <div class="kpi"><b>7</b><span>caractéristiques audio analysées (0 → 1)</span></div>
  </div>

  <div class="public"><span class="pill">Public cible</span>
    <span><b>Curateurs de playlists, programmateurs, DJs</b> : des passionnés de musique, non statisticiens, qui veulent explorer, filtrer et comparer.</span></div>

  <div class="objs">
    <a class="obj" href="#o1"><div class="num">O1</div><h3>Caractériser les genres</h3>
      <p>Quel est le profil sonore de chaque genre et lesquels se ressemblent&nbsp;?</p><div class="tech">→ Carte ACP + biplot</div></a>
    <a class="obj" href="#o2"><div class="num">O2</div><h3>Comprendre la popularité</h3>
      <p>Quelles caractéristiques vont avec les morceaux les plus écoutés&nbsp;?</p><div class="tech">→ Heatmap de corrélations</div></a>
    <a class="obj" href="#o3"><div class="num">O3</div><h3>Construire une sélection</h3>
      <p>Comment trouver vite des morceaux pour une ambiance donnée&nbsp;?</p><div class="tech">→ Coordonnées parallèles</div></a>
    <a class="obj" href="#o4"><div class="num">O4</div><h3>Repérer l'atypique</h3>
      <p>Quels morceaux sortent du profil habituel de leur genre&nbsp;?</p><div class="tech">→ Strip plot des écarts</div></a>
  </div>
</section>

<!-- ======================= O1 ======================= -->
<section id="o1">
  <div class="head"><span class="num">O1</span><h2>La carte des genres</h2></div>
  <p class="question">Quel est le profil sonore de chaque genre, et quels genres se ressemblent&nbsp;?</p>
  <div class="tags"><span class="pill tag-tech">ACP + biplot</span><span class="pill">T1 · Comparer les profils</span><span class="pill">T2 · Trouver les genres proches</span></div>
  <div class="grid3">
    <div class="box"><h4>Ce qui est calculé</h4>Profil moyen de chaque genre sur 7 caractéristiques, standardisé, puis projeté en 2D par une
      <b>analyse en composantes principales</b> ({{VAR_EXP}} de l'information conservée).</div>
    <div class="box"><h4>Comment lire</h4><ul>
      <li><b>Proches = similaires.</b></li>
      <li>Les <b>flèches</b> indiquent vers où une caractéristique augmente.</li>
      <li>Couleur = <b>popularité médiane</b>.</li></ul></div>
    <div class="box key"><h4>À retenir</h4>« house » est relié à ses voisins : <b>{{VOISINS}}</b>…
      Les genres les plus singuliers sont <b>{{SINGULIERS}}</b>, très éloignés du reste du catalogue.</div>
  </div>
  <div class="chart">{{FIG1}}<div class="hint">Survolez un point pour afficher le nom du genre · molette pour zoomer · double-clic pour réinitialiser</div></div>
</section>

<!-- ======================= O2 ======================= -->
<section id="o2">
  <div class="head"><span class="num">O2</span><h2>La popularité dépend du genre</h2></div>
  <p class="question">Quelles caractéristiques audio sont associées aux morceaux les plus écoutés&nbsp;?</p>
  <div class="tags"><span class="pill tag-tech">Heatmap de corrélations</span><span class="pill">T4 · Corréler caractéristiques et popularité</span><span class="pill">T3 · Popularité par genre</span></div>
  <div class="grid3">
    <div class="box"><h4>Ce qui est calculé</h4>Pour chaque genre, la <b>corrélation de Spearman</b> (ρ) entre la popularité et chaque caractéristique.
      Morceaux à popularité 0 exclus. Les 30 genres aux liens les plus marqués sont affichés et ordonnés par similarité.</div>
    <div class="box"><h4>Comment lire</h4><ul>
      <li><b style="color:#ff8a57">Orange</b> : plus la caractéristique est élevée, plus le morceau est populaire.</li>
      <li><b style="color:#6aa8f0">Bleu</b> : l'inverse.</li>
      <li>Gris : pas de lien.</li></ul></div>
    <div class="box key"><h4>À retenir</h4>Sur tout le catalogue, les liens sont faibles : |ρ| ≤ <b>{{RHO_GLOBAL}}</b>.
      Mais <b>à l'intérieur des genres</b>, {{N_FORT}} liens dépassent |ρ| = 0,3 (jusqu'à ρ = <b>{{RHO_PAIRE}}</b> pour
      <b>{{PAIRE_F}}</b> en <b>{{PAIRE_G}}</b>). Il n'y a pas de recette universelle : elle dépend du genre.</div>
  </div>
  <div class="chart">{{FIG2}}<div class="hint">Survolez une case pour lire la valeur exacte de ρ</div></div>
</section>

<!-- ======================= O3 ======================= -->
<section id="o3">
  <div class="head"><span class="num">O3</span><h2>Construire une playlist</h2></div>
  <p class="question">Comment trouver rapidement des morceaux répondant à une ambiance donnée&nbsp;?</p>
  <div class="tags"><span class="pill tag-tech">Coordonnées parallèles + brushing</span><span class="pill">T6 · Filtrer par plages</span><span class="pill">T7 · Consulter le détail</span></div>
  <div class="grid3">
    <div class="box"><h4>Ce qui est calculé</h4>Chaque <b>ligne = un morceau</b> (échantillon de 8&nbsp;000 morceaux distincts) traversant 7 axes.
      Les filtres sélectionnés sur les axes se combinent.</div>
    <div class="box"><h4>Comment lire</h4><ul>
      <li><b>Glissez verticalement sur un axe</b> pour filtrer.</li>
      <li>Cliquez sur une zone filtrée pour la retirer.</li>
      <li>Couleur = popularité (vert clair = très populaire).</li></ul></div>
    <div class="box key"><h4>À retenir</h4>Préréglage <b>« soirée dansante »</b> : 120–130 BPM, énergie &gt; 0,8, danceability &gt; 0,7.
      Sur tout le catalogue, <b>{{N_SEL}} morceaux</b> correspondent. Le plus populaire : <b>{{BEST}}</b> ({{BEST_POP}}/100).</div>
  </div>
  <div class="chart">{{FIG3}}</div>
</section>

<!-- ======================= O4 ======================= -->
<section id="o4">
  <div class="head"><span class="num">O4</span><h2>Les morceaux hors norme</h2></div>
  <p class="question">Quels morceaux sortent du profil habituel de leur genre&nbsp;?</p>
  <div class="tags"><span class="pill tag-tech">Strip plot</span><span class="pill">T8 · Localiser les valeurs aberrantes</span></div>
  <div class="grid3">
    <div class="box"><h4>Ce qui est calculé</h4>Pour chaque morceau, la <b>distance au profil moyen de son genre</b> : écart en z-scores sur les 7 caractéristiques,
      combinés (distance euclidienne). Les 12 genres les plus hétérogènes sont affichés.</div>
    <div class="box"><h4>Comment lire</h4><ul>
      <li><b>Un point = un morceau.</b> La masse en bas = la norme du genre.</li>
      <li>Au-dessus de la ligne pointillée (seuil {{SEUIL}}) : le 1&nbsp;% le plus atypique.</li>
      <li><b>Point blanc</b> = le plus atypique du genre.</li></ul></div>
    <div class="box key"><h4>À retenir</h4><b>{{N_ATYP}} morceaux</b> dépassent le seuil. Record : <b>{{ATYP}}</b> ({{ATYP_G}}),
      dont l'<b>{{CAUSE}}</b> est à <b>+{{Z_CAUSE}} écarts-types</b> de la moyenne du genre.</div>
  </div>
  <div class="chart">{{FIG4}}<div class="hint">Survolez un point pour voir le titre et l'artiste</div></div>
</section>

<footer>Données : <i>Spotify Tracks Dataset</i> (Kaggle, maharshipandya) · {{N_UNIQUES}} morceaux distincts après nettoyage · Projet UE Visualisation des données · DI FEDE Santiago, SELMANE Ilyane, TOUPIN Nicolas</footer>
<div class="keys"><kbd>←</kbd> <kbd>→</kbd> naviguer</div>

<script>
  // Navigation au clavier + onglet actif
  const ids = ["intro", "o1", "o2", "o3", "o4"];
  const links = [...document.querySelectorAll("nav .links a")];
  const courant = () => {
    let i = 0;
    ids.forEach((id, k) => { if (document.getElementById(id).getBoundingClientRect().top < innerHeight * 0.4) i = k; });
    return i;
  };
  const maj = () => links.forEach((a, k) => a.classList.toggle("active", k === courant()));
  addEventListener("scroll", maj, { passive: true }); maj();
  addEventListener("keydown", e => {
    if (e.target.closest && e.target.closest(".js-plotly-plot input")) return;
    const d = { ArrowRight: 1, ArrowDown: 1, PageDown: 1, ArrowLeft: -1, ArrowUp: -1, PageUp: -1 }[e.key];
    if (!d) return;
    e.preventDefault();
    const i = Math.min(ids.length - 1, Math.max(0, courant() + d));
    document.getElementById(ids[i]).scrollIntoView({ behavior: "smooth" });
  });
</script>
</body>
</html>
"""

html = TEMPLATE
for k, v in VALS.items():
    html = html.replace("{{" + k + "}}", v)
assert html.count("{{") == 1, "placeholder non remplacé"
html = html.replace("{{PLOTLYJS}}", get_plotlyjs())  # en dernier : plotly.js embarqué (hors ligne)
(ICI / "presentation_spotify.html").write_text(html, encoding="utf-8")

print("presentation_spotify.html générée")
for k, v in VALS.items():
    if not k.startswith("FIG"):
        print(f"  {k}: {v}")
