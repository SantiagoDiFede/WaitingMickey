import pandas as pd
import kagglehub
from d3blocks import D3Blocks

# --- Load ---
path = kagglehub.dataset_download("maharshipandya/-spotify-tracks-dataset")
df = pd.read_csv(f"{path}/dataset.csv", index_col=0)
df["explicit"] = df["explicit"].astype(int)

numeric_cols = [
    "popularity", "duration_ms", "explicit", "danceability", "energy", "key",
    "loudness", "mode", "speechiness", "acousticness", "instrumentalness",
    "liveness", "valence", "tempo", "time_signature",
]

d3 = D3Blocks()

# --- 1. Heatmap: correlation between all numeric columns ---
corr = df[numeric_cols].corr().round(2)
corr_long = (
    corr.reset_index()
        .melt(id_vars="index", var_name="target", value_name="weight")
        .rename(columns={"index": "source"})
)
d3.heatmap(corr_long, title="Audio feature correlations",
           filepath="1_heatmap.html")

# --- 2. Scatter: every text column goes into the tooltip ---
sample = df.sample(5000, random_state=42).reset_index(drop=True)
tooltip = (
    sample["track_name"] + " | " + sample["artists"] + " | " +
    sample["album_name"] + " | " + sample["track_genre"] +
    " | id: " + sample["track_id"]
).values

d3.scatter(
    sample["energy"].values,
    sample["valence"].values,
    size=(sample["popularity"] / 10 + 1).values,
    color=sample["track_genre"].values,
    tooltip=tooltip,
    title="Energy (x) vs Valence (y), size = popularity, color = genre",
    filepath="2_scatter.html",
)

# --- 3. Violin plots: each numeric feature across the top genres ---
top_genres = df["track_genre"].value_counts().head(8).index
sub = df[df["track_genre"].isin(top_genres)]
for feature in ["danceability", "energy", "valence", "tempo", "loudness",
                "acousticness", "speechiness", "instrumentalness",
                "liveness", "popularity", "duration_ms"]:
    d3.violin(x=sub["track_genre"].values, y=sub[feature].values,
              title=f"{feature} by genre",
              filepath=f"3_violin_{feature}.html", showfig=False)

# --- 4. Sankey: genre -> explicit -> mode -> key (categorical columns) ---
flows = []
for a, b in [("track_genre", "explicit"), ("explicit", "mode"), ("mode", "key")]:
    g = (sub.assign(**{a: sub[a].astype(str) + f" ({a})",
                       b: sub[b].astype(str) + f" ({b})"})
            .groupby([a, b]).size().reset_index(name="weight"))
    g.columns = ["source", "target", "weight"]
    flows.append(g)
d3.sankey(pd.concat(flows), title="Genre → explicit → mode → key",
          filepath="4_sankey.html")

# --- 5. Chord: top artists vs top genres ---
top_artists = df["artists"].value_counts().head(15).index
ch = (df[df["artists"].isin(top_artists) & df["track_genre"].isin(top_genres)]
        .groupby(["artists", "track_genre"]).size().reset_index())
ch.columns = ["source", "target", "weight"]
d3.chord(ch, title="Top artists vs top genres", filepath="5_chord.html")

# --- 6. Parallel coordinates: all the audio features at once ---
pc = sub[["track_genre", "danceability", "energy", "valence", "acousticness",
          "speechiness", "liveness", "instrumentalness"]].sample(1500, random_state=1)
d3.parallelcoordinates(pc, color="track_genre", filepath="6_parallel.html")