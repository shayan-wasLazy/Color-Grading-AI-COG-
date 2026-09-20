import ast
import random
from pathlib import Path

import pandas as pd


# ============================================================
# CONFIG
# ============================================================

RANDOM_SEED = 42

TARGET_HOURS = 18
TARGET_SECONDS = TARGET_HOURS * 60 * 60

MIN_DURATION = 30
MAX_DURATION = 180

# Maximum number of clips from one movie
MAX_CLIPS_PER_MOVIE = 3

# Genres that are particularly useful for cinematic color grading
TARGET_GENRES = {
    "Drama",
    "Thriller",
    "Sci-Fi",
    "Romance",
    "Crime",
    "Mystery",
    "Fantasy",
    "Adventure",
    "Action",
    "War",
    "Horror",
}

# Words commonly associated with clips we don't want
EXCLUDE_TERMS = [
    "trailer",
    "teaser",
    "behind the scenes",
    "behind-the-scenes",
    "interview",
    "making of",
    "featurette",
    "deleted scene",
    "deleted scenes",
    "clip compilation",
    "recap",
    "review",
    "reaction",
    "credits",
]


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
METADATA_DIR = DATA_DIR / "metadata"

OUTPUT_DIR = DATA_DIR / "selected"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("Loading metadata...")

clips = pd.read_csv(METADATA_DIR / "clips.csv")
movie_info = pd.read_csv(METADATA_DIR / "movie_info.csv")
movies = pd.read_csv(METADATA_DIR / "movies.csv")
split = pd.read_csv(METADATA_DIR / "split.csv")
descriptions = pd.read_csv(METADATA_DIR / "descriptions.csv")
durations = pd.read_csv(METADATA_DIR / "durations.csv")


print(f"Clips:         {len(clips):,}")
print(f"Movies:        {len(movies):,}")
print(f"Movie info:    {len(movie_info):,}")
print(f"Descriptions:  {len(descriptions):,}")
print(f"Durations:     {len(durations):,}")


# ============================================================
# MERGE METADATA
# ============================================================

df = clips.copy()

# Add train/test split
df = df.merge(
    split[["imdbid", "split"]],
    on="imdbid",
    how="left"
)

# Add genre/country
df = df.merge(
    movie_info[["imdbid", "genre", "country"]],
    on="imdbid",
    how="left"
)

# Add descriptions
df = df.merge(
    descriptions[["videoid", "description"]],
    on="videoid",
    how="left"
)

# Add duration
df = df.merge(
    durations[["videoid", "duration"]],
    on="videoid",
    how="left"
)


# ============================================================
# CLEAN
# ============================================================

df["duration"] = pd.to_numeric(
    df["duration"],
    errors="coerce"
)

df["description"] = df["description"].fillna("")
df["clip_name"] = df["clip_name"].fillna("")
df["genre"] = df["genre"].fillna("[]")


def parse_genres(value):
    try:
        parsed = ast.literal_eval(value)

        if isinstance(parsed, list):
            return set(parsed)

        return set()

    except Exception:
        return set()


df["genre_set"] = df["genre"].apply(parse_genres)


# ============================================================
# FILTER 1: TRAIN SET ONLY
# ============================================================

df = df[df["split"] == "train"].copy()

print(f"\nAfter train split: {len(df):,}")


# ============================================================
# FILTER 2: DURATION
# ============================================================

df = df[
    (df["duration"] >= MIN_DURATION)
    &
    (df["duration"] <= MAX_DURATION)
].copy()

print(f"After duration filter: {len(df):,}")


# ============================================================
# FILTER 3: GENRE
# ============================================================

df["useful_genre"] = df["genre_set"].apply(
    lambda genres: len(genres & TARGET_GENRES) > 0
)

df = df[df["useful_genre"]].copy()

print(f"After genre filter: {len(df):,}")


# ============================================================
# FILTER 4: REMOVE BAD CLIP TYPES
# ============================================================

def contains_excluded_term(row):

    text = (
        str(row["clip_name"]) + " " +
        str(row["description"])
    ).lower()

    return any(
        term in text
        for term in EXCLUDE_TERMS
    )


df["excluded"] = df.apply(
    contains_excluded_term,
    axis=1
)

df = df[~df["excluded"]].copy()

print(f"After content filter: {len(df):,}")


# ============================================================
# ADD A GENRE LABEL
# ============================================================

def get_primary_genre(genres):

    matches = genres & TARGET_GENRES

    if not matches:
        return "Other"

    # deterministic ordering
    return sorted(matches)[0]


df["primary_genre"] = df["genre_set"].apply(
    get_primary_genre
)


# ============================================================
# RANDOMIZE
# ============================================================

df = df.sample(
    frac=1,
    random_state=RANDOM_SEED
).reset_index(drop=True)


# ============================================================
# SELECT BALANCED CLIPS
# ============================================================

selected = []

movie_clip_count = {}
genre_duration = {}

current_duration = 0


for _, row in df.iterrows():

    imdbid = row["imdbid"]
    genre = row["primary_genre"]
    duration = float(row["duration"])

    # Don't take too many clips from one movie
    if movie_clip_count.get(imdbid, 0) >= MAX_CLIPS_PER_MOVIE:
        continue

    # Avoid letting one genre dominate
    current_genre_duration = genre_duration.get(genre, 0)

    # Rough upper bound per genre
    max_genre_duration = TARGET_SECONDS * 0.20

    if current_genre_duration + duration > max_genre_duration:
        continue

    # Don't exceed target
    if current_duration + duration > TARGET_SECONDS:
        continue

    selected.append(row)

    movie_clip_count[imdbid] = (
        movie_clip_count.get(imdbid, 0) + 1
    )

    genre_duration[genre] = (
        genre_duration.get(genre, 0) + duration
    )

    current_duration += duration

    if current_duration >= TARGET_SECONDS:
        break


selected_df = pd.DataFrame(selected)


# ============================================================
# SORT
# ============================================================

selected_df = selected_df.sort_values(
    ["primary_genre", "title", "clip_idx"]
)


# ============================================================
# SAVE
# ============================================================

output_columns = [
    "videoid",
    "imdbid",
    "title",
    "year",
    "upload_year",
    "clip_idx",
    "clip_tot",
    "clip_name",
    "description",
    "duration",
    "primary_genre",
    "genre",
    "country",
]

selected_df[output_columns].to_csv(
    OUTPUT_DIR / "selected_clips.csv",
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("COG DATASET SELECTION COMPLETE")
print("=" * 60)

print(
    f"Selected clips:  {len(selected_df):,}"
)

print(
    f"Unique movies:   {selected_df['imdbid'].nunique():,}"
)

print(
    f"Total duration:   "
    f"{selected_df['duration'].sum() / 3600:.2f} hours"
)

print(
    f"Average clip:     "
    f"{selected_df['duration'].mean():.1f} seconds"
)

print("\nGenre distribution:")

genre_summary = (
    selected_df
    .groupby("primary_genre")["duration"]
    .agg(["count", "sum"])
    .sort_values("sum", ascending=False)
)

genre_summary["hours"] = genre_summary["sum"] / 3600

print(
    genre_summary[
        ["count", "hours"]
    ].to_string()
)

print(
    "\nSaved to:"
)

print(
    OUTPUT_DIR / "selected_clips.csv"
)