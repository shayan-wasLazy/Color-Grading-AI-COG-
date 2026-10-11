import ast
from pathlib import Path

import pandas as pd


# ============================================================
# CONFIG
# ============================================================

RANDOM_SEED = 42

TARGET_TOTAL_HOURS = 18
TARGET_TOTAL_SECONDS = TARGET_TOTAL_HOURS * 60 * 60

MIN_DURATION = 30
MAX_DURATION = 180

MAX_CLIPS_PER_MOVIE = 3

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

DATA_ROOT = PROJECT_ROOT / "data"

CM_DIR = PROJECT_ROOT / "CondensedMovies"
METADATA_DIR = CM_DIR / "data" / "metadata"

SELECTED_DIR = DATA_ROOT / "selected"

ORIGINAL_SELECTION = SELECTED_DIR / "selected_clips.csv"
FAILED_FILE = SELECTED_DIR / "failed_downloads.txt"

OUTPUT_FILE = SELECTED_DIR / "replacement_clips.csv"


# ============================================================
# LOAD ORIGINAL + FAILED IDS
# ============================================================

print("Loading existing dataset information...")

original = pd.read_csv(ORIGINAL_SELECTION)

failed_ids = set()

if FAILED_FILE.exists():
    with open(FAILED_FILE, "r", encoding="utf-8") as f:
        failed_ids = {
            line.strip()
            for line in f
            if line.strip()
        }

print(f"Original selected clips : {len(original):,}")
print(f"Failed video IDs        : {len(failed_ids):,}")


# ============================================================
# CALCULATE CURRENT USABLE DATASET
# ============================================================

successful = original[
    ~original["videoid"].isin(failed_ids)
].copy()

successful_duration = successful["duration"].sum()

remaining_seconds = (
    TARGET_TOTAL_SECONDS -
    successful_duration
)

print()
print("=" * 60)
print("CURRENT COG DATASET")
print("=" * 60)

print(
    f"Successful clips : {len(successful):,}"
)

print(
    f"Current duration : "
    f"{successful_duration / 3600:.3f} hours"
)

print(
    f"Replacement needed : "
    f"{remaining_seconds / 3600:.3f} hours"
)


if remaining_seconds <= 0:
    print("\nDataset already reached target.")
    exit()


# ============================================================
# LOAD METADATA
# ============================================================

print("\nLoading metadata...")

clips = pd.read_csv(
    METADATA_DIR / "clips.csv"
)

movie_info = pd.read_csv(
    METADATA_DIR / "movie_info.csv"
)

split = pd.read_csv(
    METADATA_DIR / "split.csv"
)

descriptions = pd.read_csv(
    METADATA_DIR / "descriptions.csv"
)

durations = pd.read_csv(
    METADATA_DIR / "durations.csv"
)


# ============================================================
# MERGE
# ============================================================

df = clips.copy()

df = df.merge(
    split[["imdbid", "split"]],
    on="imdbid",
    how="left"
)

df = df.merge(
    movie_info[["imdbid", "genre", "country"]],
    on="imdbid",
    how="left"
)

df = df.merge(
    descriptions[["videoid", "description"]],
    on="videoid",
    how="left"
)

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
# EXACT SAME FILTERS AS ORIGINAL SCRIPT
# ============================================================

# Train only
df = df[
    df["split"] == "train"
].copy()

print(
    f"After train split: "
    f"{len(df):,}"
)


# Duration
df = df[
    (df["duration"] >= MIN_DURATION)
    &
    (df["duration"] <= MAX_DURATION)
].copy()

print(
    f"After duration filter: "
    f"{len(df):,}"
)


# Genre
df["useful_genre"] = df["genre_set"].apply(
    lambda genres:
        len(genres & TARGET_GENRES) > 0
)

df = df[
    df["useful_genre"]
].copy()

print(
    f"After genre filter: "
    f"{len(df):,}"
)


# Content filtering
def contains_excluded_term(row):

    text = (
        str(row["clip_name"]) +
        " " +
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

df = df[
    ~df["excluded"]
].copy()

print(
    f"After content filter: "
    f"{len(df):,}"
)


# ============================================================
# REMOVE EVERYTHING WE ALREADY KNOW ABOUT
# ============================================================

# Original 517 clips
original_ids = set(
    original["videoid"]
)

# Failed clips
# They are removed because we already know
# those YouTube IDs are unavailable/private.
exclude_ids = original_ids | failed_ids

df = df[
    ~df["videoid"].isin(exclude_ids)
].copy()

print()
print(
    f"Remaining replacement candidates: "
    f"{len(df):,}"
)


# ============================================================
# PRIMARY GENRE
# ============================================================

def get_primary_genre(genres):

    matches = genres & TARGET_GENRES

    if not matches:
        return "Other"

    return sorted(matches)[0]


df["primary_genre"] = df[
    "genre_set"
].apply(
    get_primary_genre
)


# ============================================================
# CALCULATE ORIGINAL GENRE DISTRIBUTION
# ============================================================

original_genre_duration = (
    original
    .groupby("primary_genre")["duration"]
    .sum()
)

print()
print("Original genre distribution:")

for genre, seconds in (
    original_genre_duration
    .sort_values(ascending=False)
    .items()
):

    print(
        f"{genre:12s} "
        f"{seconds / 3600:.2f} h"
    )


# ============================================================
# RANDOMIZE CANDIDATES
# ============================================================

df = df.sample(
    frac=1,
    random_state=RANDOM_SEED
).reset_index(drop=True)


# ============================================================
# REPLACEMENT SELECTION
# ============================================================

selected = []

movie_clip_count = {}

replacement_genre_duration = {}

current_duration = 0


# We want the final dataset to roughly preserve
# the genre proportions of the original 18-hour design.

original_total = original["duration"].sum()

genre_targets = {}

for genre, seconds in original_genre_duration.items():

    proportion = seconds / original_total

    genre_targets[genre] = (
        proportion *
        TARGET_TOTAL_SECONDS
    )


# ============================================================
# PASS 1
# Prioritize genres whose final target has the
# largest remaining deficit.
# ============================================================

remaining = df.copy()


while current_duration < remaining_seconds:

    best_candidate = None
    best_score = None

    for idx, row in remaining.iterrows():

        video_id = row["videoid"]
        imdbid = row["imdbid"]
        genre = row["primary_genre"]

        duration = float(row["duration"])

        # Maximum 3 clips/movie
        if (
            movie_clip_count.get(imdbid, 0)
            >= MAX_CLIPS_PER_MOVIE
        ):
            continue

        # Don't exceed required replacement duration
        if (
            current_duration + duration
            > remaining_seconds
        ):
            continue

        target = genre_targets.get(
            genre,
            0
        )

        current_genre = (
            original_genre_duration.get(
                genre,
                0
            )
            +
            replacement_genre_duration.get(
                genre,
                0
            )
        )

        deficit = target - current_genre

        # Favor genres currently below
        # their intended final representation.
        score = deficit

        if best_score is None or score > best_score:

            best_score = score
            best_candidate = (
                idx,
                row
            )

    if best_candidate is None:
        break

    idx, row = best_candidate

    selected.append(row)

    imdbid = row["imdbid"]
    genre = row["primary_genre"]
    duration = float(row["duration"])

    movie_clip_count[imdbid] = (
        movie_clip_count.get(imdbid, 0)
        + 1
    )

    replacement_genre_duration[genre] = (
        replacement_genre_duration.get(
            genre,
            0
        )
        +
        duration
    )

    current_duration += duration

    remaining = remaining.drop(
        index=idx
    )


# ============================================================
# BUILD DATAFRAME
# ============================================================

replacement_df = pd.DataFrame(
    selected
)


# ============================================================
# SORT
# ============================================================

replacement_df = replacement_df.sort_values(
    [
        "primary_genre",
        "title",
        "clip_idx"
    ]
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

replacement_df[
    output_columns
].to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 60)
print("COG REPLACEMENT SELECTION COMPLETE")
print("=" * 60)

print(
    f"Replacement clips : "
    f"{len(replacement_df):,}"
)

print(
    f"Replacement hours : "
    f"{replacement_df['duration'].sum() / 3600:.3f}"
)

print(
    f"Final estimated hours : "
    f"{(successful_duration + replacement_df['duration'].sum()) / 3600:.3f}"
)

print(
    f"Unique replacement movies : "
    f"{replacement_df['imdbid'].nunique():,}"
)

print("\nReplacement genre distribution:")

genre_summary = (
    replacement_df
    .groupby("primary_genre")["duration"]
    .agg(["count", "sum"])
    .sort_values(
        "sum",
        ascending=False
    )
)

genre_summary["hours"] = (
    genre_summary["sum"] / 3600
)

print(
    genre_summary[
        ["count", "hours"]
    ].to_string()
)

print()
print("Saved to:")
print(OUTPUT_FILE)