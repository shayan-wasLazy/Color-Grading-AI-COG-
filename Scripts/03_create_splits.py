from pathlib import Path
import pandas as pd
import numpy as np


ROOT = Path("F:/Cog")

FRAME_CSV = ROOT / "data" / "frames" / "frames_clean.csv"
SPLIT_DIR = ROOT / "data" / "splits"

SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def main():

    df = pd.read_csv(FRAME_CSV)

    print(f"Total frames: {len(df):,}")
    print(f"Total videos: {df['videoid'].nunique()}")

    # One video can contain many frames.
    # We split VIDEOS, not frames.

    videos = df["videoid"].unique()

    rng = np.random.default_rng(SEED)

    rng.shuffle(videos)

    n_videos = len(videos)

    n_train = int(n_videos * TRAIN_RATIO)
    n_val = int(n_videos * VAL_RATIO)

    train_videos = videos[:n_train]

    val_videos = videos[
        n_train:n_train + n_val
    ]

    test_videos = videos[
        n_train + n_val:
    ]

    train_df = df[
        df["videoid"].isin(train_videos)
    ].copy()

    val_df = df[
        df["videoid"].isin(val_videos)
    ].copy()

    test_df = df[
        df["videoid"].isin(test_videos)
    ].copy()

    SPLIT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    train_df.to_csv(
        SPLIT_DIR / "train.csv",
        index=False
    )

    val_df.to_csv(
        SPLIT_DIR / "val.csv",
        index=False
    )

    test_df.to_csv(
        SPLIT_DIR / "test.csv",
        index=False
    )

    print("\n============================")
    print("COG DATASET SPLIT")
    print("============================")

    print(
        f"Train videos : {len(train_videos)}"
    )

    print(
        f"Val videos   : {len(val_videos)}"
    )

    print(
        f"Test videos  : {len(test_videos)}"
    )

    print("\nFrames")

    print(
        f"Train : {len(train_df):,}"
    )

    print(
        f"Val   : {len(val_df):,}"
    )

    print(
        f"Test  : {len(test_df):,}"
    )

    print(
        f"\nSaved to {SPLIT_DIR}"
    )


if __name__ == "__main__":
    main()