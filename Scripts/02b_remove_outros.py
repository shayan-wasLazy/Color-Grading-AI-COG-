from pathlib import Path
import pandas as pd


ROOT = Path("F:/Cog")

FRAME_DIR = ROOT / "data" / "frames"
FRAME_CSV = FRAME_DIR / "frames.csv"

FRAMES_TO_REMOVE = 15

OUTPUT_CSV = FRAME_DIR / "frames_clean.csv"


def main():

    df = pd.read_csv(FRAME_CSV)

    print(f"Original frames: {len(df):,}")
    print(f"Videos: {df['videoid'].nunique()}")

    cleaned_groups = []

    removed_total = 0

    for video_id, group in df.groupby("videoid", sort=False):

        # Sort chronologically
        group = group.sort_values(
            "timestamp"
        ).reset_index(drop=True)

        original_count = len(group)

        if original_count <= FRAMES_TO_REMOVE:
            print(
                f"WARNING: {video_id} has only "
                f"{original_count} frames"
            )

            cleaned = group.iloc[0:0]

        else:
            cleaned = group.iloc[
                :-FRAMES_TO_REMOVE
            ]

        removed = original_count - len(cleaned)

        removed_total += removed

        cleaned_groups.append(cleaned)

    clean_df = pd.concat(
        cleaned_groups,
        ignore_index=True
    )

    clean_df.to_csv(
        OUTPUT_CSV,
        index=False
    )

    print("\n-----------------------------")
    print("Outro removal complete")
    print("-----------------------------")

    print(
        f"Original frames : {len(df):,}"
    )

    print(
        f"Removed frames  : {removed_total:,}"
    )

    print(
        f"Clean frames    : {len(clean_df):,}"
    )

    print(
        f"Videos          : "
        f"{clean_df['videoid'].nunique()}"
    )

    print(
        f"Saved           : {OUTPUT_CSV}"
    )


if __name__ == "__main__":
    main()