import csv
import subprocess
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REPLACEMENT_CSV = (
    PROJECT_ROOT
    / "data"
    / "selected"
    / "replacement_clips.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "videos_selected"
)

FAILED_FILE = (
    PROJECT_ROOT
    / "data"
    / "selected"
    / "failed_replacements.txt"
)


# ============================================================
# SETTINGS
# ============================================================

MAX_HEIGHT = 720


# ============================================================
# LOAD REPLACEMENTS
# ============================================================

def load_replacements():

    print("Loading replacement clips...")

    with open(
        REPLACEMENT_CSV,
        "r",
        encoding="utf-8"
    ) as f:

        reader = csv.DictReader(f)

        rows = list(reader)

    print(
        f"Replacement clips found: {len(rows)}"
    )

    return rows


# ============================================================
# DOWNLOAD
# ============================================================

def download_video(video_id, upload_year):

    year_dir = (
        OUTPUT_DIR /
        str(upload_year)
    )

    year_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_template = str(
        year_dir /
        f"{video_id}.%(ext)s"
    )

    url = (
        f"https://www.youtube.com/watch?v={video_id}"
    )

    print()
    print("=" * 70)
    print(f"Video ID : {video_id}")
    print(f"Year     : {upload_year}")
    print("=" * 70)

    command = [
        "yt-dlp",

        # ----------------------------------------------------
        # Maximum 720p
        # ----------------------------------------------------

        "-f",
        "bestvideo[height<=720]+bestaudio/"
        "best[height<=720]",

        # ----------------------------------------------------
        # Output
        # ----------------------------------------------------

        "-o",
        output_template,

        # Merge into MKV
        "--merge-output-format",
        "mkv",

        # Don't overwrite existing files
        "--no-overwrites",

        # Resume partial downloads
        "--continue",

        # Never download playlists
        "--no-playlist",

        # ----------------------------------------------------
        # URL
        # ----------------------------------------------------

        url,
    ]

    result = subprocess.run(command)

    return result.returncode == 0


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("COG - REPLACEMENT DOWNLOADER")
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    clips = load_replacements()

    failed = []

    total = len(clips)

    print()
    print(f"Videos to download: {total}")
    print(f"Output directory:   {OUTPUT_DIR}")
    print(f"Maximum resolution: {MAX_HEIGHT}p")
    print()

    for index, row in enumerate(
        clips,
        start=1
    ):

        video_id = row["videoid"].strip()

        upload_year = row["upload_year"].strip()

        print()
        print(
            f"[{index}/{total}]"
        )

        # ----------------------------------------------------
        # Check whether this replacement already exists
        # ----------------------------------------------------

        year_dir = (
            OUTPUT_DIR /
            upload_year
        )

        existing_files = list(
            year_dir.glob(
                f"{video_id}.*"
            )
        )

        if existing_files:

            print(
                f"Already exists: {video_id}"
            )

            continue

        # ----------------------------------------------------
        # Download
        # ----------------------------------------------------

        success = download_video(
            video_id,
            upload_year
        )

        if not success:

            print(
                f"FAILED: {video_id}"
            )

            failed.append(
                video_id
            )

    # ========================================================
    # SAVE FAILURES
    # ========================================================

    if failed:

        with open(
            FAILED_FILE,
            "w",
            encoding="utf-8"
        ) as f:

            for video_id in failed:

                f.write(
                    video_id + "\n"
                )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    print()
    print("=" * 70)
    print("REPLACEMENT DOWNLOAD COMPLETE")
    print("=" * 70)

    print(
        f"Total replacements : {total}"
    )

    print(
        f"Failed             : {len(failed)}"
    )

    print()

    if failed:

        print(
            "Failed IDs saved to:"
        )

        print(
            FAILED_FILE
        )

    else:

        print(
            "All replacement videos downloaded successfully!"
        )


if __name__ == "__main__":
    main()