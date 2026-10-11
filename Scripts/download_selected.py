import os
import csv
import subprocess
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

SELECTED_CSV = ROOT / "data" / "selected" / "selected_clips.csv"
OUTPUT_DIR = ROOT / "data" / "videos_selected"

FAILED_FILE = ROOT / "data" / "selected" / "failed_downloads.txt"


# ============================================================
# SETTINGS
# ============================================================

# Maximum video resolution
MAX_HEIGHT = 720

# Stop after approximately this much data
TARGET_GB = 30

# Set to False if you want to download everything selected
USE_SIZE_LIMIT = False


# ============================================================
# HELPERS
# ============================================================

def get_directory_size_gb(path):
    """Return directory size in GB."""

    total = 0

    if not path.exists():
        return 0

    for file in path.rglob("*"):
        if file.is_file():
            try:
                total += file.stat().st_size
            except OSError:
                pass

    return total / (1024 ** 3)


def load_selected_clips():
    """Load selected clips from CSV."""

    print("Loading selected clips...")

    with open(SELECTED_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        rows = list(reader)

    print(f"Selected clips found: {len(rows)}")

    return rows


def find_video_id(row):
    """
    Extract the video ID.

    Expected column:
        videoid
    """

    video_id = row.get("videoid")

    if not video_id:
        raise ValueError(
            f"Could not find 'videoid' column/value in row:\n{row}"
        )

    return video_id.strip()


def get_upload_year(row):
    """Get upload year for folder organization."""

    year = row.get("upload_year")

    if year:
        try:
            return str(int(float(year)))
        except ValueError:
            pass

    return "unknown"


# ============================================================
# DOWNLOAD
# ============================================================

def download_video(video_id, upload_year):

    year_dir = OUTPUT_DIR / upload_year
    year_dir.mkdir(parents=True, exist_ok=True)

    output_template = str(
        year_dir / f"{video_id}.%(ext)s"
    )

    url = f"https://www.youtube.com/watch?v={video_id}"

    print()
    print("=" * 70)
    print(f"Video ID : {video_id}")
    print(f"Year     : {upload_year}")
    print("=" * 70)

    command = [
        "yt-dlp",

        # ----------------------------------------------------
        # Video quality
        # ----------------------------------------------------

        "-f",
        "bestvideo[height<=720]+bestaudio/"
        "best[height<=720]",

        # ----------------------------------------------------
        # Output
        # ----------------------------------------------------

        "-o",
        output_template,

        # Merge into MKV like the original dataset
        "--merge-output-format",
        "mkv",

        # Don't download if file already exists
        "--no-overwrites",

        # Continue partial downloads
        "--continue",

        # Don't download thumbnails/subtitles/etc.
        "--no-playlist",

        # ----------------------------------------------------
        # YouTube URL
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
    print("COG - SELECTED CONDENSED MOVIES DOWNLOADER")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    clips = load_selected_clips()

    failed = []

    total = len(clips)

    print()
    print(f"Videos to download: {total}")
    print(f"Output directory:   {OUTPUT_DIR}")
    print(f"Maximum resolution:  {MAX_HEIGHT}p")
    print()

    for index, row in enumerate(clips, start=1):

        video_id = find_video_id(row)
        upload_year = get_upload_year(row)

        # ----------------------------------------------------
        # Size check
        # ----------------------------------------------------

        current_size = get_directory_size_gb(OUTPUT_DIR)

        if USE_SIZE_LIMIT and current_size >= TARGET_GB:

            print()
            print("=" * 70)
            print("TARGET SIZE REACHED")
            print("=" * 70)
            print(f"Current size: {current_size:.2f} GB")
            print(f"Target size : {TARGET_GB:.2f} GB")
            break

        print()
        print(f"[{index}/{total}]")
        print(f"Current dataset size: {current_size:.2f} GB")

        # ----------------------------------------------------
        # Check if already downloaded
        # ----------------------------------------------------

        existing_files = list(
            (OUTPUT_DIR / upload_year).glob(f"{video_id}.*")
        )

        if existing_files:

            print(f"Already exists: {video_id}")
            continue

        # ----------------------------------------------------
        # Download
        # ----------------------------------------------------

        success = download_video(
            video_id,
            upload_year
        )

        if not success:

            print(f"FAILED: {video_id}")

            failed.append(video_id)

    # ========================================================
    # SAVE FAILED DOWNLOADS
    # ========================================================

    if failed:

        with open(
            FAILED_FILE,
            "w",
            encoding="utf-8"
        ) as f:

            for video_id in failed:
                f.write(video_id + "\n")

    # ========================================================
    # FINAL REPORT
    # ========================================================

    final_size = get_directory_size_gb(OUTPUT_DIR)

    print()
    print("=" * 70)
    print("DOWNLOAD COMPLETE")
    print("=" * 70)

    print(f"Dataset size : {final_size:.2f} GB")
    print(f"Failed       : {len(failed)}")

    if failed:
        print()
        print(f"Failed IDs saved to:")
        print(FAILED_FILE)


if __name__ == "__main__":
    main()