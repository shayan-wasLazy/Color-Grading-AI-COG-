from pathlib import Path
import cv2
import pandas as pd
from tqdm import tqdm

ROOT = Path("F:/Cog")

VIDEO_DIR = ROOT / "data" / "videos_selected"
SELECTED_CSV = ROOT / "data" / "selected" / "selected_clips.csv"

OUTPUT = ROOT / "data" / "selected" / "video_validation.csv"


def validate_video(path):

    cap = cv2.VideoCapture(str(path))

    if not cap.isOpened():
        return {
            "valid": False,
            "fps": 0,
            "frames": 0,
            "duration": 0,
            "width": 0,
            "height": 0
        }

    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps and fps > 0:
        duration = frames / fps
    else:
        duration = 0

    cap.release()

    valid = (
        frames > 0 and
        width > 0 and
        height > 0 and
        fps > 0
    )

    return {
        "valid": valid,
        "fps": fps,
        "frames": frames,
        "duration": duration,
        "width": width,
        "height": height
    }


def main():

    rows = []

    videos = list(VIDEO_DIR.rglob("*.mkv"))

    print(f"Found {len(videos)} videos")

    for video in tqdm(videos):

        info = validate_video(video)

        rows.append({
            "video_path": str(video),
            "videoid": video.stem,
            **info
        })

    df = pd.DataFrame(rows)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(OUTPUT, index=False)

    print("\nValidation complete")
    print(f"Valid: {df['valid'].sum()}")
    print(f"Invalid: {(~df['valid']).sum()}")
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()