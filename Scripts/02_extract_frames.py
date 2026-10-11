from pathlib import Path
import cv2
import pandas as pd
from tqdm import tqdm


ROOT = Path("F:/Cog")

VIDEO_DIR = ROOT / "data" / "videos_selected"
FRAME_DIR = ROOT / "data" / "frames"

SELECTED_CSV = ROOT / "data" / "selected" / "selected_clips.csv"
VALIDATION_CSV = ROOT / "data" / "selected" / "video_validation.csv"

INTERVAL_SECONDS = 2
JPEG_QUALITY = 95


def extract_video_frames(video_path, output_dir, video_id):

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"Could not open: {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        cap.release()
        return []

    duration = total_frames / fps

    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = []

    frame_number = 0
    next_sample_time = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        timestamp = frame_number / fps

        if timestamp >= next_sample_time:

            filename = f"{video_id}_{timestamp:010.2f}.jpg"

            output_path = output_dir / filename

            cv2.imwrite(
                str(output_path),
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
            )

            metadata.append({
                "videoid": video_id,
                "frame_path": str(output_path),
                "timestamp": timestamp,
                "frame_number": frame_number,
                "width": frame.shape[1],
                "height": frame.shape[0]
            })

            next_sample_time += INTERVAL_SECONDS

        frame_number += 1

    cap.release()

    return metadata


def main():

    FRAME_DIR.mkdir(parents=True, exist_ok=True)

    videos = list(VIDEO_DIR.rglob("*.mkv"))

    print(f"Videos: {len(videos)}")

    all_metadata = []

    for video_path in tqdm(videos):

        video_id = video_path.stem

        output_dir = FRAME_DIR / video_id

        # Skip if already extracted
        existing = list(output_dir.glob("*.jpg"))

        if existing:
            print(f"Skipping {video_id} - already extracted")

            continue

        metadata = extract_video_frames(
            video_path,
            output_dir,
            video_id
        )

        all_metadata.extend(metadata)

    if all_metadata:

        df = pd.DataFrame(all_metadata)

        output_csv = FRAME_DIR / "frames.csv"

        df.to_csv(output_csv, index=False)

        print("\nExtraction complete")
        print(f"Frames: {len(df):,}")
        print(f"Videos: {df['videoid'].nunique()}")
        print(f"Saved: {output_csv}")


if __name__ == "__main__":
    main()