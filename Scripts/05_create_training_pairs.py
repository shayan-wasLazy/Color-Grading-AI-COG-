from pathlib import Path
import random
import sys
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


# ============================================================
# IMPORT LUT UTILITIES
# ============================================================

sys.path.append(
    str(Path(__file__).parent)
)

from lut_utils import apply_lut


# ============================================================
# PATHS
# ============================================================

ROOT = Path("F:/Cog")

SPLIT_DIR = (
    ROOT
    / "data"
    / "splits"
)

LUT_DIR = (
    ROOT
    / "data"
    / "luts"
)

PAIR_DIR = (
    ROOT
    / "data"
    / "pairs"
)

PROCESSED_DIR = (
    ROOT
    / "data"
    / "processed"
)


# ============================================================
# SETTINGS
# ============================================================

SEED = 42

# Number of CPU workers used for image/LUT processing.
#
# If your CPU becomes very heavily loaded or your PC becomes
# unresponsive, reduce this to 4.
#
NUM_WORKERS = 8

JPEG_QUALITY = 95

TRAIN_SAMPLES = 10_000
VAL_SAMPLES = 1_500
TEST_SAMPLES = 1_500


random.seed(SEED)
np.random.seed(SEED)


# ============================================================
# LOAD LUTS
# ============================================================

def load_luts():

    paths = sorted(
        LUT_DIR.glob("*.npy")
    )

    if not paths:

        raise RuntimeError(
            f"No LUT files found in {LUT_DIR}"
        )

    luts = []

    for path in paths:

        lut = np.load(path)

        luts.append(
            (
                path.name,
                lut
            )
        )

    return luts


# ============================================================
# LOAD IMAGE
# ============================================================

def load_image(path):

    image = cv2.imread(
        str(path)
    )

    if image is None:
        return None

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    return image


# ============================================================
# SAVE IMAGE
# ============================================================

def save_image(
    image,
    path
):

    image_bgr = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2BGR
    )

    cv2.imwrite(
        str(path),
        image_bgr,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            JPEG_QUALITY
        ]
    )


# ============================================================
# CREATE ONE STYLED REFERENCE
# ============================================================

def create_styled_reference(job):

    (
        reference_path,
        output_path,
        lut
    ) = job

    # --------------------------------------------------------
    # Don't regenerate existing file
    # --------------------------------------------------------

    if output_path.exists():
        return True


    # --------------------------------------------------------
    # Load reference
    # --------------------------------------------------------

    reference = load_image(
        reference_path
    )

    if reference is None:
        return False


    # --------------------------------------------------------
    # Apply LUT
    # --------------------------------------------------------

    styled_reference = apply_lut(
        reference,
        lut
    )


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_image(
        styled_reference,
        output_path
    )

    return True


# ============================================================
# CREATE ONE TARGET
# ============================================================

def create_target(job):

    (
        source_path,
        target_path,
        lut
    ) = job

    # --------------------------------------------------------
    # Don't regenerate existing target
    # --------------------------------------------------------

    if target_path.exists():
        return True


    # --------------------------------------------------------
    # Load source
    # --------------------------------------------------------

    source = load_image(
        source_path
    )

    if source is None:
        return False


    # --------------------------------------------------------
    # Apply LUT
    # --------------------------------------------------------

    target = apply_lut(
        source,
        lut
    )


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_image(
        target,
        target_path
    )

    return True


# ============================================================
# CHOOSE REFERENCE FRAME FOR EACH VIDEO
# ============================================================

def choose_reference_frames(df):

    reference_frames = {}


    for video_id, group in df.groupby(
        "videoid"
    ):

        # Fixed random_state means that the same video gets
        # the same reference frame whenever this script runs.

        row = group.sample(
            n=1,
            random_state=SEED
        ).iloc[0]


        reference_frames[video_id] = {
            "videoid":
                video_id,

            "frame_path":
                row["frame_path"]
        }


    return reference_frames


# ============================================================
# CREATE REFERENCE BANK
# ============================================================

def create_reference_bank(
    df,
    luts,
    split_name
):

    print(
        f"\nCreating reference bank "
        f"for {split_name}..."
    )


    # --------------------------------------------------------
    # Choose exactly ONE reference frame per video
    # --------------------------------------------------------

    reference_frames = (
        choose_reference_frames(df)
    )


    print(
        f"Reference videos: "
        f"{len(reference_frames)}"
    )


    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    reference_root = (
        PROCESSED_DIR
        / split_name
        / "references"
    )

    reference_root.mkdir(
        parents=True,
        exist_ok=True
    )


    # --------------------------------------------------------
    # Create processing jobs
    # --------------------------------------------------------

    jobs = []


    for reference_video, info in (
        reference_frames.items()
    ):

        reference_path = Path(
            info["frame_path"]
        )


        for lut_name, lut in luts:

            lut_stem = Path(
                lut_name
            ).stem


            output_dir = (
                reference_root
                / reference_video
                / lut_stem
            )


            output_dir.mkdir(
                parents=True,
                exist_ok=True
            )


            output_path = (
                output_dir
                / (
                    reference_path.stem
                    + ".jpg"
                )
            )


            jobs.append(
                (
                    reference_path,
                    output_path,
                    lut
                )
            )


    # --------------------------------------------------------
    # Remove files that already exist
    # --------------------------------------------------------

    jobs_to_process = [
        job
        for job in jobs
        if not job[1].exists()
    ]


    print(
        f"Reference images already existing: "
        f"{len(jobs) - len(jobs_to_process):,}"
    )


    print(
        f"Reference images to generate: "
        f"{len(jobs_to_process):,}"
    )


    # --------------------------------------------------------
    # Parallel reference processing
    # --------------------------------------------------------

    if jobs_to_process:

        with ThreadPoolExecutor(
            max_workers=NUM_WORKERS
        ) as executor:

            results = executor.map(
                create_styled_reference,
                jobs_to_process
            )


            successful = 0

            for result in tqdm(
                results,
                total=len(jobs_to_process),
                desc=f"References {split_name}"
            ):

                if result:
                    successful += 1


        print(
            f"Created references: "
            f"{successful:,}"
        )

    else:

        print(
            "Reference bank already complete."
        )


    return reference_frames


# ============================================================
# CREATE TARGETS
# ============================================================

def create_targets(
    df,
    luts,
    split_name,
    max_samples
):

    print(
        f"\nCreating targets "
        f"for {split_name}..."
    )


    # --------------------------------------------------------
    # Select source frames
    # --------------------------------------------------------

    if max_samples is not None:

        df = df.sample(
            n=min(
                max_samples,
                len(df)
            ),
            random_state=SEED
        )

        df = df.reset_index(
            drop=True
        )


    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    target_root = (
        PROCESSED_DIR
        / split_name
        / "targets"
    )

    target_root.mkdir(
        parents=True,
        exist_ok=True
    )


    # --------------------------------------------------------
    # We first choose which LUT each sample receives.
    #
    # This is done here so the exact same information can
    # later be written into the pair CSV.
    # --------------------------------------------------------

    selected_luts = []


    for _, row in df.iterrows():

        lut_name, lut = random.choice(
            luts
        )

        selected_luts.append(
            (
                lut_name,
                lut
            )
        )


    # --------------------------------------------------------
    # Create target jobs
    # --------------------------------------------------------

    jobs = []


    for index, (_, row) in enumerate(
        df.iterrows()
    ):

        source_path = Path(
            row["frame_path"]
        )


        source_video = row[
            "videoid"
        ]


        lut_name, lut = (
            selected_luts[index]
        )


        lut_stem = Path(
            lut_name
        ).stem


        output_dir = (
            target_root
            / source_video
        )


        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )


        target_path = (
            output_dir
            / (
                source_path.stem
                + "_"
                + lut_stem
                + ".jpg"
            )
        )


        jobs.append(
            (
                source_path,
                target_path,
                lut
            )
        )


    # --------------------------------------------------------
    # Don't regenerate existing targets
    # --------------------------------------------------------

    jobs_to_process = [
        job
        for job in jobs
        if not job[1].exists()
    ]


    print(
        f"Targets already existing: "
        f"{len(jobs) - len(jobs_to_process):,}"
    )


    print(
        f"Targets to generate: "
        f"{len(jobs_to_process):,}"
    )


    # --------------------------------------------------------
    # Parallel target processing
    # --------------------------------------------------------

    if jobs_to_process:

        with ThreadPoolExecutor(
            max_workers=NUM_WORKERS
        ) as executor:

            results = executor.map(
                create_target,
                jobs_to_process
            )


            successful = 0

            for result in tqdm(
                results,
                total=len(jobs_to_process),
                desc=f"Targets {split_name}"
            ):

                if result:
                    successful += 1


        print(
            f"Created targets: "
            f"{successful:,}"
        )

    else:

        print(
            "Targets already complete."
        )


    return df, selected_luts


# ============================================================
# CREATE PAIR CSV
# ============================================================

def create_pair_csv(
    df,
    selected_luts,
    reference_frames,
    luts,
    split_name
):

    print(
        f"\nCreating pair CSV "
        f"for {split_name}..."
    )


    reference_root = (
        PROCESSED_DIR
        / split_name
        / "references"
    )


    target_root = (
        PROCESSED_DIR
        / split_name
        / "targets"
    )


    rows = []


    # --------------------------------------------------------
    # Build lookup for LUT arrays
    # --------------------------------------------------------

    lut_dict = {
        name: lut
        for name, lut in luts
    }


    # --------------------------------------------------------
    # Group source videos
    #
    # Used to select a different reference video.
    # --------------------------------------------------------

    grouped = {
        video_id: group
        for video_id, group
        in df.groupby("videoid")
    }


    video_ids = list(
        grouped.keys()
    )


    # --------------------------------------------------------
    # Build rows
    # --------------------------------------------------------

    for index, (_, row) in enumerate(
        df.iterrows()
    ):

        source_path = Path(
            row["frame_path"]
        )


        source_video = row[
            "videoid"
        ]


        # ----------------------------------------------------
        # Choose a different reference video
        # ----------------------------------------------------

        possible_refs = [
            video_id
            for video_id in video_ids
            if video_id != source_video
        ]


        if not possible_refs:
            continue


        # Deterministic reference selection
        reference_video = random.choice(
            possible_refs
        )


        # ----------------------------------------------------
        # IMPORTANT:
        #
        # Use the EXACT reference frame that was used
        # to construct the reference bank.
        # ----------------------------------------------------

        reference_path = Path(
            reference_frames[
                reference_video
            ]["frame_path"]
        )


        # ----------------------------------------------------
        # Get the LUT assigned to this source sample
        # ----------------------------------------------------

        lut_name, lut = (
            selected_luts[index]
        )


        lut_stem = Path(
            lut_name
        ).stem


        # ----------------------------------------------------
        # Target path
        # ----------------------------------------------------

        target_path = (
            target_root
            / source_video
            / (
                source_path.stem
                + "_"
                + lut_stem
                + ".jpg"
            )
        )


        # ----------------------------------------------------
        # Styled reference path
        # ----------------------------------------------------

        styled_reference_path = (
            reference_root
            / reference_video
            / lut_stem
            / (
                reference_path.stem
                + ".jpg"
            )
        )


        # ----------------------------------------------------
        # Verify files
        # ----------------------------------------------------

        if not target_path.exists():
            continue


        if not styled_reference_path.exists():
            continue


        # ----------------------------------------------------
        # Add pair
        # ----------------------------------------------------

        rows.append({

            "source_path":
                str(source_path),

            "reference_path":
                str(
                    styled_reference_path
                ),

            "target_path":
                str(target_path),

            "lut_path":
                str(
                    LUT_DIR
                    / lut_name
                ),

            "source_video":
                source_video,

            "reference_video":
                reference_video,

            "timestamp":
                row["timestamp"],

            "lut":
                lut_name
        })


    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    PAIR_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    output_path = (
        PAIR_DIR
        / f"{split_name}.csv"
    )


    output = pd.DataFrame(
        rows
    )


    output.to_csv(
        output_path,
        index=False
    )


    print(
        f"\n{split_name}: "
        f"{len(output):,} pairs"
    )


    print(
        f"Saved: "
        f"{output_path}"
    )


    return output


# ============================================================
# PROCESS ONE SPLIT
# ============================================================

def process_split(
    split_name,
    max_samples
):

    # --------------------------------------------------------
    # Load frame split
    # --------------------------------------------------------

    csv_path = (
        SPLIT_DIR
        / f"{split_name}.csv"
    )


    df = pd.read_csv(
        csv_path
    )


    print(
        f"\n{'=' * 60}"
    )

    print(
        f"Split: {split_name}"
    )

    print(
        f"Loaded frames: "
        f"{len(df):,}"
    )

    print(
        f"{'=' * 60}"
    )


    # --------------------------------------------------------
    # Load LUTs
    # --------------------------------------------------------

    luts = load_luts()


    print(
        f"Loaded LUTs: "
        f"{len(luts)}"
    )


    # --------------------------------------------------------
    # Create reference bank
    # --------------------------------------------------------

    reference_frames = (
        create_reference_bank(
            df,
            luts,
            split_name
        )
    )


    # --------------------------------------------------------
    # Create targets
    #
    # This also determines which LUT each source receives.
    # --------------------------------------------------------

    sampled_df, selected_luts = (
        create_targets(
            df,
            luts,
            split_name,
            max_samples
        )
    )


    # --------------------------------------------------------
    # Create CSV using EXACT same samples + LUTs
    # --------------------------------------------------------

    output = create_pair_csv(
        sampled_df,
        selected_luts,
        reference_frames,
        luts,
        split_name
    )


    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    expected = min(
        max_samples,
        len(df)
    )


    print(
        f"\nExpected pairs: "
        f"{expected:,}"
    )

    print(
        f"Actual pairs: "
        f"{len(output):,}"
    )


    if len(output) != expected:

        print(
            "\nWARNING:"
        )

        print(
            "Some pairs could not be created."
        )

    else:

        print(
            "All requested pairs created successfully."
        )


# ============================================================
# MAIN
# ============================================================

def main():

    process_split(
        "train",
        TRAIN_SAMPLES
    )


    process_split(
        "val",
        VAL_SAMPLES
    )


    process_split(
        "test",
        TEST_SAMPLES
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()