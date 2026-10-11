import os
import math
import torch
import numpy as np
import pandas as pd

from tqdm import tqdm

from Scripts.dataset import COGDataset
from models.cog_baseline import COGBaseline
from models.differentiable_lut import apply_lut


# ============================================================
# CONFIG
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINT = "F:/Cog/checkpoints/best.pt"
CSV_PATH = "F:/Cog/data/pairs/val.csv"

IMAGE_SIZE = 256
LUT_SIZE = 16

OUTPUT_DIR = "F:/Cog/evaluation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RESULTS_CSV = os.path.join(
    OUTPUT_DIR,
    "validation_metrics.csv"
)

LUT_RESULTS_CSV = os.path.join(
    OUTPUT_DIR,
    "metrics_by_lut.csv"
)


# ============================================================
# SSIM
# ============================================================

def ssim(
    img1,
    img2,
    C1=0.01 ** 2,
    C2=0.03 ** 2
):
    """
    Simple global SSIM implementation.

    img1/img2:
        Tensor [C,H,W]
        Values in [0,1]
    """

    mu1 = img1.mean()
    mu2 = img2.mean()

    sigma1 = ((img1 - mu1) ** 2).mean()
    sigma2 = ((img2 - mu2) ** 2).mean()

    covariance = (
        (img1 - mu1) *
        (img2 - mu2)
    ).mean()

    numerator = (
        (2 * mu1 * mu2 + C1) *
        (2 * covariance + C2)
    )

    denominator = (
        (mu1 ** 2 + mu2 ** 2 + C1) *
        (sigma1 + sigma2 + C2)
    )

    return (
        numerator / denominator
    ).item()


# ============================================================
# PSNR
# ============================================================

def psnr(
    prediction,
    target
):
    """
    PSNR assuming image values are in [0,1].
    """

    mse = torch.mean(
        (prediction - target) ** 2
    ).item()

    if mse == 0:
        return float("inf")

    return 10 * math.log10(
        1.0 / mse
    )


# ============================================================
# LOAD DATASET
# ============================================================

dataset = COGDataset(
    CSV_PATH,
    image_size=IMAGE_SIZE
)

print()
print("========================================")
print("COG METRIC EVALUATION")
print("========================================")
print()

print("Device:", DEVICE)
print("Validation samples:", len(dataset))


# ============================================================
# LOAD MODEL
# ============================================================

model = COGBaseline(
    freeze_encoder=True,
    lut_size=LUT_SIZE
).to(DEVICE)


checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE
)

if "model_state_dict" in checkpoint:

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

else:

    model.load_state_dict(
        checkpoint
    )


model.eval()

print("Model loaded successfully.")
print()


# ============================================================
# METRIC STORAGE
# ============================================================

all_results = []

lut_results = {}


# ============================================================
# EVALUATION LOOP
# ============================================================

with torch.no_grad():

    for i in tqdm(
        range(len(dataset)),
        desc="Evaluating"
    ):

        sample = dataset[i]

        source = (
            sample["source"]
            .unsqueeze(0)
            .to(DEVICE)
        )

        reference = (
            sample["reference"]
            .unsqueeze(0)
            .to(DEVICE)
        )

        target = (
            sample["target"]
            .unsqueeze(0)
            .to(DEVICE)
        )

        lut_name = sample["lut_name"]


        # ----------------------------------------------------
        # Generate LUT
        # ----------------------------------------------------

        predicted_lut = model(
            source,
            reference
        )


        # ----------------------------------------------------
        # Apply LUT
        # ----------------------------------------------------

        prediction = apply_lut(
            source,
            predicted_lut
        )

        prediction = prediction.clamp(
            0,
            1
        )


        # ----------------------------------------------------
        # Calculate metrics
        # ----------------------------------------------------

        l1 = torch.mean(
            torch.abs(
                prediction - target
            )
        ).item()


        mse = torch.mean(
            (prediction - target) ** 2
        ).item()


        psnr_value = psnr(
            prediction[0],
            target[0]
        )


        ssim_value = ssim(
            prediction[0],
            target[0]
        )


        # ----------------------------------------------------
        # Store result
        # ----------------------------------------------------

        result = {
            "index": i,
            "lut": lut_name,
            "l1": l1,
            "mse": mse,
            "psnr": psnr_value,
            "ssim": ssim_value
        }

        all_results.append(result)


        # ----------------------------------------------------
        # Per-LUT storage
        # ----------------------------------------------------

        if lut_name not in lut_results:
            lut_results[lut_name] = []

        lut_results[lut_name].append(
            result
        )


# ============================================================
# DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    all_results
)


# ============================================================
# OVERALL RESULTS
# ============================================================

overall_l1 = results_df["l1"].mean()
overall_mse = results_df["mse"].mean()
overall_psnr = results_df["psnr"].mean()
overall_ssim = results_df["ssim"].mean()


print()
print("========================================")
print("OVERALL RESULTS")
print("========================================")
print()

print(
    f"L1   : {overall_l1:.6f}"
)

print(
    f"MSE  : {overall_mse:.6f}"
)

print(
    f"PSNR : {overall_psnr:.4f} dB"
)

print(
    f"SSIM : {overall_ssim:.6f}"
)


# ============================================================
# SAVE INDIVIDUAL RESULTS
# ============================================================

results_df.to_csv(
    RESULTS_CSV,
    index=False
)


# ============================================================
# PER-LUT RESULTS
# ============================================================

lut_summary = []


for lut_name, values in lut_results.items():

    lut_df = pd.DataFrame(
        values
    )

    lut_summary.append({
        "lut": lut_name,
        "samples": len(lut_df),
        "l1": lut_df["l1"].mean(),
        "mse": lut_df["mse"].mean(),
        "psnr": lut_df["psnr"].mean(),
        "ssim": lut_df["ssim"].mean()
    })


lut_summary_df = pd.DataFrame(
    lut_summary
)


lut_summary_df = lut_summary_df.sort_values(
    "lut"
)


# ============================================================
# PRINT PER-LUT RESULTS
# ============================================================

print()
print("========================================")
print("RESULTS BY LUT")
print("========================================")
print()

print(
    lut_summary_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# ============================================================
# SAVE PER-LUT RESULTS
# ============================================================

lut_summary_df.to_csv(
    LUT_RESULTS_CSV,
    index=False
)


# ============================================================
# FINISHED
# ============================================================

print()
print("========================================")
print("EVALUATION COMPLETE")
print("========================================")
print()

print(
    "Individual results:"
)

print(
    RESULTS_CSV
)

print()

print(
    "Per-LUT results:"
)

print(
    LUT_RESULTS_CSV
)