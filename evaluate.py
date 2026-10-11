import os
import torch
import matplotlib.pyplot as plt

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

# Number of samples to visualize
NUM_SAMPLES = 10

OUTPUT_DIR = "F:/Cog/evaluation"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD DATASET
# ============================================================

dataset = COGDataset(
    CSV_PATH,
    image_size=IMAGE_SIZE
)

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
    model.load_state_dict(checkpoint)

model.eval()

print("Model loaded successfully.")


# ============================================================
# VISUAL EVALUATION
# ============================================================

with torch.no_grad():

    for i in range(NUM_SAMPLES):

        sample = dataset[i]

        source = sample["source"].unsqueeze(0).to(DEVICE)
        reference = sample["reference"].unsqueeze(0).to(DEVICE)
        target = sample["target"].unsqueeze(0).to(DEVICE)

        # ----------------------------------------------------
        # Predict LUT
        # ----------------------------------------------------

        predicted_lut = model(
            source,
            reference
        )

        # ----------------------------------------------------
        # Apply predicted LUT
        # ----------------------------------------------------

        predicted = apply_lut(
            source,
            predicted_lut
        )

        # ----------------------------------------------------
        # Convert tensors to images
        # ----------------------------------------------------

        source_img = (
            source[0]
            .cpu()
            .permute(1, 2, 0)
            .numpy()
        )

        reference_img = (
            reference[0]
            .cpu()
            .permute(1, 2, 0)
            .numpy()
        )

        target_img = (
            target[0]
            .cpu()
            .permute(1, 2, 0)
            .numpy()
        )

        predicted_img = (
            predicted[0]
            .cpu()
            .permute(1, 2, 0)
            .numpy()
        )

        predicted_img = predicted_img.clip(0, 1)

        # ----------------------------------------------------
        # Plot
        # ----------------------------------------------------

        fig, axes = plt.subplots(
            1,
            4,
            figsize=(16, 4)
        )

        axes[0].imshow(source_img)
        axes[0].set_title("Source")

        axes[1].imshow(reference_img)
        axes[1].set_title("Reference")

        axes[2].imshow(target_img)
        axes[2].set_title("Ground Truth")

        axes[3].imshow(predicted_img)
        axes[3].set_title("Predicted")

        for ax in axes:
            ax.axis("off")

        plt.tight_layout()

        save_path = os.path.join(
            OUTPUT_DIR,
            f"sample_{i:02d}.png"
        )

        plt.savefig(
            save_path,
            dpi=150,
            bbox_inches="tight"
        )

        plt.close()

        print(
            f"[{i + 1}/{NUM_SAMPLES}] "
            f"Saved: {save_path}"
        )


print()
print("Visual evaluation complete.")
print("Results saved to:")
print(OUTPUT_DIR)