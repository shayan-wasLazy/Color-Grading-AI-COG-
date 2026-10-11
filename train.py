from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from Scripts.dataset import COGDataset
from models.cog_baseline import COGBaseline
from models.differentiable_lut import apply_lut


# ============================================================
# CONFIG
# ============================================================

ROOT = Path("F:/Cog")

TRAIN_CSV = (
    ROOT
    / "data"
    / "pairs"
    / "train.csv"
)

VAL_CSV = (
    ROOT
    / "data"
    / "pairs"
    / "val.csv"
)

CHECKPOINT_DIR = (
    ROOT
    / "checkpoints"
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ------------------------------------------------------------
# Training settings
# ------------------------------------------------------------

IMAGE_SIZE = 256

LUT_SIZE = 16

BATCH_SIZE = 2

EPOCHS = 2

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

NUM_WORKERS = 0

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# PRINT DEVICE
# ============================================================

print(
    "Device:",
    DEVICE
)


if DEVICE == "cuda":

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    print(
        "CUDA:",
        torch.version.cuda
    )


# ============================================================
# DATASETS
# ============================================================

print(
    "\nLoading datasets..."
)


train_dataset = COGDataset(
    str(TRAIN_CSV),
    image_size=IMAGE_SIZE
)


val_dataset = COGDataset(
    str(VAL_CSV),
    image_size=IMAGE_SIZE
)


print(
    "Train samples:",
    len(train_dataset)
)

print(
    "Validation samples:",
    len(val_dataset)
)


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE == "cuda")
)


val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE == "cuda")
)


# ============================================================
# MODEL
# ============================================================

model = COGBaseline(
    freeze_encoder=True,
    lut_size=LUT_SIZE
).to(DEVICE)


print(
    "\nModel created."
)


# ============================================================
# COUNT TRAINABLE PARAMETERS
# ============================================================

trainable_params = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)


total_params = sum(
    parameter.numel()
    for parameter in model.parameters()
)


print(
    "Total parameters:",
    f"{total_params:,}"
)

print(
    "Trainable parameters:",
    f"{trainable_params:,}"
)


# ============================================================
# LOSS
# ============================================================

criterion = nn.L1Loss()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    filter(
        lambda parameter:
            parameter.requires_grad,
        model.parameters()
    ),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# MIXED PRECISION
# ============================================================

use_amp = (
    DEVICE == "cuda"
)


scaler = torch.amp.GradScaler(
    "cuda",
    enabled=use_amp
)


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch():

    model.train()

    running_loss = 0.0

    total_samples = 0


    for batch_index, batch in enumerate(
        train_loader
    ):

        # ----------------------------------------------------
        # Load batch
        # ----------------------------------------------------

        source = batch[
            "source"
        ].to(
            DEVICE,
            non_blocking=True
        )


        reference = batch[
            "reference"
        ].to(
            DEVICE,
            non_blocking=True
        )


        target = batch[
            "target"
        ].to(
            DEVICE,
            non_blocking=True
        )


        # ----------------------------------------------------
        # Clear gradients
        # ----------------------------------------------------

        optimizer.zero_grad(
            set_to_none=True
        )


        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        with torch.autocast(
            device_type=DEVICE,
            dtype=torch.float16,
            enabled=use_amp
        ):

            predicted_lut = model(
                source,
                reference
            )


            predicted_image = apply_lut(
                source,
                predicted_lut
            )


            loss = criterion(
                predicted_image,
                target
            )


        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        scaler.scale(
            loss
        ).backward()


        # ----------------------------------------------------
        # Update model
        # ----------------------------------------------------

        scaler.step(
            optimizer
        )


        scaler.update()


        # ----------------------------------------------------
        # Track loss
        # ----------------------------------------------------

        batch_size = (
            source.shape[0]
        )


        running_loss += (
            loss.item()
            * batch_size
        )


        total_samples += (
            batch_size
        )


        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            batch_index + 1
        ) % 100 == 0:

            current_loss = (
                running_loss
                / total_samples
            )

            print(
                f"  Batch "
                f"{batch_index + 1:,}"
                f"/{len(train_loader):,}"
                f" | Loss: "
                f"{current_loss:.6f}"
            )


    epoch_loss = (
        running_loss
        / total_samples
    )


    return epoch_loss


# ============================================================
# VALIDATION
# ============================================================

def validate():

    model.eval()

    running_loss = 0.0

    total_samples = 0


    with torch.no_grad():

        for batch in val_loader:

            # ------------------------------------------------
            # Load batch
            # ------------------------------------------------

            source = batch[
                "source"
            ].to(
                DEVICE,
                non_blocking=True
            )


            reference = batch[
                "reference"
            ].to(
                DEVICE,
                non_blocking=True
            )


            target = batch[
                "target"
            ].to(
                DEVICE,
                non_blocking=True
            )


            # ------------------------------------------------
            # Forward
            # ------------------------------------------------

            with torch.autocast(
                device_type=DEVICE,
                dtype=torch.float16,
                enabled=use_amp
            ):

                predicted_lut = model(
                    source,
                    reference
                )


                predicted_image = apply_lut(
                    source,
                    predicted_lut
                )


                loss = criterion(
                    predicted_image,
                    target
                )


            batch_size = (
                source.shape[0]
            )


            running_loss += (
                loss.item()
                * batch_size
            )


            total_samples += (
                batch_size
            )


    validation_loss = (
        running_loss
        / total_samples
    )


    return validation_loss


# ============================================================
# TRAINING LOOP
# ============================================================

best_val_loss = float(
    "inf"
)


print(
    "\nStarting training..."
)

print(
    "=" * 60
)


for epoch in range(
    EPOCHS
):

    print(
        f"\nEpoch "
        f"{epoch + 1}/{EPOCHS}"
    )

    print(
        "-" * 60
    )


    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    train_loss = train_one_epoch()


    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    val_loss = validate()


    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print(
        f"\nEpoch "
        f"{epoch + 1}/{EPOCHS}"
    )

    print(
        f"Train Loss: "
        f"{train_loss:.6f}"
    )

    print(
        f"Val Loss:   "
        f"{val_loss:.6f}"
    )


    # --------------------------------------------------------
    # Save latest checkpoint
    # --------------------------------------------------------

    latest_checkpoint = (
        CHECKPOINT_DIR
        / "latest.pt"
    )


    torch.save(
        {
            "epoch":
                epoch + 1,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "train_loss":
                train_loss,

            "val_loss":
                val_loss
        },
        latest_checkpoint
    )


    # --------------------------------------------------------
    # Save best checkpoint
    # --------------------------------------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss


        best_checkpoint = (
            CHECKPOINT_DIR
            / "best.pt"
        )


        torch.save(
            {
                "epoch":
                    epoch + 1,

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "train_loss":
                    train_loss,

                "val_loss":
                    val_loss
            },
            best_checkpoint
        )


        print(
            "\nNew best model saved:"
        )

        print(
            best_checkpoint
        )


    # --------------------------------------------------------
    # GPU memory
    # --------------------------------------------------------

    if DEVICE == "cuda":

        allocated = (
            torch.cuda.memory_allocated()
            / 1024**3
        )

        reserved = (
            torch.cuda.memory_reserved()
            / 1024**3
        )


        print(
            f"GPU memory: "
            f"{allocated:.2f} GB allocated | "
            f"{reserved:.2f} GB reserved"
        )


print(
    "\n"
    + "=" * 60
)

print(
    "Training complete."
)

print(
    "Best validation loss:",
    f"{best_val_loss:.6f}"
)

print(
    "Best checkpoint:",
    CHECKPOINT_DIR / "best.pt"
)