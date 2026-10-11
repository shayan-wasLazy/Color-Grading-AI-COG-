from pathlib import Path
import numpy as np
import pandas as pd


ROOT = Path("F:/Cog")

LUT_DIR = ROOT / "data" / "luts"

LUT_SIZE = 33


def clamp(x):
    return np.clip(x, 0.0, 1.0)


def apply_contrast(x, amount):

    return clamp(
        (x - 0.5) * amount + 0.5
    )


def apply_saturation(rgb, amount):

    luminance = (
        0.2126 * rgb[..., 0]
        + 0.7152 * rgb[..., 1]
        + 0.0722 * rgb[..., 2]
    )

    luminance = luminance[..., None]

    return clamp(
        luminance
        + (rgb - luminance) * amount
    )


def create_lut(
    name,
    contrast=1.0,
    saturation=1.0,
    temperature=0.0,
    tint=0.0,
    gamma=1.0,
    lift=0.0,
    red_gain=1.0,
    green_gain=1.0,
    blue_gain=1.0
):

    values = np.linspace(
        0,
        1,
        LUT_SIZE
    )

    r, g, b = np.meshgrid(
        values,
        values,
        values,
        indexing="ij"
    )

    rgb = np.stack(
        [r, g, b],
        axis=-1
    )

    # Contrast
    rgb = apply_contrast(
        rgb,
        contrast
    )

    # Saturation
    rgb = apply_saturation(
        rgb,
        saturation
    )

    # Temperature
    rgb[..., 0] += temperature
    rgb[..., 2] -= temperature

    # Tint
    rgb[..., 1] += tint

    # Channel gains
    rgb[..., 0] *= red_gain
    rgb[..., 1] *= green_gain
    rgb[..., 2] *= blue_gain

    # Lift
    rgb += lift

    # Gamma
    rgb = np.power(
        clamp(rgb),
        gamma
    )

    return clamp(rgb).astype(
        np.float32
    )


def main():

    LUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    luts = [

        {
            "name": "neutral",
            "contrast": 1.0,
            "saturation": 1.0,
            "temperature": 0,
            "tint": 0,
            "gamma": 1.0
        },

        {
            "name": "warm",
            "contrast": 1.05,
            "saturation": 1.10,
            "temperature": 0.06,
            "tint": 0.01,
            "gamma": 0.98
        },

        {
            "name": "cool",
            "contrast": 1.05,
            "saturation": 1.05,
            "temperature": -0.06,
            "tint": 0,
            "gamma": 1.02
        },

        {
            "name": "teal_orange",
            "contrast": 1.12,
            "saturation": 1.20,
            "temperature": 0.03,
            "tint": -0.015,
            "gamma": 0.96,
            "red_gain": 1.08,
            "green_gain": 0.98,
            "blue_gain": 1.08
        },

        {
            "name": "desaturated",
            "contrast": 1.08,
            "saturation": 0.65,
            "temperature": 0,
            "tint": 0,
            "gamma": 1.02
        },

        {
            "name": "high_contrast",
            "contrast": 1.30,
            "saturation": 1.10,
            "temperature": 0,
            "tint": 0,
            "gamma": 0.95
        },

        {
            "name": "matte",
            "contrast": 0.85,
            "saturation": 0.85,
            "temperature": 0.01,
            "tint": 0,
            "gamma": 1.05,
            "lift": 0.025
        },

        {
            "name": "bleach",
            "contrast": 1.20,
            "saturation": 0.45,
            "temperature": -0.01,
            "tint": 0.005,
            "gamma": 0.95
        },

        {
            "name": "green",
            "contrast": 1.05,
            "saturation": 1.05,
            "temperature": -0.02,
            "tint": 0.045,
            "gamma": 1.0
        },

        {
            "name": "blue_shadows",
            "contrast": 1.10,
            "saturation": 1.10,
            "temperature": -0.03,
            "tint": 0,
            "gamma": 1.0,
            "blue_gain": 1.10
        },

        {
            "name": "warm_highlights",
            "contrast": 1.08,
            "saturation": 1.12,
            "temperature": 0.055,
            "tint": 0.01,
            "gamma": 0.94,
            "red_gain": 1.06
        },

        {
            "name": "cinematic_dark",
            "contrast": 1.18,
            "saturation": 0.90,
            "temperature": -0.015,
            "tint": 0.005,
            "gamma": 1.10
        }

    ]

    metadata = []

    for index, params in enumerate(luts):

        name = params.pop("name")

        lut = create_lut(
            name,
            **params
        )

        filename = (
            f"lut_{index:03d}_{name}.npy"
        )

        np.save(
            LUT_DIR / filename,
            lut
        )

        metadata.append({
            "lut_id": index,
            "name": name,
            "filename": filename,
            **params
        })

        print(
            f"Created {filename}"
        )

    metadata_df = pd.DataFrame(
        metadata
    )

    metadata_df.to_csv(
        LUT_DIR / "lut_metadata.csv",
        index=False
    )

    print(
        f"\nCreated {len(luts)} LUTs"
    )


if __name__ == "__main__":
    main()