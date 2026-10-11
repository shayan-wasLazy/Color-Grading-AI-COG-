import numpy as np


def apply_lut(image, lut):

    """
    image:
        RGB uint8
        H x W x 3

    lut:
        D x D x D x 3
        float32
    """

    image = (
        image.astype(np.float32)
        / 255.0
    )

    size = lut.shape[0]

    coords = image * (
        size - 1
    )

    r = coords[..., 0]
    g = coords[..., 1]
    b = coords[..., 2]

    r0 = np.floor(r).astype(np.int32)
    g0 = np.floor(g).astype(np.int32)
    b0 = np.floor(b).astype(np.int32)

    r1 = np.minimum(
        r0 + 1,
        size - 1
    )

    g1 = np.minimum(
        g0 + 1,
        size - 1
    )

    b1 = np.minimum(
        b0 + 1,
        size - 1
    )

    dr = r - r0
    dg = g - g0
    db = b - b0

    c000 = lut[r0, g0, b0]
    c001 = lut[r0, g0, b1]

    c010 = lut[r0, g1, b0]
    c011 = lut[r0, g1, b1]

    c100 = lut[r1, g0, b0]
    c101 = lut[r1, g0, b1]

    c110 = lut[r1, g1, b0]
    c111 = lut[r1, g1, b1]

    c00 = (
        c000 * (1 - db)[..., None]
        + c001 * db[..., None]
    )

    c01 = (
        c010 * (1 - db)[..., None]
        + c011 * db[..., None]
    )

    c10 = (
        c100 * (1 - db)[..., None]
        + c101 * db[..., None]
    )

    c11 = (
        c110 * (1 - db)[..., None]
        + c111 * db[..., None]
    )

    c0 = (
        c00 * (1 - dg)[..., None]
        + c01 * dg[..., None]
    )

    c1 = (
        c10 * (1 - dg)[..., None]
        + c11 * dg[..., None]
    )

    output = (
        c0 * (1 - dr)[..., None]
        + c1 * dr[..., None]
    )

    return np.clip(
        output * 255,
        0,
        255
    ).astype(np.uint8)