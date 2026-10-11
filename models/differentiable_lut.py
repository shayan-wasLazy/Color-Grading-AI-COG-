import torch


def apply_lut(image, lut):

    """
    image:
        B x 3 x H x W
        values [0, 1]

    lut:
        B x D x D x D x 3

    Returns:
        B x 3 x H x W
    """

    B, C, H, W = image.shape

    D = lut.shape[1]

    # RGB coordinates
    coords = (
        image
        .permute(0, 2, 3, 1)
        * (D - 1)
    )

    r = coords[..., 0]
    g = coords[..., 1]
    b = coords[..., 2]

    r0 = torch.floor(r).long()
    g0 = torch.floor(g).long()
    b0 = torch.floor(b).long()

    r1 = torch.clamp(
        r0 + 1,
        max=D - 1
    )

    g1 = torch.clamp(
        g0 + 1,
        max=D - 1
    )

    b1 = torch.clamp(
        b0 + 1,
        max=D - 1
    )

    dr = r - r0.float()
    dg = g - g0.float()
    db = b - b0.float()

    batch_idx = (
        torch.arange(B, device=image.device)
        .view(B, 1, 1)
        .expand(B, H, W)
    )

    def lookup(
        ri,
        gi,
        bi
    ):

        return lut[
            batch_idx,
            ri,
            gi,
            bi
        ]

    c000 = lookup(r0, g0, b0)
    c001 = lookup(r0, g0, b1)

    c010 = lookup(r0, g1, b0)
    c011 = lookup(r0, g1, b1)

    c100 = lookup(r1, g0, b0)
    c101 = lookup(r1, g0, b1)

    c110 = lookup(r1, g1, b0)
    c111 = lookup(r1, g1, b1)

    c00 = (
        c000 * (1 - db[..., None])
        + c001 * db[..., None]
    )

    c01 = (
        c010 * (1 - db[..., None])
        + c011 * db[..., None]
    )

    c10 = (
        c100 * (1 - db[..., None])
        + c101 * db[..., None]
    )

    c11 = (
        c110 * (1 - db[..., None])
        + c111 * db[..., None]
    )

    c0 = (
        c00 * (1 - dg[..., None])
        + c01 * dg[..., None]
    )

    c1 = (
        c10 * (1 - dg[..., None])
        + c11 * dg[..., None]
    )

    output = (
        c0 * (1 - dr[..., None])
        + c1 * dr[..., None]
    )

    return output.permute(
        0, 3, 1, 2
    )



    