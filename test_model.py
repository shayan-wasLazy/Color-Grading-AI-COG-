import torch

from Scripts.dataset import COGDataset
from models.cog_baseline import COGBaseline
from models.differentiable_lut import apply_lut


device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)


dataset = COGDataset(
    "F:/Cog/data/pairs/train.csv",
    image_size=256
)


sample = dataset[0]


source = sample["source"].unsqueeze(0)
reference = sample["reference"].unsqueeze(0)

source = source.to(device)
reference = reference.to(device)


model = COGBaseline(
    freeze_encoder=True,
    lut_size=16
).to(device)


# with torch.no_grad():

predicted_lut = model(
    source,
    reference
)

output = apply_lut(
    source,
    predicted_lut
)


print(
    "Source:",
    source.shape
)

print(
    "Reference:",
    reference.shape
)

print(
    "Predicted LUT:",
    predicted_lut.shape
)

print(
    "Output:",
    output.shape
)

print(
    "Output range:",
    output.min().item(),
    output.max().item()
)

# -------------------------
# Gradient test
# -------------------------

predicted_lut.retain_grad()

loss = output.mean()

loss.backward()

print(
    "LUT generator gradient:",
    model.lut_generator.network[0].weight.grad is None
)

print(
    "Gradient mean:",
    model.lut_generator.network[0].weight.grad.abs().mean().item()
)

print(
    "LUT gradient is None:",
    predicted_lut.grad is None
)

print(
    "LUT gradient shape:",
    predicted_lut.grad.shape
)

print(
    "LUT gradient mean:",
    predicted_lut.grad.abs().mean().item()
)

