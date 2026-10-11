import numpy as np
import pandas as pd
import torch

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class COGDataset(Dataset):

    def __init__(
        self,
        csv_path,
        image_size=256
    ):

        self.df = pd.read_csv(csv_path)

        self.transform = transforms.Compose([
            transforms.Resize(
                (image_size, image_size)
            ),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.df)

    def load_image(self, path):

        image = Image.open(
            path
        ).convert("RGB")

        return self.transform(image)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        source = self.load_image(
            row["source_path"]
        )

        reference = self.load_image(
            row["reference_path"]
        )

        target = self.load_image(
            row["target_path"]
        )

        lut = np.load(
            row["lut_path"]
        )

        lut = torch.from_numpy(
            lut
        ).float()

        return {
            "source": source,
            "reference": reference,
            "target": target,
            "lut": lut,

            "source_video":
                row["source_video"],

            "reference_video":
                row["reference_video"],

            "timestamp":
                row["timestamp"],

            "lut_name":
                row["lut"]
        }