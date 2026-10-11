from torchvision import transforms
import torch
import torch.nn as nn
from torchvision.models import (
    resnet50,
    ResNet50_Weights
)


class ResNetEncoder(nn.Module):

    def __init__(
        self,
        pretrained=True,
        freeze=True
    ):

        super().__init__()

        weights = (
            ResNet50_Weights.DEFAULT
            if pretrained
            else None
        )

        backbone = resnet50(
            weights=weights
        )

        # Remove classification layer
        self.features = nn.Sequential(
            *list(backbone.children())[:-1]
        )

        self.feature_dim = 2048

        if freeze:

            for param in self.features.parameters():
                param.requires_grad = False
                
        self.normalize = transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406
            ],
            std=[
                0.229,
                0.224,
                0.225
            ]
        )


    def forward(self, x):

        x = self.normalize(x)
        x = self.features(x)

        return x.flatten(1)


class LUTGenerator(nn.Module):

    def __init__(
        self,
        feature_dim=2048,
        lut_size=16
    ):

        super().__init__()

        self.lut_size = lut_size

        output_dim = (
            lut_size
            * lut_size
            * lut_size
            * 3
        )

        self.network = nn.Sequential(

            nn.Linear(
                feature_dim,
                1024
            ),

            nn.ReLU(),

            nn.Linear(
                1024,
                2048
            ),

            nn.ReLU(),

            nn.Linear(
                2048,
                output_dim
            ),

            nn.Sigmoid()
        )


    def forward(self, x):

        lut = self.network(x)

        lut = lut.view(
            -1,
            self.lut_size,
            self.lut_size,
            self.lut_size,
            3
        )

        return lut


class COGBaseline(nn.Module):

    def __init__(
        self,
        freeze_encoder=True,
        lut_size=16
    ):

        super().__init__()

        self.encoder = ResNetEncoder(
            pretrained=True,
            freeze=freeze_encoder
        )

        self.lut_generator = LUTGenerator(
            feature_dim=2048,
            lut_size=lut_size
        )


    def forward(
        self,
        source,
        reference
    ):

        source_features = self.encoder(
            source
        )

        reference_features = self.encoder(
            reference
        )

        # Reference-style difference
        style_difference = (
            reference_features
            - source_features
        )

        lut = self.lut_generator(
            style_difference
        )

        return lut


