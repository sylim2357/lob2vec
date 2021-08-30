import torch.nn.functional as F
import torch.nn as nn
import torch

from deeplob import DeepLob
from translob import (
    FeatureExtractor,
    PositionalEncodingLayer,
    TransformerAggregator,
)


class DeepLobPreText(nn.Module):
    def __init__(self, deeplob=None):
        super().__init__()
        if deeplob:
            self.deeplob = deeplob
        else:
            self.deeplob = DeepLob()
        self.gelu = nn.GELU()
        # self.proj = nn.Linear(64, 128)
        self.proj = nn.Sequential(
            nn.Linear(64, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 256),
        )

    def forward(self, x):
        x = self.deeplob(x)
        x = self.gelu(x)
        x = self.proj(x)
        # x = F.normalize(x, dim=1)

        return x


class DeepLobPred(nn.Module):
    def __init__(self, deeplob=None):
        super().__init__()
        if deeplob:
            self.deeplob = deeplob
        else:
            self.deeplob = DeepLob()
        self.gelu = nn.GELU()
        self.fc = nn.Linear(256, 3)

    def forward(self, x):
        with torch.no_grad():
            x = self.deeplob(x)
        x = self.gelu(x)
        x = self.fc(x)
        pred = torch.softmax(x, dim=1)

        return pred


class TransLobEncoder(nn.Module):
    def __init__(self, extractor=None, pos_encoding=None, aggregator=None):
        super().__init__()
        if extractor:
            self.extractor = extractor
        else:
            self.extractor = FeatureExtractor(
                [
                    (14, 2, 1, 1),
                    (14, 2, 1, 2),
                    (14, 2, 1, 4),
                    (14, 2, 1, 8),
                    (14, 2, 1, 16),
                ],
                40,
                0.1,
                nn.ReLU(),
            )
        if pos_encoding:
            self.pos_encoding = pos_encoding
        else:
            self.pos_encoding = PositionalEncodingLayer()
        if aggregator:
            self.aggregator = aggregator
        else:
            self.aggregator = TransformerAggregator(
                d_model=64,
                n_head=4,
                n_encoder_layers=2,
                dim_feedforward=256,
                dropout=0.1,
                activation='relu',
                tr_weight_share=True,
            )

        self.fc = nn.Linear(15, 64)

        self.layernorm = nn.LayerNorm([14, 100])
        self.fc1 = nn.Linear(100 * 64, 128)

    def forward(self, x):
        x = self.extractor(x)
        x = self.layernorm(x)
        x = self.pos_encoding(x)
        x = x.permute([0, 2, 1])
        x = self.fc(x)
        x = self.aggregator(x)
        x = x.permute([0, 2, 1])
        x = x.flatten(1)
        x = self.fc1(x)
        # x = F.normalize(x, dim=1)

        return x


class TransLobPreText(nn.Module):
    def __init__(self, enc=None):
        super().__init__()
        if enc:
            self.enc = enc
        else:
            self.enc = TransLobEncoder()

        self.gelu = nn.GELU()
        norm = nn.LayerNorm(512)
        self.proj = nn.Sequential(
            nn.Linear(128, 512),
            norm,
            nn.GELU(),
            nn.Linear(512, 512),
            norm,
            nn.GELU(),
            nn.Linear(512, 512),
        )

    def forward(self, x):
        x = self.enc(x)
        x = self.gelu(x)
        x = self.proj(x)
        # x = F.normalize(x, dim=1)

        return x


class TransLobPred(nn.Module):
    def __init__(self, enc=None):
        super().__init__()
        if enc:
            self.enc = enc
        else:
            self.enc = TransLobEncoder()

        self.gelu = nn.GELU()
        self.fc = nn.Linear(64, 3)

    def forward(self, x):
        x = self.enc(x)
        x = self.gelu(x)
        x = self.fc(x)
        x = torch.softmax(x, dim=1)

        return x