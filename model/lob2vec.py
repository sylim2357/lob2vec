import torch.nn.functional as F
import torch.nn as nn
import torch

from model.deeplob import DeepLob
from model.translob import TransLobEncoder


class DeepLobPreText(nn.Module):
    def __init__(self, deeplob=None, norm=False):
        super().__init__()
        if deeplob:
            self.enc = deeplob
        else:
            self.enc = DeepLob(norm=norm)
        self.norm = norm
        self.gelu = nn.GELU()
        # self.proj = nn.Linear(64, 128)
        self.proj = nn.Sequential(
            nn.Linear(64, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 256),
        )
        # self.proj = nn.Sequential(
        #     nn.Linear(64, 64),
        #     nn.GELU(),
        #     nn.Linear(64, 64),
        # )

    def forward(self, x):
        x = self.enc(x)
        # x = self.gelu(x)
        x = self.proj(x)
        if self.norm:
            x = F.normalize(x, dim=1)

        return x


class DeepLobPred(nn.Module):
    def __init__(self, deeplob=None, norm=False):
        super().__init__()
        if deeplob:
            self.deeplob = deeplob
        else:
            self.deeplob = DeepLob(norm=norm)
        self.gelu = nn.GELU()
        self.fc = nn.Sequential(
            nn.Linear(64, 64),
            nn.Linear(64, 3)
        )
        # self.fc = nn.Linear(64, 3)

    def forward(self, x):
        # with torch.no_grad():
        x = self.deeplob(x)
        x = self.gelu(x)
        x = self.fc(x)
        pred = torch.softmax(x, dim=1)

        return pred


class TransLobPreText(nn.Module):
    def __init__(self, enc=None, norm=False):
        super().__init__()
        if enc:
            self.enc = enc
        else:
            self.enc = TransLobEncoder(norm=norm)

        self.norm = norm
        self.gelu = nn.GELU()
        self.proj = nn.Sequential(
            nn.Linear(512, 512),
            nn.LayerNorm(512, eps=1e-05),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512, eps=1e-05),
            nn.GELU(),
            nn.Linear(1024, 1024),
        )

    def forward(self, x):
        x = self.enc(x)
        x = self.gelu(x)
        x = self.proj(x)
        if self.norm:
            x = F.normalize(x, dim=1)

        return x


class TransLobPred(nn.Module):
    def __init__(self, enc=None, norm=False):
        super().__init__()
        if enc:
            self.enc = enc
        else:
            self.enc = TransLobEncoder(norm=norm)

        self.gelu = nn.GELU()
        self.fc = nn.Linear(512, 3)

    def forward(self, x):
        with torch.no_grad():
            x = self.enc(x)
        x = self.gelu(x)
        x = self.fc(x)
        x = torch.softmax(x, dim=1)

        return x
