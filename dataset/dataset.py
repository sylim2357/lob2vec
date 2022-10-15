from torch.utils.data import Dataset
import numpy as np
import torch


class LobDataset(Dataset):
    """Characterizes a dataset for PyTorch"""

    def __init__(self, data, k, num_classes, T):
        """Initialization"""
        self.k = k
        self.num_classes = num_classes
        self.T = T

        x = data.iloc[:, :-5].to_numpy()
        y = data.iloc[:, -5:].to_numpy()

        y = y[:, self.k] + 1
        self.length = len(x) - T + 1

        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)

    def __len__(self):
        """Denotes the total number of samples"""
        return self.length

    def __getitem__(self, index):
        """
        Generates samples of data
        Returns: (batch x n_views x features ...)
        """
        pass

    def data_prep(self, X, Y, T):
        [N, D] = X.shape
        df = np.array(X)
        dY = np.array(Y)
        dataY = dY[T - 1 : N]

        dataX = np.zeros((N - T + 1, T, D))
        for i in range(0, N - T + 1):
            dataX[i] = df[i : i + T]
        for i in range(T, N + 1):
            dataX[i - T] = df[i - T : i, :]

        return dataX, dataY


class DeepLobDataset(LobDataset):
    def __init__(self, data, k, num_classes, T):
        super().__init__(data, k, num_classes, T)

    def __getitem__(self, index):
        feat = self.x[index : index + self.T, :].float()
        feat1idx, _ = torch.randperm(len(feat))[:100].sort()
        feat2idx, _ = torch.randperm(len(feat))[:100].sort()
        feat1 = feat[feat1idx]
        feat2 = feat[feat2idx]
        # noise = torch.normal(0, 1, (feat.size(0), 20))
        # aug1 = torch.cat((feat[:, :20], noise), dim=1)
        #
        # feat_highs = feat[:, 20:]
        # feat_highs[:, ::2] = torch.normal(0, 1, (feat.size(0), 10))
        # aug2 = torch.cat((feat[:, :20], feat_highs), dim=1)
        #
        # return (
        #     torch.stack(
        #         (feat.unsqueeze(0), aug1.unsqueeze(0), aug2.unsqueeze(0)),
        #         dim=0,
        #     ),
        #     self.y[index + self.T - 1],
        # )
        return (feat1.unsqueeze(0), feat2.unsqueeze(0)), self.y[index+self.T-1]
        # return feat.unsqueeze(0).unsqueeze(0), self.y[index+self.T-1]


class TransLobDataset(LobDataset):
    def __init__(self, data, k, num_classes, T):
        super().__init__(data, k, num_classes, T)

    def __getitem__(self, index):
        feat = self.x[index : index + self.T, :].transpose(0, 1).float()
        # mask_idx = torch.rand(self.T * 40).ge(0.8).nonzero().reshape(-1)
        # noise = torch.normal(0, 1, (len(mask_idx),))
        # feat_masked = (
        #     feat.view(-1).scatter(0, mask_idx, noise).view(40, self.T)
        # )

        # feat_highs = feat[20:, :]
        # feat_highs[::2, :] = torch.normal(0, 1, (10, feat.size(1)))
        # aug2 = torch.cat((feat[:20, :], feat_highs), dim=0)

        # x = feat.unsqueeze(dim=0)

        # return (
        #     torch.stack((feat, feat_masked, aug2), dim=0),
        #     self.y[index + self.T - 1],
        # )
        return (feat.unsqueeze(0), self.y[index + self.T - 1])
        # return x, self.y[index + self.T - 1]


class DownstreamDeepLobDataset(LobDataset):
    def __init__(self, data, k, num_classes, T):
        super().__init__(data, k, num_classes, T)

    def __getitem__(self, index):
        feat = self.x[index : index + self.T, :].float()

        return feat.unsqueeze(0), self.y[index + self.T - 1]


class DownstreamTransLobDataset(LobDataset):
    def __init__(self, data, k, num_classes, T):
        super().__init__(data, k, num_classes, T)

    def __getitem__(self, index):
        feat = self.x[index : index + self.T, :].transpose(0, 1).float()

        return feat, self.y[index + self.T - 1]
