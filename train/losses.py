import torch.nn.functional as F
import torch.nn as nn
import torch


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(
        self, temperature=0.07, contrast_mode='all', base_temperature=0.07
    ):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (
            torch.device('cuda') if features.is_cuda else torch.device('cpu')
        )

        if len(features.shape) < 3:
            raise ValueError(
                '`features` needs to be [bsz, n_views, ...],'
                'at least 3 dimensions are required'
            )
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError(
                    'Num of labels does not match num of features'
                )
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature
        )
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0,
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class IMixSupConLoss(nn.Module):
    def __init__(
        self, temperature=0.07, contrast_mode='all', base_temperature=0.07
    ):
        super(IMixSupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        device = (
            torch.device('cuda') if features.is_cuda else torch.device('cpu')
        )

        if len(features.shape) < 3:
            raise ValueError(
                '`features` needs to be [bsz, n_views, ...],'
                'at least 3 dimensions are required'
            )
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        n_views = features.shape[1]

        label_mask = torch.eq(
            torch.arange(3).view(-1, 1).to(device), labels
        ).float()
        num_idx = label_mask.sum(1).int()
        label_idx = label_mask.nonzero()[:, 1].split(num_idx.tolist())
        y1_list = []

        for l in label_idx:
            z1 = features[l]
            z1 = z1.reshape((len(l) * n_views, -1))
            z2_idx = torch.randperm(z1.size(0))
            z2 = z1[z2_idx]
            lam = (
                torch.distributions.beta.Beta(1, 1)
                .sample((z1.size(0), 1))
                .to(device)
            )
            z1_interpolate = lam * z1 + (1 - lam) * z2
            y1_list.append(z1_interpolate)

        contrast_feature = torch.cat(y1_list, dim=0)
        labels = torch.zeros(len(label_idx[0]))
        labels = torch.cat((labels, torch.ones(len(label_idx[1]))), dim=0)
        labels = (
            torch.cat((labels, torch.ones(len(label_idx[2])) + 1), dim=0)
            .contiguous()
            .view(-1, 1)
            .to(device)
        )
        mask = torch.eq(labels, labels.T).float().to(device)  # be careful
        contrast_count = features.shape[1]
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature
        )
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0,
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class VICLoss(nn.Module):
    def __init__(self, num_classes=3):
        super(VICLoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, features, aug, labels=None, mask=None, print_c=False):
        loss, c = self.vicreg_loss_func(features, aug)

        if print_c:
            print(c)
            self.cross_cov = c

        return loss

    def invariance_loss(
        self: nn.Module, z1: torch.Tensor, z2: torch.Tensor
    ) -> torch.Tensor:
        """Computes mse loss given batch of projected features z1 from view 1 and
        projected features z2 from view 2.
        Args:
            z1 (torch.Tensor): NxD Tensor containing projected features from view 1.
            z2 (torch.Tensor): NxD Tensor containing projected features from view 2.
        Returns:
            torch.Tensor: invariance loss (mean squared error).
        """

        return F.mse_loss(z1, z2)

    def variance_loss(
        self: nn.Module, z1: torch.Tensor, z2: torch.Tensor
    ) -> torch.Tensor:
        """Computes variance loss given batch of projected features z1 from view 1 and
        projected features z2 from view 2.
        Args:
            z1 (torch.Tensor): NxD Tensor containing projected features from view 1.
            z2 (torch.Tensor): NxD Tensor containing projected features from view 2.
        Returns:
            torch.Tensor: variance regularization loss.
        """

        eps = 1e-4
        std_z1 = torch.sqrt(z1.var(dim=0) + eps)
        std_z2 = torch.sqrt(z2.var(dim=0) + eps)
        std_loss = torch.mean(F.relu(1 - std_z1)) + torch.mean(
            F.relu(1 - std_z2)
        )
        return std_loss

    def covariance_loss(
        self: nn.Module, z1: torch.Tensor, z2: torch.Tensor
    ) -> torch.Tensor:
        """Computes covariance loss given batch of projected features z1 from view 1 and
        projected features z2 from view 2.
        Args:
            z1 (torch.Tensor): NxD Tensor containing projected features from view 1.
            z2 (torch.Tensor): NxD Tensor containing projected features from view 2.
        Returns:
            torch.Tensor: covariance regularization loss.
        """

        N, D = z1.size()

        z1 = z1 - z1.mean(dim=0)
        z2 = z2 - z2.mean(dim=0)
        cov_z1 = (z1.T @ z1) / (N - 1)
        cov_z2 = (z2.T @ z2) / (N - 1)

        diag = torch.eye(D, device=z1.device)
        cov_loss = (
            cov_z1[~diag.bool()].pow_(2).sum() / D
            + cov_z2[~diag.bool()].pow_(2).sum() / D
        )
        return cov_loss, (z1.T @ z2) / (N - 1)

    def vicreg_loss_func(
        self: nn.Module,
        z1: torch.Tensor,
        z2: torch.Tensor,
        sim_loss_weight: float = 50.0,
        var_loss_weight: float = 50.0,
        cov_loss_weight: float = 1.0,
    ) -> torch.Tensor:
        """Computes VICReg's loss given batch of projected features z1 from view 1 and
        projected features z2 from view 2.
        Args:
            z1 (torch.Tensor): NxD Tensor containing projected features from view 1.
            z2 (torch.Tensor): NxD Tensor containing projected features from view 2.
            sim_loss_weight (float): invariance loss weight.
            var_loss_weight (float): variance loss weight.
            cov_loss_weight (float): covariance loss weight.
        Returns:
            torch.Tensor: VICReg loss.
        """

        sim_loss = self.invariance_loss(z1, z2)
        var_loss = self.variance_loss(z1, z2)
        cov_loss, c = self.covariance_loss(z1, z2)

        loss = (
            sim_loss_weight * sim_loss
            + var_loss_weight * var_loss
            + cov_loss_weight * cov_loss
        )
        return loss, c


class BTLoss(nn.Module):
    def __init__(self, num_classes=3):
        super(BTLoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, features, aug, labels=None, mask=None, print_c=False):
        lamdb = 5e-4
        c = F.normalize(features).T @ F.normalize(aug)
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal(c).add_(1).pow_(2).sum()  # HSIC

        if print_c:
            print(c)
            self.cross_cov = c
        loss = on_diag + lamdb * off_diag

        return loss

    def off_diagonal(self, x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class VICSupConLoss(VICLoss):
    def __init__(self, num_classes=3):
        super(VICSupConLoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, features, labels=None, print_c=False):
        device = features.device if features.is_cuda else torch.device('cpu')

        if len(features.shape) < 2:
            raise ValueError(
                '`features` needs to be [bsz, features],'
                'at least 2 dimensions are required'
            )
        if len(features.shape) > 2:
            features = features.view(features.shape[0], features.shape[1], -1)

        label_mask = torch.eq(
            torch.arange(self.num_classes).view(-1, 1).to(device), labels
        ).float()
        num_idx = label_mask.sum(1).int()
        label_idx = label_mask.nonzero()[:, 1].split(num_idx.tolist())

        loss = torch.zeros(1, device=device)

        y1_list = []
        y2_list = []

        for l in label_idx:
            z1 = features[l]
            z2_idx = torch.randperm(z1.size(0))
            z2 = z1[z2_idx]
            y1_list.append(z1)
            y2_list.append(z2)

        y1 = torch.cat(y1_list, dim=0)
        y2 = torch.cat(y2_list, dim=0)

        loss, c = self.vicreg_loss_func(y1, y2)
        if print_c:
            print(c)

        return loss
