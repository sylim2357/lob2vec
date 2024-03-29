import torch.nn.functional as F
import torch.nn as nn
import torch


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(
        self, temperature=0.007, contrast_mode='all', base_temperature=0.007
    ):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None, **kwargs):
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


class SupConMixUpLoss(nn.Module):
    """Random mixup within batch."""

    def __init__(
        self, temperature=0.07, contrast_mode='all', base_temperature=0.07
    ):
        super(SupConMixUpLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def make_batches(self, features):
        lam = torch.rand(1, device=features.device) * 0.01 + 0.99
        features_perm = features[torch.randperm(len(features))]
        batch1 = lam * features + (1 - lam) * features_perm

        lam = torch.rand(1, device=features.device) * 0.01 + 0.99
        features_perm = features[torch.randperm(len(features))]
        batch2 = lam * features + (1 - lam) * features_perm
        return torch.stack((batch1, batch2), dim=1)

    def forward(self, features, labels=None, mask=None, **kwargs):
        device = (
            torch.device('cuda') if features.is_cuda else torch.device('cpu')
        )
        mixup_features = self.make_batches(features.squeeze())

        if len(mixup_features.shape) < 3:
            raise ValueError(
                '`features` needs to be [bsz, n_views, ...],'
                'at least 3 dimensions are required'
            )
        if len(mixup_features.shape) > 3:
            mixup_features = mixup_features.view(
                mixup_features.shape[0], mixup_features.shape[1], -1
            )

        batch_size = mixup_features.shape[0]
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

        contrast_count = mixup_features.shape[1]
        contrast_feature = torch.cat(
            torch.unbind(mixup_features, dim=1), dim=0
        )
        if self.contrast_mode == 'one':
            anchor_feature = mixup_features[:, 0]
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


class VICandSupConLoss(nn.Module):
    def __init__(
        self, temperature=0.07, contrast_mode='all', base_temperature=0.07
    ):
        super(VICandSupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.vic = VICLoss()

    def forward(self, features, labels=None, mask=None, print_c=False):
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
        label_idxs = label_mask.nonzero()[:, 1].split(num_idx.tolist())
        label_idx = torch.cat(label_idxs, dim=0)
        labels = labels[label_idx].view(-1, 1)
        contrast_feature = features[label_idx].reshape(
            (batch_size * n_views, -1)
        )

        y_list = []

        for l in label_idxs:
            z1 = features[l]
            z1 = z1.reshape((len(l) * n_views, -1))
            z2_idx = torch.randperm(z1.size(0))
            z2 = z1[z2_idx]
            y_list.append(z2)

        contrast_feature_2 = torch.cat(y_list, dim=0)
        lam = (
            torch.distributions.beta.Beta(1, 1)
            .sample((contrast_feature_2.size(0), 1))
            .to(device)
        )
        z1_interpolate = (
            lam * contrast_feature + (1 - lam) * contrast_feature_2
        )
        z2_interpolate = (
            lam * contrast_feature_2 + (1 - lam) * contrast_feature
        )

        loss1 = self.vic(z1_interpolate, z2_interpolate, print_c=print_c)
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

        return 25 * loss + loss1


class VICandSupConMixupLoss(nn.Module):
    """mixup within same label -> vic & supcon"""

    def __init__(
        self, temperature=0.01, contrast_mode='all', base_temperature=0.01
    ):
        super(VICandSupConMixupLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.vic = VICLoss()

    def forward(self, features, labels=None, mask=None, print_c=False):
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
        label_idxs = label_mask.nonzero()[:, 1].split(num_idx.tolist())
        label_idx = torch.cat([l.repeat(n_views).reshape(len(l)*n_views) for l in label_idxs], dim=0)
        labels = labels[label_idx].view(-1, 1)

        z_list = []

        for l in label_idxs:
            if len(l) > 0:
                z1 = features[l]
                z1 = z1.reshape((len(l) * n_views, -1))
                z2_idx = torch.randperm(len(l))
                z2 = z1[z2_idx]
                lam1 = torch.rand(1).to(device) * 0.01 + 0.99
                z_mixup1 = lam1 * z1 + (1 - lam1) * z2
                z_mixup2 = lam1 * z2 + (1 - lam1) * z1

                z3_idx = torch.randperm(z2.size(0))
                z3 = z1[z3_idx]
                z4_idx = torch.randperm(z2.size(0))
                z4 = z1[z4_idx]
                lam2 = torch.rand(1).to(device) * 0.01 + 0.99
                z_mixup3 = lam2 * z3 + (1 - lam2) * z4
                z_mixup4 = lam2 * z4 + (1 - lam2) * z3

                z_list.append(torch.stack((z_mixup1, z_mixup2, z_mixup3, z_mixup4), dim=1))
                # z_list.append(torch.stack((z_mixup1, z_mixup2), dim=1))
        z = torch.cat(z_list, dim=0)
        loss1 = self.vic(z[:,0,::].squeeze(), z[:,1,::].squeeze(),print_c=print_c)
        mask = torch.eq(labels, labels.T).float().to(device)  # be careful
        contrast_count = z.shape[1]
        contrast_feature = torch.cat(torch.unbind(z, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = z[:, 0]
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

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        # logits = anchor_dot_contrast
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
        log_prob = logits - 10*torch.log(exp_logits.sum(1, keepdim=True))
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        # loss
<<<<<<< HEAD
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
=======
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        # loss = loss.view(anchor_count, batch_size).mean()
>>>>>>> 7eca5ce148d9f3a24a371d22cf2044f5100ec305
        loss = loss.mean()
        # if print_c:
        #     print(f'supcon loss is {loss}')
        return loss + loss1*0.2
        # return loss


class VICLoss(nn.Module):
    def __init__(self, num_classes=3):
        super(VICLoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, features, aug, labels=None, mask=None, print_c=False):
        loss, c, _, _, _ = self.vicreg_loss_func(features, aug)

        if print_c:
            print('cov')
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

        eps = 1e-5
        std_z1 = torch.sqrt(z1.var(dim=0) + eps)
        std_z2 = torch.sqrt(z2.var(dim=0) + eps)
        std_loss = torch.mean(F.relu(1 - std_z1)) + torch.mean(F.relu(1 - std_z2))
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
        # print(z1.shape)
        N, D = z1.size()
        #
        # z1 = z1 - z1.mean(dim=0)
        # z2 = z2 - z2.mean(dim=0)
        # cov_z1 = (z1.T @ z1) / (N - 1)
        # cov_z2 = (z2.T @ z2) / (N - 1)
        #
        # diag = torch.eye(D, device=z1.device)
        # cov_loss = (
        #     cov_z1[~diag.bool()].pow_(2).sum() / D
        #     + cov_z2[~diag.bool()].pow_(2).sum() / D
        # )
        # return cov_loss, (z1.T @ z2) / (N - 1)
        norm_z1 = z1 - z1.mean(dim=0)
        norm_z2 = z2 - z2.mean(dim=0)
        norm_z1 = F.normalize(norm_z1, p=2, dim=0)  # (batch * feature); l2-norm
        norm_z2 = F.normalize(norm_z2, p=2, dim=0)
        fxf_cov_z1 = torch.mm(norm_z1.T, norm_z1)  # (feature * feature)
        fxf_cov_z2 = torch.mm(norm_z2.T, norm_z2)
        fxf_cov_z1.fill_diagonal_(0.0)
        fxf_cov_z2.fill_diagonal_(0.0)
        cov_loss = (fxf_cov_z1 ** 2).mean() + (fxf_cov_z2 ** 2).mean()
        return cov_loss, (norm_z1.T @ norm_z1)

    def vicreg_loss_func(
        self: nn.Module,
        z1: torch.Tensor,
        z2: torch.Tensor,
<<<<<<< HEAD
        sim_loss_weight: float = 1.0,
        var_loss_weight: float = 2500.0,
        cov_loss_weight: float = 2500.0,
=======
        sim_loss_weight: float = 25.0,
        var_loss_weight: float = 10.0,
        cov_loss_weight: float = 10.0,
>>>>>>> 7eca5ce148d9f3a24a371d22cf2044f5100ec305
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
        # loss = var_loss_weight * var_loss + cov_loss_weight * cov_loss
        return loss, c, sim_loss_weight*sim_loss, var_loss_weight*var_loss, cov_loss_weight*cov_loss


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


class VICSupLoss(VICLoss):
    def __init__(self, num_classes=3):
        super(VICSupLoss, self).__init__()
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

        batch_size = features.shape[0]
        n_views = features.shape[1]

        label_mask = torch.eq(
            torch.arange(3).view(-1, 1).to(device), labels
        ).float()
        # print('label_mask')
        # print(label_mask)
        num_idx = label_mask.sum(1).int()
        # print('num_idx')
        # print(num_idx)
        label_idxs = label_mask.nonzero()[:, 1].split(num_idx.tolist())
        # print('label_idxs')
        # print(label_idxs)
        label_idx = torch.cat(label_idxs, dim=0)
        # print('label_idx')
        # print(label_idx)
        labels = labels[label_idx].view(-1, 1)
        # print('labels')
        # print(labels)
        contrast_feature = features[label_idx].reshape(
            (batch_size * n_views, -1)
        )
        # print('contrast_feature')
        # print(contrast_feature)

        # y_list = []
        z1_list = []
        z2_list = []

        for l in label_idxs:
            if len(l) > 0:
                z1 = features[l]
                z1 = z1.reshape((len(l) * n_views, -1))
                z2_idx = torch.randperm(z1.size(0))
                z2 = z1[z2_idx]
<<<<<<< HEAD
                y_list.append(z2)

        contrast_feature_2 = torch.cat(y_list, dim=0)
        loss, c, s, v, cov = self.vicreg_loss_func(contrast_feature, contrast_feature_2)
=======
                lam = torch.rand(1).to(device) * 0.2 + 0.8
                z_mixup1 = lam * z1 + (1 - lam) * z2

                lam = torch.rand(1).to(device) * 0.2 + 0.8
                z3_idx = torch.randperm(z2.size(0))
                z3 = z2[z3_idx]
                z_mixup2 = lam * z1 + (1 - lam) * z3
                z1_list.append(z_mixup1)
                z2_list.append(z_mixup2)

            # z1_list.append(z_mixup1)
            # z2_list.append(z_mixup2)
            # z1 = features[l]
            # z1 = z1.reshape((len(l) * n_views, -1))
            # z2_idx = torch.randperm(z1.size(0))
            # z2 = z1[z2_idx]
            # y_list.append(z2)

        # contrast_feature_2 = torch.cat(y_list, dim=0)
        z1 = torch.cat(z1_list, dim=0)
        z2 = torch.cat(z2_list, dim=0)
        # lam = (
        #     torch.distributions.beta.Beta(1, 1)
        #     .sample((contrast_feature_2.size(0), 1))
        #     .to(device)
        # )
        # z1_interpolate = (
        #     lam * contrast_feature + (1 - lam) * contrast_feature_2
        # )
        # z2_interpolate = (
        #     lam * contrast_feature_2 + (1 - lam) * contrast_feature
        # )
        loss, c = self.vicreg_loss_func(z1, z2)
>>>>>>> 7eca5ce148d9f3a24a371d22cf2044f5100ec305

        if print_c:
            print('cov matrix: ')
            print(c)
            print(s)
            print(v)
            print(cov)

        return loss
