# -*- coding: utf-8 -*-
from model.transformer import TransformerAggregator
from utils.model_utils import *
from torch.nn import Module
import torch.nn as nn
import torch
import math


class Lob2Vec(Module):
    """
    Lob2Vec module analogous to Wav2vec
    """

    def __init__(self):
        super().__init__()

        kernels = [4,4,4,4,4,4,2,2,2,2,2,2]
        dilations = [2,1,2,1,2,1,2,1,2,1,2,1]
        conv_aggregator_layers = [(64, 2, 1), (64, 3, 1), (64, 4, 1), (64, 5, 1), \
    (64, 6, 1), (64, 7, 1), (64, 8, 1), (64, 9, 1), \
    (64, 10, 1)]
        num_cnn_layers = 12
        d_model = 64
        input_dim = 40
        num_negatives = 10
        dropout = 0.5
        log_compression = True
        skip_connections_feat = True
        residual_scale = 0.5
        non_affine_group_norm = False
        agg_zero_pad =True
        is_inst_norm = False

        ###
        num_enc_layers = 5
        tr_weight_share = True
        skip_connections_agg = True

        ###
        sample_distance = None
        balanced_classes = False
        multi_label_cpc = False

        ###
        dropout_features = 0.5
        dropout_agg = 0.5

        assert (
            len(kernels) == len(dilations) == num_cnn_layers
        ), 'length of kernels should match length of dilations and number of cnn layers'

        prediction_steps = 20
        offset = -1

        # if FLAGS.activation == 'relu':
        #     activation = nn.ReLU()
        # elif FLAGS.activation == 'gelu':
        #     activation = nn.GELU()
        # else:
        #     raise Exception('unknown activation ')

        activation = nn.GELU()

        # feature extraction with CNN: X -> Z
        feature_enc_layers = []
        for i in range(num_cnn_layers):
            feature_enc_layers.append(
                (d_model, kernels[i], 1, dilations[i])
            )

        self.feature_extractor = ConvFeatureExtractionModel(
            conv_layers=feature_enc_layers,
            input_dim=input_dim,
            dropout=dropout,
            log_compression=log_compression,
            skip_connections=skip_connections_feat,
            residual_scale=residual_scale,
            non_affine_group_norm=non_affine_group_norm,
            zero_pad=agg_zero_pad,
            is_inst_norm=is_inst_norm,
            activation=activation,
        )
        embed = feature_enc_layers[-1][0]

        # determine offset
        if offset == -1:
            kin = 0
            sin = 0
            din = 0
            for _, k, stride, d in feature_enc_layers:
                if din == 0:
                    din = d
                if kin == 0:
                    kin = k
                kin = kin + (k - 1) * din
                # rin = rin + (k - 1) * sin
                if sin == 0:
                    sin = stride
                else:
                    sin *= stride

            offset = math.ceil(kin / sin)

        offset = int(offset)

        # context aggregator with CNN: Z -> C
        # aggregator = 'cnn'
        aggregator = 'transformer'
        if aggregator == 'cnn':
            agg_layers = conv_aggregator_layers
            agg_dim = embed
            self.feature_aggregator = ConvAggregator(
                conv_layers=agg_layers,
                embed=embed,
                dropout=dropout,
                skip_connections=skip_connections_agg,
                residual_scale=residual_scale,
                non_affine_group_norm=non_affine_group_norm,
                conv_bias=True,
                zero_pad=agg_zero_pad,
                activation=activation,
            )
        else:
            agg_dim = feature_enc_layers[-1][0]
            self.feature_aggregator = TransformerAggregator(
                d_model=agg_dim,
                n_encoder_layers=num_enc_layers,
                dim_feedforward=agg_dim * 4,
                dropout=dropout,
                tr_weight_share=tr_weight_share,
            )

        # predictions in the Z space to compute losses
        self.lob2vec_predictions = Lob2VecPredictionsModel(
            in_dim=agg_dim,
            out_dim=embed,
            prediction_steps=prediction_steps,
            n_negatives=num_negatives,
            sample_distance=sample_distance,
            dropout=dropout,
            offset=offset,
            balanced_classes=balanced_classes,
            multi_label_cpc=multi_label_cpc,
        )

        self.dropout_feats = nn.Dropout(p=dropout_features)
        self.dropout_agg = nn.Dropout(p=dropout_agg)

    def forward(self, source):
        result = {}

        features = self.feature_extractor(source)

        x = self.dropout_feats(features)
        x = x.permute([2, 0, 1])
        x, _ = self.feature_aggregator(x)
        x = x.permute([1, 2, 0])

        x = self.dropout_agg(x)

        x = self.lob2vec_predictions(x, features)

        return x


class ConvFeatureExtractionModel(Module):
    def __init__(
        self,
        conv_layers,
        input_dim,
        dropout,
        log_compression,
        skip_connections,
        residual_scale,
        non_affine_group_norm,
        zero_pad,
        is_inst_norm,
        activation,
    ):
        super().__init__()

        def block(n_in, n_out, k, stride, dilation):

            pad = (
                ZeroPad1d((k - 1) * dilation, 0)
                if zero_pad
                else nn.ReplicationPad1d(((k - 1) * dilation, 0))
            )

            return nn.Sequential(
                pad,
                nn.Conv1d(
                    n_in,
                    n_out,
                    k,
                    stride=stride,
                    dilation=dilation,
                    bias=False,
                ),
                nn.Dropout(p=dropout),
                norm_block(
                    is_inst_norm=is_inst_norm,
                    dim=n_out,
                    affine=not non_affine_group_norm,
                ),
                activation,
            )

        in_d = input_dim
        self.conv_layers = nn.ModuleList()
        for dim, k, stride, dilation in conv_layers:
            self.conv_layers.append(block(in_d, dim, k, stride, dilation))
            in_d = dim

        self.log_compression = log_compression
        self.skip_connections = skip_connections
        self.residual_scale = math.sqrt(residual_scale)

    def forward(self, x):
        # x: BxCxT

        for conv in self.conv_layers:
            residual = x
            x = conv(x)
            if self.skip_connections and x.size(1) == residual.size(1):
                tsz = x.size(2)
                r_tsz = residual.size(2)
                residual = residual[..., :: r_tsz // tsz][..., :tsz]
                x = (x + residual) * self.residual_scale

        if self.log_compression:
            x = x.abs()
            x = x + 1
            x = x.log()

        return x


class ConvAggregator(nn.Module):
    def __init__(
        self,
        conv_layers,
        embed,
        dropout,
        skip_connections,
        residual_scale,
        non_affine_group_norm,
        conv_bias,
        zero_pad,
        activation,
    ):
        super().__init__()

        def block(n_in, n_out, k, stride):
            # padding dims only really make sense for stride = 1
            ka = k // 2
            kb = ka - 1 if k % 2 == 0 else ka

            pad = (
                ZeroPad1d(ka + kb, 0)
                if zero_pad
                else nn.ReplicationPad1d((ka + kb, 0))
            )

            return nn.Sequential(
                pad,
                nn.Conv1d(n_in, n_out, k, stride=stride, bias=conv_bias),
                nn.Dropout(p=dropout),
                norm_block(False, n_out, affine=not non_affine_group_norm),
                activation,
            )

        in_d = embed
        self.conv_layers = nn.ModuleList()
        self.residual_proj = nn.ModuleList()
        for dim, k, stride in conv_layers:
            if in_d != dim and skip_connections:
                self.residual_proj.append(nn.Conv1d(in_d, dim, 1, bias=False))
            else:
                self.residual_proj.append(None)

            self.conv_layers.append(block(in_d, dim, k, stride))
            in_d = dim
        self.conv_layers = nn.Sequential(*self.conv_layers)
        self.skip_connections = skip_connections
        self.residual_scale = math.sqrt(residual_scale)

    def forward(self, x):
        for rproj, conv in zip(self.residual_proj, self.conv_layers):
            residual = x
            x = conv(x)
            if self.skip_connections:
                if rproj is not None:
                    residual = rproj(residual)
                x = (x + residual) * self.residual_scale
        return x


class Lob2VecPredictionsModel(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        prediction_steps,
        n_negatives,
        sample_distance,
        dropout,
        offset,
        balanced_classes,
        multi_label_cpc,
    ):
        super().__init__()

        self.n_negatives = n_negatives
        self.sample_distance = sample_distance
        self.project_to_steps = nn.ConvTranspose2d(
            in_dim, out_dim, (1, prediction_steps)
        )
        self.dropout = nn.Dropout(p=dropout)
        self.offset = offset
        self.balanced_classes = balanced_classes
        self.multi_label_cpc = multi_label_cpc

    def sample_negatives(self, y):
        bsz, fsz, tsz = y.shape

        y = y.transpose(0, 1)  # BxCxT -> CxBxT
        y = y.contiguous().view(fsz, -1)  # CxBxT => Cx(B*T)

        high = (
            tsz
            if self.sample_distance is None
            else min(tsz, self.sample_distance)
        )
        assert high > 1

        neg_idxs = torch.randint(
            low=0, high=high, size=(bsz, self.n_negatives * tsz)
        )

        with torch.no_grad():
            if self.n_negatives > 0:
                tszs = (
                    buffered_arange(tsz)  # arange
                    .unsqueeze(-1)  # tsz x 1
                    .expand(-1, self.n_negatives)  # tsz x n_negatives
                    .flatten()
                )

                neg_idxs = torch.randint(
                    low=0, high=high - 1, size=(bsz, self.n_negatives * tsz)
                )
                neg_idxs[neg_idxs >= tszs] += 1  # to remove true samples

        for i in range(1, bsz):
            neg_idxs[i] += i * high

        negs = y[..., neg_idxs.view(-1)]
        negs = negs.view(fsz, bsz, self.n_negatives, tsz).permute(
            2, 1, 0, 3
        )  # to NxBxCxT

        return negs

    def forward(self, x, y):

        x = x.unsqueeze(-1)  # BxCxTx1
        x = self.project_to_steps(x)  # BxCxTxS, "h(c)"
        x = self.dropout(x)

        negatives = self.sample_negatives(y)  # NxBxCxT
        y = y.unsqueeze(0)  # 1xBxCxT
        targets = torch.cat([y, negatives], dim=0)  # Copies x B x C x T, "z"

        copies = targets.size(0)
        bsz, dim, tsz, steps = x.shape
        steps = min(steps, tsz - self.offset)
        if self.multi_label_cpc:
            predictions = x.new(
                bsz,
                copies * (tsz - self.offset + 1) * steps
                - ((steps + 1) * steps // 2) * copies,
            )
        else:
            predictions = x.new(
                bsz * copies * (tsz - self.offset + 1) * steps
                - ((steps + 1) * steps // 2) * copies * bsz
            )

        start = end = 0
        for i in range(steps):
            offset = i + self.offset
            if self.multi_label_cpc:
                end = start + (tsz - offset) * copies
                predictions[:, start:end] = (
                    torch.einsum(
                        'bct,nbct->tbn',
                        x[..., :-offset, i],
                        targets[..., offset:],
                    )
                    .permute([1, 2, 0])
                    .reshape(bsz, -1)
                )
            else:
                end = start + (tsz - offset) * bsz * copies
                predictions[start:end] = torch.einsum(
                    'bct,nbct->tbn', x[..., :-offset, i], targets[..., offset:]
                ).flatten()
            start = end

        p = predictions[0] if self.multi_label_cpc else predictions
        assert end == p.numel(), "{} != {}".format(end, p.numel())
        if self.multi_label_cpc:
            predictions = predictions.view(bsz, copies, -1)
        else:
            predictions = predictions.view(copies, -1)

        return predictions


def norm_block(is_inst_norm, dim, affine=True):
    if is_inst_norm:
        # mod = nn.Sequential(
        #     TransposeLast(),
        #     Fp32LayerNorm(dim, elementwise_affine=affine),
        #     TransposeLast(),
        # )
        mod = Fp32InstanceNorm(dim, affine=affine)
    else:
        mod = nn.Sequential(
            TransposeLast(),
            Fp32LayerNorm(dim, elementwise_affine=affine),
            TransposeLast(),
        )
        # mod = Fp32GroupNorm(1, dim, affine=affine)

    return mod


def buffered_arange(max):
    if not hasattr(buffered_arange, 'buf'):
        buffered_arange.buf = torch.LongTensor()
    if max > buffered_arange.buf.numel():
        buffered_arange.buf.resize_(max)
        torch.arange(max, out=buffered_arange.buf)

    return buffered_arange.buf[:max]
