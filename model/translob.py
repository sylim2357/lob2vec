from torch.nn.modules.container import ModuleList
from torch.nn.init import xavier_uniform_
from typing import Optional
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn import Module
import numpy as np
import copy
import torch


class TransLobEncoder(nn.Module):
    def __init__(
        self, extractor=None, pos_encoding=None, aggregator=None, norm=False
    ):
        super().__init__()
        if extractor:
            self.extractor = extractor
        else:
            self.extractor = FeatureExtractor(
                [
<<<<<<< HEAD
                    (127, 4, 1, 2),
                    (127, 4, 1, 2),
                    (127, 4, 1, 2),
                    (127, 4, 1, 2),
                    (127, 4, 1, 2),
=======
                    (63, 4, 1, 1),
                    (63, 4, 1, 2),
                    (63, 4, 1, 4),
                    (63, 4, 1, 8),
                    (63, 4, 1, 16),
>>>>>>> 7eca5ce148d9f3a24a371d22cf2044f5100ec305
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
<<<<<<< HEAD
                d_model=128,
                n_head=8,
                n_encoder_layers=8,
                dim_feedforward=512,
                dropout=0.3,
                activation='gelu',
=======
                d_model=256,
                n_head=4,
                n_encoder_layers=2,
                dim_feedforward=1024,
                dropout=0.1,
                activation='relu',
>>>>>>> 7eca5ce148d9f3a24a371d22cf2044f5100ec305
                tr_weight_share=True,
            )
        self.norm = norm

        self.fc = nn.Linear(128, 128)

        self.layernorm = nn.LayerNorm([127, 100])
        self.fc1 = nn.Linear(100 * 128, 512)

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
        if self.norm:
            x = F.normalize(x, dim=1)

        return x


class TransformerAggregator(Module):
    """
    Args:
        d_model: the number of expected features in the encoder/decoder inputs (default=64).
        n_head: the number of heads in the multiheadattention models (default=8).
        n_encoder_layers: the number of sub-encoder-layers in the encoder (default=6).
        n_decoder_layers: the number of sub-decoder-layers in the decoder (default=6).
        dim_feedforward: the dimension of the feedforward network model (default=64).
        dropout: the dropout value (default=0.1).
        activation: the activation function of encoder/decoder intermediate layer, relu or gelu (default=gelu).
    """

    def __init__(
        self,
        d_model: int = 8,
        n_head: int = 8,
        n_encoder_layers: int = 2,
        dim_feedforward: int = 60,
        clamp_len: int = -1,
        dropout: float = 0.1,
        activation: str = 'gelu',
        tr_weight_share: bool = True,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.n_head = n_head
        d_head = int(d_model / n_head)

        # encoder block
        encoder_layer = TransformerEncoderLayer(
            d_model, n_head, d_head, dim_feedforward, dropout, activation
        )
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = TransformerEncoder(
            encoder_layer, n_encoder_layers, tr_weight_share, encoder_norm
        )

        self._reset_parameters()

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            src: input sequence [bsz, src_len, dim]
            src_mask: the additive mask for the src sequence (optional). [src_len, src_len]
        Returns:
            output: [bsz, tgt_len, dim]
        """

        bsz = src.size(0)
        if src.size(2) != self.d_model:
            raise RuntimeError(
                "the feature number of src and tgt must be equal to d_model"
            )

        ## positional encodings
        qlen = src.size(1)
        # src_mask = _generate_square_subsequent_mask(qlen).to(
        #     torch.device(src.device)
        # )
        # encoding = self.encoder(src, mask=src_mask)
        encoding = self.encoder(src)

        return encoding

    def _reset_parameters(self):
        """ Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)


class TransformerEncoder(Module):
    """
    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        n_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).
    Returns:
        output: [src_len, bsz, dim]
    """

    def __init__(self, encoder_layer, n_layers, tr_weight_share, norm=None):
        super(TransformerEncoder, self).__init__()
        self.n_layers = n_layers
        self.tr_weight_share = tr_weight_share
        if tr_weight_share:
            self.encoder_layer = encoder_layer
        else:
            self.layers = _get_clones(encoder_layer, n_layers)
        self.n_layers = n_layers
        self.norm = norm
        self.dropout = nn.Dropout()

    def forward(
        self,
        src: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            src: the sequence to the encoder (required). [src_len, bsz, dim]
            mask: the mask for the src sequence (optional). [src_len, src_len]
        """

        output = src

        if self.tr_weight_share:
            for idx in range(self.n_layers):
                output = self.encoder_layer(output, mask)
        else:
            for idx, mod in enumerate(self.layers):
                output = mod(output, mask)

        # The layer-norm: To be moved
        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerEncoderLayer(Module):
    """
    Args:
        d_model: the number of expected features in the input (required).
        n_head: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=64).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=gelu).
    """

    def __init__(
        self,
        d_model,
        n_head,
        d_head,
        dim_feedforward=60,
        dropout=0.1,
        activation='gelu',
    ):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = RelativeMultiheadAttention(
            d_model, n_head, d_head, dropout=dropout
        )

        self.dropout_pos = nn.Dropout(dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            src: the sequence to the encoder layer (required).
            pos_emb: positional embedding.
            mem: the memory cell.
            src_mask: the mask for the src sequence (optional).
        """

        src2, _ = self.self_attn(
            src,
            src,
            src,
            attn_mask=src_mask,
        )  # [bsz, src_len, d_model]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src


class RelativeMultiheadAttention(Module):
    """
    Reference: arxiv.org/abs/1901.02860
    """

    def __init__(self, d_model, n_head, d_head, dropout):
        super(RelativeMultiheadAttention, self).__init__()

        self.n_head = n_head
        self.d_head = d_head
        self.scale = 1 / (d_head ** 0.5)

        self.q_project_weight = nn.Parameter(
            torch.randn(d_model, n_head, d_head, dtype=torch.float32)
        )
        nn.init.kaiming_normal_(self.q_project_weight)
        self.k_project_weight = nn.Parameter(
            torch.randn(d_model, n_head, d_head)
        )
        nn.init.kaiming_normal_(self.k_project_weight)
        self.v_project_weight = nn.Parameter(
            torch.randn(d_model, n_head, d_head)
        )
        nn.init.kaiming_normal_(self.v_project_weight)

        self.project_o = nn.Parameter(torch.randn(d_model, n_head, d_head))

        self.drop_attn = nn.Dropout(dropout)

        self.max_position_embeddings = 100
        self.distance_embedding = nn.Embedding(
            2 * self.max_position_embeddings - 1, d_head
        )

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.n_heads, self.d_head)
        x = x.view(*new_x_shape)

        return x.permute(0, 2, 1, 3)

    def forward(self, q, k, v, attn_mask):
        seq_len = q.size(1)

        q_head = torch.einsum('bih,hnd->bind', q, self.q_project_weight)
        k_head = torch.einsum('bih,hnd->bind', q, self.k_project_weight)
        v_head = torch.einsum('bih,hnd->bind', q, self.v_project_weight)

        position_ids_l = torch.arange(
            seq_len, dtype=torch.long, device=q_head.device
        ).view(-1, 1)
        position_ids_r = torch.arange(
            seq_len, dtype=torch.long, device=q_head.device
        ).view(1, -1)
        distance = position_ids_l - position_ids_r
        positional_embedding = self.distance_embedding(
            distance + self.max_position_embeddings - 1
        )
        positional_embedding = positional_embedding.to(dtype=q_head.dtype)

        attn_scores = torch.einsum('bind, bjnd-> bnij', q_head, k_head)

        relative_position_scores_query = torch.einsum(
            'blnd,lrd->bnlr', q_head, positional_embedding
        )
        relative_position_scores_key = torch.einsum(
            'brnd,lrd->bnlr', k_head, positional_embedding
        )

        attn_scores += (
            relative_position_scores_query + relative_position_scores_key
        )
        attn_scores /= self.scale

        if attn_mask is not None:
            attn_scores - 1e30 * attn_mask

        attn_probs = nn.Softmax(dim=-1)(attn_scores)
        attn_probs = self.drop_attn(attn_probs)

        attn_vec = torch.einsum('bnij,bjnd->bind', attn_probs, v_head)

        attn_out = torch.einsum('bind,hnd->bih', attn_vec, self.project_o)

        return (
            attn_out,
            attn_vec.sum(dim=2) / self.n_head,
        )  # [bsz, seq_len, d_model]


class FeatureExtractor(nn.Module):
    def __init__(
        self,
        conv_layers,
        input_dim,
        dropout,
        activation,
    ):
        super().__init__()

        def block(n_in, n_out, k, stride, dilation):

            pad = ZeroPad1d((k - 1) * dilation, 0)

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
                activation,
            )

        in_d = input_dim
        self.conv_layers = nn.ModuleList()
        for dim, k, stride, dilation in conv_layers:
            self.conv_layers.append(block(in_d, dim, k, stride, dilation))
            in_d = dim

    def forward(self, x):
        for conv in self.conv_layers:
            x = conv(x)

        return x


class ZeroPad1d(nn.Module):
    def __init__(self, pad_left, pad_right):
        super().__init__()
        self.pad_left = pad_left
        self.pad_right = pad_right

    def forward(self, x):

        return F.pad(x, (self.pad_left, self.pad_right))


class PositionalEncodingLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        d_model, steps = x.shape[-2:]
        ps = np.zeros([1, steps], dtype=np.float32)
        for tx in range(steps):
            ps[:, tx] = [(2 / (steps - 1)) * tx - 1]

        ps_expand = torch.from_numpy(ps).unsqueeze(0).to(x.device)
        ps_tiled = ps_expand.repeat([x.shape[0], 1, 1])

        x = torch.cat([x, ps_tiled], axis=1)

        return x


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == 'relu':
        return F.relu
    elif activation == 'gelu':
        return F.gelu


def _generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float(1.0))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask
