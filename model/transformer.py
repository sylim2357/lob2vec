from torch.nn.modules.container import ModuleList
from torch.nn.init import xavier_uniform_
from typing import Optional, Any
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn import Module
import copy
import torch

cuda = 'cuda:0'

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
        d_model: int = 512,
        n_head: int = 8,
        # d_head: int = d_model / n_head,
        n_encoder_layers: int = 1,
        dim_feedforward: int = 2048,
        clamp_len: int = -1,
        dropout: float = 0.1,
        activation: str = 'gelu',
        tr_weight_share: bool = True,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.n_head = n_head
        self.clamp_len = clamp_len
        self.pos_emb_dropout = nn.Dropout(dropout)
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
        mems: Optional[list] = None,
        src_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            src: the sequence to the encoder (required). [src_len, bsz, dim]
            mems: the stack of memory from previous sequence (optional) [mem_len, bsz, dim] * num_layers
            src_mask: the additive mask for the src sequence (optional). [src_len, src_len]
        Returns:
            output: [tgt_len, bsz, dim]
        """

        bsz = src.size(1)
        if src.size(2) != self.d_model:
            print(src.size())
            print(self.d_model)
            raise RuntimeError(
                "the feature number of src and tgt must be equal to d_model"
            )

        ## positional encodings
        qlen = src.size(0)
        if mems is None:
            klen = qlen
        else:
            klen = qlen + mems.size(0)

        # for encoder
        enc_pos_emb = self._relative_positional_encoding(
            qlen,
            klen,
            self.d_model,
            self.clamp_len,
            'bi',
            bsz=bsz,
            dtype=torch.float32,
        ).to(torch.device(cuda))

        enc_pos_emb = self.pos_emb_dropout(enc_pos_emb)

        src_mask = _generate_square_subsequent_mask(qlen).to(
            torch.device(cuda)
        )
        encoding, new_mems = self.encoder(
            src,
            enc_pos_emb,
            mems=mems,
            mask=src_mask,
        )

        return encoding, new_mems

    def _reset_parameters(self):
        """ Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def _relative_positional_encoding(
        self, qlen, klen, d_model, clamp_len, attn_type, bsz=None, dtype=None
    ):
        """create relative positional encoding."""

        freq_seq = torch.arange(0, d_model, 2.0)
        if dtype is not None and dtype != torch.float32:
            freq_seq = freq_seq.type(dtype)
        inv_freq = 1 / (10000 ** (freq_seq / d_model))

        if attn_type == 'bi':
            # beg, end = klen - 1, -qlen
            beg, end = klen, -qlen
            bi_data = True
        elif attn_type == 'uni':
            # beg, end = klen - 1, -1
            beg, end = klen, -1
            bi_data = False
        else:
            raise ValueError('Unknown `attn_type` {}.'.format(attn_type))

        if bi_data and bsz % 2 == 0:
            fwd_pos_seq = torch.arange(beg, end, -1.0)
            bwd_pos_seq = torch.arange(-beg, -end, 1.0)

            if dtype is not None and dtype != torch.float32:
                fwd_pos_seq = fwd_pos_seq.type(dtype=dtype)
                bwd_pos_seq = bwd_pos_seq.type(dtype=dtype)

            if clamp_len > 0:
                fwd_pos_seq = torch.clamp(fwd_pos_seq, -clamp_len, clamp_len)
                bwd_pos_seq = torch.clamp(bwd_pos_seq, -clamp_len, clamp_len)

            fwd_pos_emb = self.positional_embedding(
                fwd_pos_seq, inv_freq, bsz // 2
            )
            bwd_pos_emb = self.positional_embedding(
                bwd_pos_seq, inv_freq, bsz // 2
            )

            pos_emb = torch.cat([fwd_pos_emb, bwd_pos_emb], dim=1)

        else:
            fwd_pos_seq = torch.arange(beg, end, -1.0)
            if dtype is not None and dtype != torch.float32:
                fwd_pos_seq = fwd_pos_seq.type(dtype=dtype)
            if clamp_len > 0:
                fwd_pos_seq = torch.clamp(fwd_pos_seq, -clamp_len, clamp_len)
            pos_emb = self.positional_embedding(fwd_pos_seq, inv_freq, bsz)

        return pos_emb

    def positional_embedding(self, pos_seq, inv_freq, bsz):
        sinusoid_inp = torch.einsum('i,d->id', pos_seq, inv_freq)
        pos_emb = torch.cat(
            [torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)], dim=-1
        )
        pos_emb = pos_emb[:, None, :]

        if bsz is not None:
            pos_emb = pos_emb.repeat(1, bsz, 1)
            # pos_emb = torch.tile(pos_emb, [1, bsz, 1])

        return pos_emb


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
        self.new_mems = []
        self.n_layers = n_layers
        self.norm = norm
        self.dropout = nn.Dropout()

    def forward(
        self,
        src: Tensor,
        pos_emb: Tensor,
        mems: Optional[list] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            src: the sequence to the encoder (required). [src_len, bsz, dim]
            pos_emb: positional embedding [klen, bsz, dim]
            mems: the stack of memory from previous sequence (optional) [mem_len, bsz, dim] * num_layers
            mask: the mask for the src sequence (optional). [src_len, src_len]
        """

        def fwd(fn, output, mems, pos_emb, mask):
            mem = mems[idx] if mems is not None else None
            self.new_mems.append(
                self._cache_mem(
                    output,
                    mem,
                    mem_len=mem.shape[0] if mem is not None else 0,
                    reuse_len=None,
                )
            )
            output = fn(
                output,
                pos_emb,
                mem,
                src_mask=mask,
            )

            return output

        output = src

        if self.tr_weight_share:
            for idx in range(self.n_layers):
                output = fwd(self.encoder_layer, output, mems, pos_emb, mask)
        else:
            for idx, mod in enumerate(self.layers):
                output = fwd(mod, output, mems, pos_emb, mask)

        # The layer-norm: To be moved
        if self.norm is not None:
            output = self.norm(output)

        return output, self.new_mems

    def _cache_mem(self, curr_out, prev_mem, mem_len, reuse_len=None):
        """cache hidden states into memory."""

        with torch.no_grad():
            if mem_len is None or mem_len == 0:
                return None
            else:
                if reuse_len is not None and reuse_len > 0:
                    curr_out = curr_out[:reuse_len]

                if prev_mem is None:
                    new_mem = curr_out[-mem_len:]
                else:
                    new_mem = torch.cat([prev_mem, curr_out], dim=0)[-mem_len:]

        return new_mem


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
        dim_feedforward=64,
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
        pos_emb: Optional[Tensor] = None,
        mem: Optional[Tensor] = None,
        src_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            src: the sequence to the encoder layer (required).
            pos_emb: positional embedding.
            mem: the memory cell.
            src_mask: the mask for the src sequence (optional).
        """
        if mem is not None and len(mem.size()) > 1:
            cat = torch.cat([mem, src], dim=0)
        else:
            cat = src

        pos_emb = self.dropout_pos(pos_emb)

        src2 = self.self_attn(src, cat, cat, pos_emb, attn_mask=src_mask,)[
            0
        ]  # [src_len, bsz, d_model]
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
        self.scale = 1 / (d_head ** 0.5)
        self.q_project_weight = nn.Parameter(
            torch.randn(d_model, n_head, d_head)
        )
        self.k_project_weight = nn.Parameter(
            torch.randn(d_model, n_head, d_head)
        )
        self.v_project_weight = nn.Parameter(
            torch.randn(d_model, n_head, d_head)
        )
        self.r_project_weight = nn.Parameter(
            torch.randn(d_model, n_head, d_head)
        )
        self.project_o = nn.Parameter(torch.randn(d_model, n_head, d_head))

        self.r_w_bias = nn.Parameter(torch.randn(n_head, d_head))
        self.r_r_bias = nn.Parameter(torch.randn(n_head, d_head))
        self.drop_attn = nn.Dropout(dropout)

    def forward(self, q, k, v, pos_emb, attn_mask):

        # content head
        q_head = torch.einsum('ibh,hnd->ibnd', q, self.q_project_weight)
        k_head = torch.einsum('ibh,hnd->ibnd', k, self.k_project_weight)
        v_head = torch.einsum('ibh,hnd->ibnd', v, self.v_project_weight)

        # position head
        k_head_pos = torch.einsum(
            'ibh,hnd->ibnd', pos_emb, self.r_project_weight
        )

        # content based attention score
        ac = torch.einsum('ibnd,jbnd->ijbn', q_head + self.r_w_bias, k_head)

        # position based attention score
        bd = torch.einsum(
            'ibnd,jbnd->ijbn', q_head + self.r_r_bias, k_head_pos
        )
        bd = self._rel_shift(bd, klen=ac.shape[1])

        attn_score = (ac + bd) * self.scale

        attn_mask = (
            0 if attn_mask is None else attn_mask.unsqueeze(-1).unsqueeze(-1)
        )
        attn_score = attn_score - 1e30 * attn_mask

        # attention probability
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.drop_attn(attn_prob)

        # attention output
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', attn_prob, v_head)

        # post-attention projection
        attn_out = torch.einsum('ibnd,hnd->ibh', attn_vec, self.project_o)

        return (
            attn_out,
            attn_vec.sum(dim=2) / self.n_head,
        )  # [src_len, bsz, d_model]

    def _rel_shift(self, x, klen=-1):
        """perform relative shift to form the relative attention score."""

        x_size = x.shape

        x = torch.reshape(x, [x_size[1], x_size[0], x_size[2], x_size[3]])
        x = x[1:, 0:, 0:, 0:]
        x = torch.reshape(x, [x_size[0], x_size[1] - 1, x_size[2], x_size[3]])
        x = x[0:, 0:klen, 0:, 0:]

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
