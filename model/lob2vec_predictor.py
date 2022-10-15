from model.lob2vec_wav2vec import Lob2Vec
from torch.nn import Module
from torch import nn
import torch


class Lob2VecPredictor(Module):
    """
    Lob2VecPredictor class contains pretrained lob2vec model and a linear
    predictor. If _endtoend, the entire model gets trained. Otherwise, only the
    predictor gets trained.
    """

    def __init__(self):
        super().__init__()

        self._endtoend = True
        self.d_model = 64
        self.seq_len = 100

        # self._endtoend = FLAGS.endtoend_train
        # self.d_model = FLAGS.d_model
        # self.seq_len = FLAGS.ds_seq_len
        # self.pt_weight_path = FLAGS.pt_weight_path
        # self.lob2vec = Lob2Vec(FLAGS)
        self.lob2vec = Lob2Vec()
        self.predictor = self._build_predictor()
        self.dropout = nn.Dropout(p=0.1)
        # self.linear = nn.Linear(11392, 3)

    @property
    def endtoend(self):
        return self._endtoend

    @endtoend.setter
    def endtoend(self, value):
        self._endtoend = value

    def forward(self, x):
        if self._endtoend:
            x = self.lob2vec.feature_extractor(x)
            x = self.lob2vec.dropout_feats(x)
            self.z = x.permute([2, 0, 1])
            x, _ = self.lob2vec.feature_aggregator(self.z)
            self.c = x.permute([1, 2, 0])
            # self.c = self.lob2vec.feature_aggregator(x)
        else:
            with torch.no_grad():
                x = self.lob2vec.feature_extractor(x)
                x = self.lob2vec.dropout_feats(x)
                self.z = x.permute([2, 0, 1])
                x, _ = self.lob2vec.feature_aggregator(self.z)
                self.c = x.permute([1, 2, 0])
        fwd = self.dropout(self.c)
        fwd = self.predictor(fwd.flatten(start_dim=1))
        # fwd = self.dropout(fwd)
        # fwd = self.linear(fwd)
        return fwd

    def train(self, mode=True):
        super().train()
        if self._endtoend:
            self.lob2vec.train(mode=mode)

    def _build_predictor(self):
        return nn.Sequential(nn.Linear(self.d_model * self.seq_len, 3))
