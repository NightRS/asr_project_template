from torch import nn
import torch.nn.functional as F
import torch

from hw_asr.base import BaseModel


class LinGRUModel(BaseModel):
    def __init__(self, n_feats, n_class, fc_hidden=512, **batch):
        super().__init__(n_feats, n_class, **batch)
        self.fc1 = nn.Linear(in_features=n_feats, out_features=n_feats * 2)
        self.rnn = nn.GRU(n_feats * 2, n_feats * 2, num_layers=1, batch_first=True, bidirectional=True)
        self.fc2 = nn.Linear(in_features=n_feats * 2 * 2, out_features=n_class)

    def forward(self, spectrogram, **batch):
        h = self.fc1(spectrogram.transpose(1, 2))
        h = torch.clip(F.relu(h), 0, 20)
        h, hn = self.rnn(h)
        h = self.fc2(h)
        return {"logits": h}

    def transform_input_lengths(self, input_lengths):
        return input_lengths
