from torch import nn
import torch.nn.functional as F

from hw_asr.base import BaseModel


class ConvGRUModel(BaseModel):
    def __init__(self, n_feats, n_class, fc_hidden=512, **batch):
        super().__init__(n_feats, n_class, **batch)
        self.kernel_size = 11
        self.conv = nn.Conv1d(n_feats, n_feats * 10, self.kernel_size, stride=2)
        self.rnn = nn.GRU(n_feats * 10, fc_hidden, num_layers=3, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(in_features=fc_hidden * 2, out_features=n_class)

    def forward(self, spectrogram, **batch):
        h = self.conv(spectrogram)
        h = F.relu(h)
        h, _ = self.rnn(h.transpose(1, 2))
        h = self.fc(h)
        return {"logits": h}

    def transform_input_lengths(self, input_lengths):
        return (input_lengths - (self.kernel_size - 1) - 1) // 2 + 1
