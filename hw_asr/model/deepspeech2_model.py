import torch
from torch import nn
import torch.nn.functional as F

from hw_asr.base import BaseModel


class DeepSpeech2(BaseModel):
    def __init__(self, n_feats, n_class, fc_hidden=512, **batch):
        super().__init__(n_feats, n_class, **batch)

        self.fc_hidden = fc_hidden

        self.fc1 = nn.Linear(in_features=n_feats, out_features=fc_hidden)

        self.kernel_size = 5
        self.stride = 2
        self.conv = nn.Conv1d(fc_hidden, fc_hidden, self.kernel_size, stride=self.stride)

        self.rnn = nn.GRU(fc_hidden, fc_hidden, num_layers=3, batch_first=True, bidirectional=True)

        self.fc2 = nn.Linear(in_features=fc_hidden, out_features=fc_hidden)
        self.fc3 = nn.Linear(in_features=fc_hidden, out_features=n_class)

    def forward(self, spectrogram, **batch):
        h = self.fc1(spectrogram.transpose(1, 2))
        h = torch.clip(F.relu(h), 0, 20)

        h = self.conv(h.transpose(1, 2))
        h = torch.clip(F.relu(h), 0, 20)

        h, _ = self.rnn(h.transpose(1, 2))
        h = h[:, :, :self.fc_hidden] + h[:, :, self.fc_hidden:]

        h = self.fc2(h)
        h = torch.clip(F.relu(h), 0, 20)

        h = self.fc3(h)
        return {"logits": h}

    def transform_input_lengths(self, input_lengths):
        return (input_lengths - (self.kernel_size1 - 1) - 1) // self.stride + 1
