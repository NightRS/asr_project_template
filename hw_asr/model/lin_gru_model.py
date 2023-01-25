from torch import nn
import torch.nn.functional as F
import torch

from hw_asr.base import BaseModel


class LinGRUModel(BaseModel):
    def __init__(self, n_feats, n_class, fc_hidden=512, **batch):
        super().__init__(n_feats, n_class, **batch)
        self.fc_hidden = fc_hidden
        self.fc1 = nn.Linear(in_features=n_feats, out_features=fc_hidden)
        self.fc2 = nn.Linear(in_features=fc_hidden, out_features=fc_hidden)
        self.fc3 = nn.Linear(in_features=fc_hidden, out_features=fc_hidden)
        self.rnn = nn.GRU(fc_hidden, fc_hidden, num_layers=3, batch_first=True, bidirectional=True)
        self.fc4 = nn.Linear(in_features=fc_hidden, out_features=fc_hidden)
        self.fc5 = nn.Linear(in_features=fc_hidden, out_features=n_class)
        self.dropout = nn.Dropout(0.05)

    def forward(self, spectrogram, **batch):
        h = self.fc1(spectrogram.transpose(1, 2))
        h = torch.clip(F.relu(h), 0, 20)
        h = self.dropout(h)
        h = self.fc2(h)
        h = torch.clip(F.relu(h), 0, 20)
        h = self.dropout(h)
        h = self.fc3(h)
        h = torch.clip(F.relu(h), 0, 20)
        h = self.dropout(h)

        h, _ = self.rnn(h)
        h = h[:, :, :self.fc_hidden] + h[:, :, self.fc_hidden:]

        h = self.fc4(h)
        h = torch.clip(F.relu(h), 0, 20)
        h = self.dropout(h)
        h = self.fc5(h)
        return {"logits": h}

    def transform_input_lengths(self, input_lengths):
        return input_lengths
