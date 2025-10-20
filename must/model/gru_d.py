#!/usr/bin/env python
# encoding: utf-8

import torch
import torch.nn as nn

from .rits import TemporalDecay


class GRUD(nn.Module):
    def __init__(self, input_size: int, rnn_hidden_size: int = 32, n_classes: int = 2, dropout_rate: float = 0.5):
        super(GRUD, self).__init__()

        self.input_size = input_size
        self.rnn_hidden_size = rnn_hidden_size
        self.n_classes = n_classes

        self.rnn_cell = nn.LSTMCell(self.input_size * 2, self.rnn_hidden_size)

        self.temp_decay_h = TemporalDecay(input_size=self.input_size, output_size=self.rnn_hidden_size, diag=False)
        self.temp_decay_x = TemporalDecay(input_size=self.input_size, output_size=self.input_size, diag=True)

        self.hist_reg = nn.Linear(self.rnn_hidden_size, self.input_size)
        self.feat_reg = nn.Linear(self.input_size, self.input_size)

        self.weight_combine = nn.Linear(self.input_size * 2, self.input_size)

        self.dropout = nn.Dropout(dropout_rate)
        self.out = nn.Linear(self.rnn_hidden_size, self.n_classes)

    def forward(self, values: torch.Tensor, masks: torch.Tensor, deltas: torch.Tensor, forwards: torch.Tensor):
        batch_size, seq_len, input_size = values.size()

        h = torch.zeros(batch_size, self.rnn_hidden_size, device=values.device)
        c = torch.zeros(batch_size, self.rnn_hidden_size, device=values.device)

        impute_list = []
        masks_float = masks.float()
        for t in range(seq_len):
            x = values[:, t, :]
            m = masks_float[:, t, :]
            d = deltas[:, t, :]
            f = forwards[:, t, :]

            gamma_h = self.temp_decay_h(d)
            gamma_x = self.temp_decay_x(d)

            h = h * gamma_h

            x_h = m * x + (1 - m) * (1 - gamma_x) * f
            inputs = torch.cat([x_h, m], dim=1)
            h, c = self.rnn_cell(inputs, (h, c))
            impute_list.append(x_h.unsqueeze(dim=1))

        impute_tensor = torch.cat(impute_list, dim=1)
        y_h = self.out(self.dropout(h))

        return y_h, impute_tensor
