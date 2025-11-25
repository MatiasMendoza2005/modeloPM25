import torch
import torch.nn as nn
from .tcn_layer import TCN

class TCNRegressor(nn.Module):
    def __init__(self, num_features, seq_length):
        super().__init__()

        # TCN recibe entrada como (batch, features, time)
        self.tcn = TCN(
            num_inputs=num_features,
            ##num_channels=[64, 64, 64, 64],  # v1
            num_channels=[64, 128, 128, 64],  # v2
            kernel_size=3,
            dropout=0.1
        )

        # fully connected layer to output prediction
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        # x: (batch, seq_len, features)
        x = x.permute(0, 2, 1)  # --> (batch, features, seq_len)

        y = self.tcn(x)

        # last time step: y[:, :, -1] → shape (batch, channels)
        y_last = y[:, :, -1]

        out = self.fc(y_last)
        return out.squeeze(-1)
