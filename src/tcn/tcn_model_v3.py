import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size]


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()

        padding = (kernel_size - 1) * dilation

        self.net = nn.Sequential(
            weight_norm(nn.Conv1d(in_channels, out_channels,
                                  kernel_size, dilation=dilation, padding=padding)),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),

            weight_norm(nn.Conv1d(out_channels, out_channels,
                                  kernel_size, dilation=dilation, padding=padding)),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else None

        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNMultiTowers(nn.Module):
    """
    TCN multi-head para 1h, 12h, 24h, 72h, 168h
    OPTIMIZADO: Reducida complejidad para evitar overfitting
    - Canales: 64→128→128→64 (vs anterior 128→256→256→512→512)
    - Dropout: 0.4 (vs anterior 0.2)
    - Capas: 4 bloques (vs anterior 5)
    """

    def __init__(self, num_inputs,
                 num_channels=[64, 128, 128, 64],  # Reducido 70%
                 kernel_size=3,
                 dropout=0.4):  # Aumentado para regularización
        super().__init__()

        layers = []
        for i in range(len(num_channels)):
            dilation = 2 ** i
            in_ch = num_inputs if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, dilation, dropout))

        self.network = nn.Sequential(*layers)

        hidden_size = num_channels[-1]

        # Multi-head specialist towers (simplificados)
        def build_head():
            return nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),  # Agregar dropout en heads
                nn.Linear(hidden_size // 2, 1)
            )

        self.head_1h = build_head()
        self.head_12h = build_head()
        self.head_24h = build_head()
        self.head_72h = build_head()
        self.head_168h = build_head()

    def forward(self, x):
        # x: (batch, features, seq_len)
        y = self.network(x)          # → (batch, channels, seq_len)
        y = y[:, :, -1]              # → último paso

        return torch.cat([
            self.head_1h(y),
            self.head_12h(y),
            self.head_24h(y),
            self.head_72h(y),
            self.head_168h(y)
        ], dim=1)
