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
    OPTIMIZADO v2: Mejor receptive field y capacidad balanceada
    
    Receptive Field Calculation:
    - Con dilations=[1,2,4,8,16,32] y kernel=3:
    - RF ≈ 2 * kernel * sum(dilations) = 2 * 3 * 63 = 378 pasos
    - Cubre toda la ventana de 168 horas + margen
    
    - Canales: 64→128→128→128→64→32 (balance capacidad/regularización)
    - Dropout: 0.4 (regularización fuerte)
    - Capas: 6 bloques (suficiente para 168h)
    """

    def __init__(self, num_inputs,
                 num_channels=[64, 128, 128, 128, 64, 32],  # MEJORADO: Más profundo
                 kernel_size=3,
                 dropout=0.4,  # Regularización fuerte
                 use_global_pooling=False):  # NUEVO: Opción de pooling
        super().__init__()
        
        self.use_global_pooling = use_global_pooling

        layers = []
        for i in range(len(num_channels)):
            dilation = 2 ** i  # [1,2,4,8,16,32] para 6 capas
            in_ch = num_inputs if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, dilation, dropout))
        
        # Receptive field total
        self.receptive_field = sum([2 ** i * (kernel_size - 1) for i in range(len(num_channels))]) + 1

        self.network = nn.Sequential(*layers)

        hidden_size = num_channels[-1]
        
        # Si usamos global pooling, el tamaño de entrada cambia
        if use_global_pooling:
            hidden_size = hidden_size * 2  # avg + max pooling

        # Multi-head specialist towers (optimizados)
        def build_head():
            return nn.Sequential(
                nn.Linear(hidden_size, hidden_size),  # MEJORADO: Mejor capacidad
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5),  # Dropout más ligero en capa final
                nn.Linear(hidden_size // 2, 1)
            )

        self.head_1h = build_head()
        self.head_12h = build_head()
        self.head_24h = build_head()
        self.head_72h = build_head()
        self.head_168h = build_head()
        
        print(f"✅ TCN inicializado:")
        print(f"   - Receptive field: {self.receptive_field} pasos")
        print(f"   - Canales: {num_channels}")
        print(f"   - Dilations: {[2**i for i in range(len(num_channels))]}")
        print(f"   - Global pooling: {use_global_pooling}")

    def forward(self, x):
        # x: (batch, features, seq_len)
        y = self.network(x)          # → (batch, channels, seq_len)
        
        if self.use_global_pooling:
            # OPCIÓN 1: Global pooling (mejor para capturar patrones globales)
            y_avg = torch.mean(y, dim=2)  # Promedio temporal
            y_max = torch.max(y, dim=2)[0]  # Máximo temporal
            y = torch.cat([y_avg, y_max], dim=1)  # Concatenar ambos
        else:
            # OPCIÓN 2: Último paso (estándar para series temporales)
            y = y[:, :, -1]              # → último paso

        return torch.cat([
            self.head_1h(y),
            self.head_12h(y),
            self.head_24h(y),
            self.head_72h(y),
            self.head_168h(y)
        ], dim=1)
