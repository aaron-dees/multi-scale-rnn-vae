import sys
sys.path.append('../')

import torch.nn as nn
import torch

class TemporalEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, e_dim, num_layers=1):
        super().__init__()

        self.rnn = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        self.fc_mu = nn.Linear(2 * hidden_dim, e_dim)
        self.fc_logvar = nn.Linear(2 * hidden_dim, e_dim)

    def forward(self, z):
        """
        z: [B, D, T]
        """
        z = z.transpose(1, 2)          # [B, T, D]
        _, h = self.rnn(z)             # h: [2*num_layers, B, H]

        h = torch.cat([h[-2], h[-1]], dim=-1)  # [B, 2H]

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h).clamp(-8,4)

        return mu, logvar

class ConvTemporalEncoder(nn.Module):
    def __init__(self, z_dim, window_size, hidden_dim, e_dim, pooling_rate):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(z_dim, hidden_dim, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=7, padding=3),
            nn.ReLU(),
            # nn.AdaptiveAvgPool1d(1)  # collapse time
            nn.AdaptiveAvgPool1d(pooling_rate)  # collapse time
        )
        self.norm = nn.LayerNorm(hidden_dim)

        self.fc_mu = nn.Linear(hidden_dim, e_dim)
        self.fc_logvar = nn.Linear(hidden_dim, e_dim)

    def forward(self, z_window):
        # z_window: [B, D, W]
        h = self.conv(z_window)
        # h = torch.flatten(h, start_dim=1)
        h = h.mean(dim=-1)
        h = self.norm(h)

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h).clamp(-8,4)
        # return h.squeeze(-1)  # [B, hidden_dim]
        return mu, logvar

class SlowWindowEncoder(nn.Module):
    def __init__(self, z_dim, window_size, hidden_dim, pooling_rate):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(z_dim, hidden_dim, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=7, padding=3),
            nn.ReLU(),
            # nn.AdaptiveAvgPool1d(1)  # collapse time
            nn.AdaptiveAvgPool1d(pooling_rate)  # collapse time
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, z_window):
        # z_window: [B, D, W]
        h = self.conv(z_window)
        # h = torch.flatten(h, start_dim=1)
        h = h.mean(dim=-1)
        h = self.norm(h)
        # return h.squeeze(-1)  # [B, hidden_dim]
        return h
