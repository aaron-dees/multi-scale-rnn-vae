import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import random

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
        logvar = self.fc_logvar(h)

        return mu, logvar


class MultiScaleDecoder(nn.Module):
    def __init__(
        self,
        z_dim,
        e_dim,
        slow_hidden_dim,
        fast_hidden_dim,
        K
    ):
        super().__init__()

        self.K = K

        self.slow_rnn = nn.GRUCell(
            input_size=e_dim,
            hidden_size=slow_hidden_dim
        )

        self.fast_rnn = nn.GRUCell(
            input_size=z_dim + slow_hidden_dim + e_dim,
            hidden_size=fast_hidden_dim
        )

        self.fc_mu = nn.Linear(fast_hidden_dim, z_dim)
        self.fc_logvar = nn.Linear(fast_hidden_dim, z_dim)

    def forward(self, z, e, teacher_forcing=True):
        """
        x: [B, D, T]   (ground truth, used for teacher forcing)
        z: [B, Z]
        """
        B, D, T = z.shape
        device = z.device

        mu_out = []
        logvar_out = []

        h_slow = torch.zeros(B, self.slow_rnn.hidden_size, device=device)
        h_fast = torch.zeros(B, self.fast_rnn.hidden_size, device=device)

        z_prev = torch.zeros(B, D, device=device)

        for t in range(T):

            if t % self.K == 0:
                h_slow = self.slow_rnn(e, h_slow)

            fast_input = torch.cat([z_prev, h_slow, e], dim=-1)
            h_fast = self.fast_rnn(fast_input, h_fast)

            mu_t = self.fc_mu(h_fast)
            logvar_t = self.fc_logvar(h_fast)

            mu_out.append(mu_t)
            logvar_out.append(logvar_t)

            if teacher_forcing:
                z_prev = z[:, :, t]
            else:
                std = torch.exp(0.5 * logvar_t)
                z_prev = mu_t + std * torch.randn_like(std)
                # z_prev = mu_t

        mu_out = torch.stack(mu_out, dim=2)       # [B, D, T]
        logvar_out = torch.stack(logvar_out, dim=2)

        return mu_out, logvar_out

class MultiScaleDecoder_deterministic(nn.Module):
    def __init__(
        self,
        z_dim,
        e_dim,
        slow_hidden_dim,
        fast_hidden_dim,
        K
    ):
        super().__init__()

        self.K = K

        self.slow_rnn = nn.GRUCell(
            input_size=e_dim,
            hidden_size=slow_hidden_dim
        )

        self.fast_rnn = nn.GRUCell(
            # input_size=z_dim + slow_hidden_dim + e_dim,
            input_size=z_dim + slow_hidden_dim,
            hidden_size=fast_hidden_dim
        )

        self.fc_out = nn.Linear(fast_hidden_dim, z_dim)

    def forward(self, z, e, teacher_forcing=True):
        """
        x: [B, D, T]   (ground truth, used for teacher forcing)
        e: [B, E]
        """
        B, D, T = z.shape
        device = z.device

        z_hat = []

        h_slow = torch.zeros(B, self.slow_rnn.hidden_size, device=device)
        h_fast = torch.zeros(B, self.fast_rnn.hidden_size, device=device)

        z_prev = torch.zeros(B, D, device=device)

        for t in range(T):

            if t % self.K == 0:
                h_slow = self.slow_rnn(e, h_slow)

            # fast_input = torch.cat([z_prev, h_slow, e], dim=-1)
            fast_input = torch.cat([z_prev, h_slow], dim=-1)
            h_fast = self.fast_rnn(fast_input, h_fast)

            # z_hat_t = self.fc_out(h_fast)
            delta = self.fc_out(h_fast)

            z_hat_t = z_prev + delta

            z_hat.append(z_hat_t)

            if teacher_forcing:
                use_gt = torch.rand(1).item() < 0.5
                if use_gt:
                    z_prev = z[:, :, t]
                else:
                    z_prev = z_hat_t
            else:
                z_prev = z_hat_t

        z_hat = torch.stack(z_hat, dim=2)       # [B, D, T]

        return z_hat 
    
class MultiScaleRNNVAE(nn.Module):
    def __init__(
        self,
        z_dim,
        e_dim,
        enc_hidden_dim,
        slow_hidden_dim,
        fast_hidden_dim,
        K
    ):
        super().__init__()

        self.encoder = TemporalEncoder(
            input_dim=z_dim,
            hidden_dim=enc_hidden_dim,
            e_dim=e_dim
        )

        self.decoder = MultiScaleDecoder(
            z_dim=z_dim,
            e_dim=e_dim,
            slow_hidden_dim=slow_hidden_dim,
            fast_hidden_dim=fast_hidden_dim,
            K=K
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, z, teacher_forcing=True):
        """
        x: [B, D, T]
        """
        mu_e, logvar_e = self.encoder(z)
        e = self.reparameterize(mu_e, logvar_e)

        mu_z, logvar_z = self.decoder(z, e, teacher_forcing)

        return mu_z, logvar_z, mu_e, logvar_e
    
class MultiScaleRNNVAE_deterministic(nn.Module):
    def __init__(
        self,
        z_dim,
        e_dim,
        enc_hidden_dim,
        slow_hidden_dim,
        fast_hidden_dim,
        K
    ):
        super().__init__()

        self.encoder = TemporalEncoder(
            input_dim=z_dim,
            hidden_dim=enc_hidden_dim,
            e_dim=e_dim
        )

        self.decoder = MultiScaleDecoder_deterministic(
            z_dim=z_dim,
            e_dim=e_dim,
            slow_hidden_dim=slow_hidden_dim,
            fast_hidden_dim=fast_hidden_dim,
            K=K
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, z, teacher_forcing=True):
        """
        x: [B, D, T]
        """
        mu_e, logvar_e = self.encoder(z)
        e = self.reparameterize(mu_e, logvar_e)

        z_hat = self.decoder(z, e, teacher_forcing)

        return z_hat, mu_e, logvar_e