import sys
sys.path.append('../')

import torch.nn as nn
import torch
import os

from models.encoders import SlowWindowEncoder

class MultiScaleDecoder_slow_seq(nn.Module):
    def __init__(self, z_dim, e_dim, window_hidden_dim, slow_hidden_dim, context_encoder_pooling_rate,
                 num_layers=2, T_pred=56, time_emb_dim=64):
        super().__init__()
        self.T_pred = T_pred

        self.window_encoder = SlowWindowEncoder(z_dim=z_dim, window_size=None, hidden_dim=window_hidden_dim, pooling_rate=context_encoder_pooling_rate)

        self.time_emb = nn.Embedding(T_pred, time_emb_dim)

        self.in_proj = nn.Sequential(
            nn.Linear(e_dim + time_emb_dim, slow_hidden_dim),
            nn.ReLU(),
            nn.Linear(slow_hidden_dim, slow_hidden_dim),
        )

        self.gru = nn.GRU(
            input_size=slow_hidden_dim,
            hidden_size=slow_hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        self.out_proj = nn.Linear(slow_hidden_dim, z_dim)

    def forward(self, e):
        # z_ctx: [B, D, T_ctx], e: [B, E]
        B = e.size(0)

        t_idx = torch.arange(self.T_pred, device=e.device)
        t = self.time_emb(t_idx).unsqueeze(0).expand(B, -1, -1)   # [B, T, Ht]

        # g_seq = g.unsqueeze(1).expand(B, self.T_pred, g.size(-1)) # [B, T, Hg]
        g_seq = e.unsqueeze(1).expand(B, self.T_pred, e.size(-1)) # [B, T, Hg]
        x = torch.cat([g_seq, t], dim=-1)                          # [B, T, Hg+Ht]
        x = self.in_proj(x)                                        # [B, T, H]

        h_out, _ = self.gru(x)                                     # [B, T, H]
        z_slow = self.out_proj(h_out).transpose(1, 2)              # [B, D, T]
        return z_slow

class MultiScaleDecoder_slow(nn.Module):
    def __init__(
        self,
        z_dim,
        e_dim,
        slow_hidden_dim,
        fast_hidden_dim,
        K,
        window_size,
        window_hidden_dim,
        context_encoder_pooling_rate
    ):
        super().__init__()

        self.K = K
        self.context_encoder_pooling_rate = context_encoder_pooling_rate

        self.window_size = window_size

        self.window_encoder = SlowWindowEncoder(
            z_dim=z_dim,
            window_size=window_size,
            hidden_dim=window_hidden_dim,
            pooling_rate=context_encoder_pooling_rate
        )

        self.slow_rnn = nn.GRUCell(
            # input_size=window_hidden_dim + e_dim,
            input_size=e_dim,
            # input_size=window_hidden_dim*self.context_encoder_pooling_rate + e_dim,
            # input_size=window_hidden_dim*self.context_encoder_pooling_rate,
            hidden_size=slow_hidden_dim
        )

        self.fc_slow = nn.Linear(slow_hidden_dim, z_dim)

    def forward(self, z_ctx, z, z_slow, z_fast, e, teacher_forcing=True, teacher_forcing_prob=0.2):
        """
        x: [B, D, T]   (ground truth, used for teacher forcing)
        e: [B, E]
        """
        B, D, T = z.shape
        device = z.device

        z_hat = []
        z_hat_slow = []

        h_slow = torch.zeros(B, self.slow_rnn.hidden_size, device=device)

        # z_prev = torch.zeros(B, D, device=device)
        z_slow_prev = z_ctx[:, :, -1]
        # z_hat_slow.append(z_slow_prev)


        for t in range(0,T):

            # z_ctx_window = z_ctx[:, :, -self.window_size:]  # [B, D, W]
            # context = self.window_encoder(z_ctx_window)

            # slow_input = torch.cat([context, e], dim=-1)
            slow_input = torch.cat([e], dim=-1)
            h_slow = self.slow_rnn(slow_input, h_slow)

            # delta_slow = self.fc_slow(h_slow)
            z_hat_slow_t = self.fc_slow(h_slow)

            # z_hat_slow_t = z_slow_prev + delta_slow
            z_hat_slow.append(z_hat_slow_t)

            if teacher_forcing:
                use_gt = torch.rand(1).item() < teacher_forcing_prob
                if use_gt:
                    z_slow_prev = z_slow[:, :, t]
                    z_ctx = torch.cat([z_ctx[:,:,1:], z_slow[:, :, t:t+1]], dim=-1)
                else:
                    z_slow_prev = z_hat_slow_t
                    z_ctx = torch.cat([z_ctx[:,:,1:], z_hat_slow_t.unsqueeze(-1)], dim=-1)
            else:
                z_slow_prev = z_hat_slow_t
                z_ctx = torch.cat([z_ctx[:,:,1:], z_hat_slow_t.unsqueeze(-1)], dim=-1)

        z_hat_slow = torch.stack(z_hat_slow, dim=2)       # [B, D, T]

        return z_hat_slow



class MultiScaleDecoder_slow_gaussian(nn.Module):
    def __init__(
        self,
        z_dim,
        e_dim,
        slow_hidden_dim,
        fast_hidden_dim,
        K,
        window_size,
        window_hidden_dim,
        logvar_min=-10.0,
        logvar_max=5.0,
    ):
        super().__init__()
        self.K = K
        self.window_size = window_size
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max

        self.window_encoder = SlowWindowEncoder(
            z_dim=z_dim,
            window_size=window_size,
            hidden_dim=window_hidden_dim
        )

        self.slow_rnn = nn.GRUCell(
            input_size=window_hidden_dim + e_dim,
            hidden_size=slow_hidden_dim
        )

        # Probabilistic head
        self.fc_mu = nn.Linear(slow_hidden_dim, z_dim)
        self.fc_logvar = nn.Linear(slow_hidden_dim, z_dim)

    def forward(
        self,
        z_ctx,      # [B, D, Wctx] (rolling context)
        z,          # [B, D, T] (not used directly except for T)
        z_slow,     # [B, D, T] (ground truth, teacher forcing)
        z_fast,
        e,          # [B, E]
        teacher_forcing=True,
        teacher_forcing_prob=0.2,
        sample_in_forward=False,   # if True, feed sampled z; else feed mean
        temperature=1.0,
    ):
        """
        Returns:
          mu:     [B, D, T]
          logvar: [B, D, T]
          z_hat:  [B, D, T]  (either mu or sampled, depending on sample_in_forward)
        """
        B, D, T = z.shape
        device = z.device

        h_slow = torch.zeros(B, self.slow_rnn.hidden_size, device=device)

        mu_seq = []
        logvar_seq = []
        z_hat_seq = []

        # start point (not strictly needed for absolute prediction, but used for rolling context)
        z_prev = z_ctx[:, :, -1]

        for t in range(T):
            # Use last window_size frames from rolling context
            z_ctx_window = z_ctx[:, :, -self.window_size:]  # [B, D, W]
            context = self.window_encoder(z_ctx_window)     # [B, H]

            slow_input = torch.cat([context, e], dim=-1)
            h_slow = self.slow_rnn(slow_input, h_slow)

            mu_t = self.fc_mu(h_slow)               # [B, D]
            logvar_t = self.fc_logvar(h_slow)       # [B, D]
            logvar_t = torch.clamp(logvar_t, self.logvar_min, self.logvar_max)

            # decide what to output / feed back
            if sample_in_forward:
                std_t = torch.exp(0.5 * logvar_t) * temperature
                eps = torch.randn_like(std_t)
                z_t = mu_t + eps * std_t
            else:
                z_t = mu_t

            mu_seq.append(mu_t)
            logvar_seq.append(logvar_t)
            z_hat_seq.append(z_t)

            # teacher forcing for rolling context
            if teacher_forcing:
                use_gt = (torch.rand(1, device=device).item() < teacher_forcing_prob)
                if use_gt:
                    z_prev = z_slow[:, :, t]
                    next_ctx = z_slow[:, :, t:t+1]
                else:
                    z_prev = z_t
                    next_ctx = z_t.unsqueeze(-1)
            else:
                z_prev = z_t
                next_ctx = z_t.unsqueeze(-1)

            # roll context: drop oldest, append newest
            z_ctx = torch.cat([z_ctx[:, :, 1:], next_ctx], dim=-1)

        mu = torch.stack(mu_seq, dim=2)         # [B, D, T]
        logvar = torch.stack(logvar_seq, dim=2) # [B, D, T]
        z_hat = torch.stack(z_hat_seq, dim=2)   # [B, D, T]

        return z_hat, mu, logvar



class MultiScaleDecoder_slow_nonAuto(nn.Module):
    def __init__(
        self,
        z_dim,
        e_dim,
        slow_hidden_dim,
        fast_hidden_dim,
        K,
        window_size,
        window_hidden_dim
    ):
        super().__init__()

        self.K = K

        self.window_size = window_size

        self.window_encoder = SlowWindowEncoder(
            z_dim=z_dim,
            window_size=window_size,
            hidden_dim=window_hidden_dim
        )

        self.slow_rnn = nn.GRUCell(
            input_size=window_hidden_dim + e_dim,
            hidden_size=slow_hidden_dim
        )

        self.fc_slow = nn.Linear(slow_hidden_dim, z_dim)

    def forward(self, z_ctx, z, z_slow, z_fast, e, teacher_forcing=True, teacher_forcing_prob=0.2):
        """
        x: [B, D, T]   (ground truth, used for teacher forcing)
        e: [B, E]
        """
        B, D, T = z.shape
        device = z.device

        z_hat = []
        z_hat_slow = []

        h_slow = torch.zeros(B, self.slow_rnn.hidden_size, device=device)

        # z_prev = torch.zeros(B, D, device=device)
        z_slow_prev = z_ctx[:, :, -1]
        # z_hat_slow.append(z_slow_prev)

        z_ctx_window = z_ctx  # [B, D, W]
        context = self.window_encoder(z_ctx_window)

        for t in range(0,T):

            slow_input = torch.cat([context, e], dim=-1)
            h_slow = self.slow_rnn(slow_input, h_slow)

            delta_slow = self.fc_slow(h_slow)
            # z_hat_slow_t = self.fc_slow(h_slow)

            z_hat_slow_t = z_slow_prev + delta_slow
            z_hat_slow.append(z_hat_slow_t)

            if teacher_forcing:
                use_gt = torch.rand(1).item() < teacher_forcing_prob
                if use_gt:
                    z_slow_prev = z_slow[:, :, t]
                else:
                    z_slow_prev = z_hat_slow_t
            else:
                z_slow_prev = z_hat_slow_t

        z_hat_slow = torch.stack(z_hat_slow, dim=2)       # [B, D, T]

        return z_hat_slow
    

# OLD
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
            input_size=z_dim + slow_hidden_dim,
            hidden_size=fast_hidden_dim
        )

        self.fc_mu = nn.Linear(fast_hidden_dim, z_dim)
        self.fc_logvar = nn.Linear(fast_hidden_dim, z_dim)

    def forward(self, z, e, teacher_forcing=True, teacher_forcing_prob=0.5):
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

        # z_prev = torch.zeros(B, D, device=device)
        z_prev = z[:, :, 0] 
        # z_hat.append(z_prev)
        mu_out.append(z_prev)
        logvar_out.append(torch.zeros_like(z_prev))

        for t in range(1,T):

            if t % self.K == 0:
                h_slow = self.slow_rnn(e, h_slow)

            fast_input = torch.cat([z_prev, h_slow], dim=-1)
            h_fast = self.fast_rnn(fast_input, h_fast)

            mu_t = self.fc_mu(h_fast)
            logvar_t = self.fc_logvar(h_fast)

            mu_out.append(mu_t)
            logvar_out.append(logvar_t)

            if teacher_forcing:
                use_gt = torch.rand(1).item() < teacher_forcing_prob
                if use_gt:
                    z_prev = z[:, :, t]
                else:
                    std = torch.exp(0.5 * logvar_t)
                    z_prev = mu_t + std * torch.randn_like(std) 
            else:
                std = torch.exp(0.5 * logvar_t)
                z_prev = mu_t + std * torch.randn_like(std) 

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
            # input_size=z_dim + slow_hidden_dim,
            input_size=z_dim,
            hidden_size=fast_hidden_dim
        )

        self.fc_out = nn.Linear(fast_hidden_dim, z_dim)

    def forward(self, z, e, teacher_forcing=True, teacher_forcing_prob=0.2):
        """
        x: [B, D, T]   (ground truth, used for teacher forcing)
        e: [B, E]
        """
        B, D, T = z.shape
        device = z.device

        z_hat = []

        h_slow = torch.zeros(B, self.slow_rnn.hidden_size, device=device)
        h_fast = torch.zeros(B, self.fast_rnn.hidden_size, device=device)

        # z_prev = torch.zeros(B, D, device=device)
        z_prev = z[:, :, 0] 
        z_hat.append(z_prev)

        for t in range(1,T):

            # if t % self.K == 0:
            #     h_slow = self.slow_rnn(e, h_slow)

            # fast_input = torch.cat([z_prev, h_slow, e], dim=-1)
            # fast_input = torch.cat([z_prev, h_slow], dim=-1)
            fast_input = torch.cat([z_prev], dim=-1)
            h_fast = self.fast_rnn(fast_input, h_fast)

            # z_hat_t = self.fc_out(h_fast)
            delta = self.fc_out(h_fast)

            z_hat_t = z_prev + delta

            z_hat.append(z_hat_t)

            if teacher_forcing:
                use_gt = torch.rand(1).item() < teacher_forcing_prob
                if use_gt:
                    z_prev = z[:, :, t]
                else:
                    z_prev = z_hat_t
            else:
                z_prev = z_hat_t

        z_hat = torch.stack(z_hat, dim=2)       # [B, D, T]

        return z_hat 
    
class MultiScaleDecoder_lowPass(nn.Module):
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
            input_size=z_dim + e_dim,
            hidden_size=slow_hidden_dim
        )

        self.fc_slow = nn.Linear(slow_hidden_dim, z_dim)

        self.fast_rnn = nn.GRUCell(
            # input_size=z_dim + slow_hidden_dim + e_dim,
            # input_size=z_dim + slow_hidden_dim,
            input_size=z_dim,
            hidden_size=fast_hidden_dim
        )

        self.fc_fast = nn.Linear(fast_hidden_dim, z_dim)

    def forward(self, z, z_slow, z_fast, e, teacher_forcing=True, teacher_forcing_prob=0.2):
        """
        x: [B, D, T]   (ground truth, used for teacher forcing)
        e: [B, E]
        """
        B, D, T = z.shape
        device = z.device

        z_hat = []
        z_hat_slow = []
        z_hat_fast = []

        h_slow = torch.zeros(B, self.slow_rnn.hidden_size, device=device)
        h_fast = torch.zeros(B, self.fast_rnn.hidden_size, device=device)

        # z_prev = torch.zeros(B, D, device=device)
        z_slow_prev = z_slow[:, :, 0]
        z_fast_prev = z_fast[:, :, 0]
        z_prev = z[:, :, 0]
        z_hat.append(z_prev)
        z_hat_slow.append(z_slow_prev)
        z_hat_fast.append(z_fast_prev)

        for t in range(1,T):

            slow_input = torch.cat([z_slow_prev, e], dim=-1)
            h_slow = self.slow_rnn(slow_input, h_slow)

            delta_slow = self.fc_slow(h_slow)

            z_hat_slow_t = z_slow_prev + delta_slow
            z_hat_slow.append(z_hat_slow_t)


            # # fast_input = torch.cat([z_prev, h_slow, e], dim=-1)
            # # fast_input = torch.cat([z_prev, h_slow], dim=-1)
            # fast_input = torch.cat([z_prev], dim=-1)
            # h_fast = self.fast_rnn(fast_input, h_fast)

            # # z_hat_t = self.fc_out(h_fast)
            # delta = self.fc_fast(h_fast)

            # # do i want to model delta between z_slow, or the previous z_fast?
            # z_hat_fast_t = z_fast_prev + delta
            # z_hat_fast.append(z_hat_fast_t)


            # z_hat_t = z_hat_slow_t + z_hat_fast_t
            # z_hat.append(z_hat_t)

            if teacher_forcing:
                use_gt = torch.rand(1).item() < teacher_forcing_prob
                if use_gt:
                    z_slow_prev = z_slow[:, :, t]
                    # z_fast_prev = z_fast[:, :, t]
                    # z_prev = z[:, :, t]
                else:
                    z_slow_prev = z_hat_slow_t
                    # z_fast_prev = z_hat_fast_t
                    # z_prev = z_hat_t
            else:
                z_slow_prev = z_hat_slow_t
                # z_fast_prev = z_hat_fast_t
                # z_prev = z_hat_t

        z_hat_slow = torch.stack(z_hat_slow, dim=2)       # [B, D, T]
        # z_hat_fast = torch.stack(z_hat_fast, dim=2)       # [B, D, T]
        # z_hat = torch.stack(z_hat, dim=2)       # [B, D, T]

        return z_hat_slow