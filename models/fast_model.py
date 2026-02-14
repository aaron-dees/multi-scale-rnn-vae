import torch
import torch.nn as nn

class SinusoidalTimeEmbedding(nn.Module):
    """
    Classic transformer-style sinusoidal time embedding.
    Returns [B, T, E]
    """
    def __init__(self, emb_dim: int, max_len: int = 4096):
        super().__init__()
        self.emb_dim = emb_dim
        self.max_len = max_len

        pe = torch.zeros(max_len, emb_dim)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / emb_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, t_idx: torch.Tensor):
        """
        t_idx: [T] (long) or [B, T] (long)
        """
        if t_idx.dim() == 1:
            return self.pe[t_idx]  # [T, E]
        # [B,T] -> [B,T,E]
        return self.pe[t_idx]

class ConditionalFastARGaussian(nn.Module):
    """
    Conditional autoregressive model for fast residual:
      p(z_fast[t] | z_fast[t-1], z_slow[t], time_emb[t], e)

    - Autoregressive LSTM (2-layer by default)
    - Diagonal Gaussian output head (mu, logvar) over D dims per step

    Shapes:
      z_slow: [B, D, T]
      z_fast: [B, D, T]   (ground truth residual for teacher forcing / training)
      e:      [B, E]      (optional global conditioning)
    """
    def __init__(
        self,
        z_dim: int,
        e_dim: int = 0,
        time_emb_dim: int = 64,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.0,
        logvar_min: float = -8.0,
        logvar_max: float = 4.0,
        max_time: int = 4096,
        use_sinusoidal_time: bool = True,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.e_dim = e_dim
        self.time_emb_dim = time_emb_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max

        # time embedding (either sinusoidal or learned)
        if use_sinusoidal_time:
            self.time_emb = SinusoidalTimeEmbedding(time_emb_dim, max_len=max_time)
            self.time_is_sinusoidal = True
        else:
            self.time_emb = nn.Embedding(max_time, time_emb_dim)
            self.time_is_sinusoidal = False

        in_dim = z_dim + z_dim + time_emb_dim + (e_dim if e_dim > 0 else 0)
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=(dropout if num_layers > 1 else 0.0),
            batch_first=True,
        )

        self.fc_mu = nn.Linear(hidden_dim, z_dim)
        self.fc_logvar = nn.Linear(hidden_dim, z_dim)

        # Optional: initialize hidden state from e (helps conditioning be used)
        if e_dim > 0:
            self.h0_proj = nn.Linear(e_dim, num_layers * hidden_dim)
            self.c0_proj = nn.Linear(e_dim, num_layers * hidden_dim)
        else:
            self.h0_proj = None
            self.c0_proj = None

    def _init_state(self, B: int, device: torch.device, e=None):
        if self.e_dim > 0 and e is not None:
            h0 = self.h0_proj(e).view(B, self.num_layers, self.hidden_dim).transpose(0, 1).contiguous()
            c0 = self.c0_proj(e).view(B, self.num_layers, self.hidden_dim).transpose(0, 1).contiguous()
            return (h0, c0)
        h0 = torch.zeros(self.num_layers, B, self.hidden_dim, device=device)
        c0 = torch.zeros(self.num_layers, B, self.hidden_dim, device=device)
        return (h0, c0)

    @torch.no_grad()
    def sample(
        self,
        z_slow: torch.Tensor,         # [B,D,T]
        e= None,# [B,E]
        temperature: float = 1.0,
        z0 = None,   # [B,D] initial z_fast[-1] seed; if None -> zeros
    ) -> torch.Tensor:
        """
        Autoregressively sample z_fast sequence.
        Returns: z_fast_hat [B, D, T]
        """
        self.eval()
        B, D, T = z_slow.shape
        device = z_slow.device

        # time indices and embeddings: [B,T,E_t]
        t_idx = torch.arange(T, device=device).long().unsqueeze(0).expand(B, T)
        t_emb = self.time_emb(t_idx)  # [B,T,time_emb_dim]

        # prepare conditioning e expanded per timestep if needed
        if self.e_dim > 0 and e is not None:
            e_rep = e.unsqueeze(1).expand(B, T, self.e_dim)  # [B,T,E]
        else:
            e_rep = None

        # init
        prev = torch.zeros(B, D, device=device) if z0 is None else z0
        state = self._init_state(B, device, e)

        outs = []
        for t in range(T):
            slow_t = z_slow[:, :, t]  # [B,D]
            emb_t = t_emb[:, t, :]    # [B,Et]
            if e_rep is not None:
                inp_t = torch.cat([prev, slow_t, emb_t, e_rep[:, t, :]], dim=-1)
            else:
                inp_t = torch.cat([prev, slow_t, emb_t], dim=-1)

            # LSTM expects [B,1,in_dim]
            out_t, state = self.lstm(inp_t.unsqueeze(1), state)  # out_t: [B,1,H]
            h_t = out_t[:, 0, :]  # [B,H]

            mu_t = self.fc_mu(h_t)
            logvar_t = torch.clamp(self.fc_logvar(h_t), self.logvar_min, self.logvar_max)
            std_t = torch.exp(0.5 * logvar_t) * temperature
            eps = torch.randn_like(std_t)
            z_t = mu_t + eps * std_t

            outs.append(z_t)
            prev = z_t

        return torch.stack(outs, dim=2)  # [B,D,T]

    def forward(
        self,
        z_slow: torch.Tensor,          # [B,D,T]
        z_fast: torch.Tensor,          # [B,D,T] ground truth residual
        e= None, # [B,E] optional
        teacher_forcing_prob: float = 1.0,
        z0 = None,   # [B,D] initial seed for prev; if None -> zeros
        sample_in_forward: bool = False,  # if True, feed sampled output when not teacher forcing
        temperature: float = 1.0,
    ):
        """
        Returns:
          mu:     [B,D,T]
          logvar: [B,D,T]
          z_hat:  [B,D,T] (the generated residual sequence)
        """
        B, D, T = z_fast.shape
        device = z_fast.device
        assert z_slow.shape == z_fast.shape, "z_slow and z_fast must have same shape [B,D,T]"

        # time embeddings [B,T,Et]
        t_idx = torch.arange(T, device=device).long().unsqueeze(0).expand(B, T)
        t_emb = self.time_emb(t_idx)

        if self.e_dim > 0 and e is not None:
            e_rep = e.unsqueeze(1).expand(B, T, self.e_dim)
        else:
            e_rep = None

        prev = torch.zeros(B, D, device=device) if z0 is None else z0
        state = self._init_state(B, device, e)

        mu_seq, logvar_seq, zhat_seq = [], [], []

        for t in range(T):
            slow_t = z_slow[:, :, t]
            emb_t = t_emb[:, t, :]
            if e_rep is not None:
                inp_t = torch.cat([prev, slow_t, emb_t, e_rep[:, t, :]], dim=-1)
            else:
                inp_t = torch.cat([prev, slow_t, emb_t], dim=-1)

            out_t, state = self.lstm(inp_t.unsqueeze(1), state)
            h_t = out_t[:, 0, :]

            mu_t = self.fc_mu(h_t)
            logvar_t = torch.clamp(self.fc_logvar(h_t), self.logvar_min, self.logvar_max)

            mu_seq.append(mu_t)
            logvar_seq.append(logvar_t)

            # choose what to "emit" for z_hat[t]
            if sample_in_forward:
                std_t = torch.exp(0.5 * logvar_t) * temperature
                z_t = mu_t + torch.randn_like(std_t) * std_t
            else:
                z_t = mu_t

            zhat_seq.append(z_t)

            # scheduled teacher forcing for next-step input
            if teacher_forcing_prob >= 1.0:
                prev = z_fast[:, :, t]
            elif teacher_forcing_prob <= 0.0:
                prev = z_t
            else:
                use_gt = (torch.rand(1, device=device).item() < teacher_forcing_prob)
                prev = z_fast[:, :, t] if use_gt else z_t

        mu = torch.stack(mu_seq, dim=2)         # [B,D,T]
        logvar = torch.stack(logvar_seq, dim=2) # [B,D,T]
        z_hat = torch.stack(zhat_seq, dim=2)    # [B,D,T]
        return z_hat, mu, logvar