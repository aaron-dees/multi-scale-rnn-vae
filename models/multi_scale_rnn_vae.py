
import sys
sys.path.append('../')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import random

from models.transformer import SlowTransformerDecoder
from models.rnn_decoders import MultiScaleDecoder_slow_seq, MultiScaleDecoder_slow, MultiScaleDecoder_slow_nonAuto, MultiScaleDecoder, MultiScaleDecoder_slow_gaussian
from models.encoders import SlowWindowEncoder, TemporalEncoder, ConvTemporalEncoder

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
    
class MultiScaleRNNVAE_lowPass(nn.Module):
    def __init__(
        self,
        z_dim,
        e_dim,
        enc_hidden_dim,
        slow_hidden_dim,
        fast_hidden_dim,
        K,
        context_window_size,
        context_window_hidden_dim,
        context_encoder_pooling_rate
    ):
        super().__init__()

        self.encoder = TemporalEncoder(
            input_dim=z_dim,
            hidden_dim=enc_hidden_dim,
            e_dim=e_dim
        )

        # self.encoder = ConvTemporalEncoder(
        #     z_dim=z_dim,
        #     window_size=context_window_size,
        #     hidden_dim=context_window_hidden_dim,
        #     e_dim=e_dim,
        #     pooling_rate=context_encoder_pooling_rate
        # )

        # self.decoder = MultiScaleDecoder_lowPass(
        #     z_dim=z_dim,
        #     e_dim=e_dim,
        #     slow_hidden_dim=slow_hidden_dim,
        #     fast_hidden_dim=fast_hidden_dim,
        #     K=K
        # )

        # self.decoder = MultiScaleDecoder_slow(
        #     z_dim=z_dim,
        #     e_dim=e_dim,
        #     slow_hidden_dim=slow_hidden_dim,
        #     fast_hidden_dim=fast_hidden_dim,
        #     K=K,
        #     window_size=context_window_size,
        #     window_hidden_dim = context_window_hidden_dim,
        #     context_encoder_pooling_rate=context_encoder_pooling_rate
        # )

        # self.decoder = MultiScaleDecoder_slow_gaussian(
        #     z_dim=z_dim,
        #     e_dim=e_dim,
        #     slow_hidden_dim=slow_hidden_dim,
        #     fast_hidden_dim=fast_hidden_dim,
        #     K=K,
        #     window_size=context_window_size,
        #     window_hidden_dim = context_window_hidden_dim
        # )


        self.decoder = MultiScaleDecoder_slow_seq(
            z_dim=z_dim,
            e_dim=e_dim,
            slow_hidden_dim=slow_hidden_dim,
            num_layers=2,
            window_hidden_dim=context_window_hidden_dim,
            T_pred=56,
            context_encoder_pooling_rate=context_encoder_pooling_rate
        )

        # self.decoder = SlowTransformerDecoder(
        #     slow_dim=z_dim,
        #     latent_dim=e_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, z):
        mu, logvar = self.encoder(z)
        e = self.reparameterize(mu, logvar)
        return e, mu, logvar
    

    def forward(self, z_ctx, z, z_slow, z_fast, teacher_forcing=True, teacher_forcing_prob=0.2):
        """
        x: [B, D, T]
        """
        # mu_e, logvar_e = self.encoder(z)
        # mu_e, logvar_e = self.encoder(z_slow)
        mu_e, logvar_e = self.encoder(z_ctx)
        e = self.reparameterize(mu_e, logvar_e)
        # e = torch.rand_like(e) * 10
        # e = mu_e

        # z_hat = self.decoder(z_ctx, z, z_slow, z_fast, e, teacher_forcing, teacher_forcing_prob)
        # z_hat, mu_z, logvar_z = self.decoder(z_ctx, z, z_slow, z_fast, e, teacher_forcing, teacher_forcing_prob)
        z_hat = self.decoder(e)

        # return z_hat, mu_z, logvar_z, mu_e, logvar_e
        return z_hat, mu_e, logvar_e
    
class MultiScaleRNNVAE_slowRate(nn.Module):
    def __init__(
        self,
        z_dim,
        e_dim,
        enc_hidden_dim,
        slow_hidden_dim,
        context_window_size,
        context_window_hidden_dim,
        context_encoder_pooling_rate,
        target_window_size
    ):
        super().__init__()

        self.encoder = TemporalEncoder(
            input_dim=z_dim,
            hidden_dim=enc_hidden_dim,
            e_dim=e_dim
        )

        self.decoder = MultiScaleDecoder_slow_seq(
            z_dim=z_dim,
            e_dim=e_dim,
            slow_hidden_dim=slow_hidden_dim,
            num_layers=2,
            window_hidden_dim=context_window_hidden_dim,
            T_pred=target_window_size,
            context_encoder_pooling_rate=context_encoder_pooling_rate
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, z):
        mu, logvar = self.encoder(z)
        e = self.reparameterize(mu, logvar)
        return e, mu, logvar

    def decode(self, e):
        return self.decoder(e)

    def forward(self, z_ctx):
        """
        x: [B, D, T]
        """
        mu_e, logvar_e = self.encoder(z_ctx)
        e = self.reparameterize(mu_e, logvar_e)
        # e = torch.rand_like(e) * 10
        # e = mu_e

        z_hat = self.decoder(e)

        return z_hat, mu_e, logvar_e