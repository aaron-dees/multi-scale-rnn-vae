import torch
import torch.nn.functional as F

def gaussian_nll(z, mu, logvar):
    """
    x, mu, logvar: [B, D, T]
    """
    return 0.5 * (
        logvar +
        (z - mu) ** 2 / torch.exp(logvar)
    ).sum(dim=[1, 2]).mean()

def kl_divergence(mu, logvar):
    return -0.5 * torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp(),
        dim=1
    ).mean()

def vae_loss(z, outputs, beta=1.0):
    mu_z, logvar_z, mu_e, logvar_e = outputs

    recon = gaussian_nll(z, mu_z, logvar_z)
    kl = kl_divergence(mu_e, logvar_e)

    return recon + beta * kl, recon, kl

def vae_loss_mse(z, outputs, beta=1.0):
    z_hat, mu_e, logvar_e = outputs

    recon = mse_loss(z_hat, z)
    kl = kl_divergence(mu_e, logvar_e)

    return recon + beta * kl, recon, kl

def mse_loss(z_pred, z):
    recon = F.mse_loss(z_pred, z, reduction='mean')
    # kl = kl_divergence(mu, logvar)
    return recon