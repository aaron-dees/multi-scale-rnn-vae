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

def gaussian_nll_diag(z_target, mu, logvar, reduce="mean"):
    """
    z_target, mu, logvar: [B, D, T]
    Returns scalar NLL (per element averaged if reduce="mean")
    """
    # elementwise nll
    nll = 0.5 * (logvar + (z_target - mu)**2 / torch.exp(logvar))

    if reduce == "mean":
        return nll.mean()
    elif reduce == "sum":
        return nll.sum()
    else:
        return nll

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

def vae_loss_mse(z, outputs, beta=1.0, w_mse=1.0, w_vel=0.5, w_pool=0.25):
    z_hat, mu_e, logvar_e = outputs

    recon = w_mse * mse_loss(z_hat, z)
    recon += w_vel * velocity_loss(z_hat, z)
    # recon += w_pool * avg_pool_loss(z_hat, z)
    kl = kl_divergence(mu_e, logvar_e)

    return recon + beta * kl, recon, kl

def vae_slow_loss(z_slow, outputs, beta=1.0, w_mse=1.0, w_vel=0.5, w_pool=0.25):
    z_hat_slow, mu_e, logvar_e = outputs

    # recon = w_mse * mse_loss(z_hat, z)
    # recon += w_vel * velocity_loss(z_hat, z)
    # recon += w_pool * avg_pool_loss(z_hat, z)
    recon = slow_loss(z_slow, z_hat_slow)
    kl = kl_divergence(mu_e, logvar_e)
    # kl = torch.tensor(0.0, device=z_slow.device)

    return recon + beta * kl, recon, kl

def mse_loss(z_hat, z):
    recon = F.mse_loss(z_hat, z, reduction='mean')
    # kl = kl_divergence(mu, logvar)
    return recon

def velocity_loss(z_hat, z_gt, reduction="mean"):
    """
    z_hat: [B, D, T]
    z_gt:  [B, D, T]
    """
    v_hat = z_hat[:, :, 1:] - z_hat[:, :, :-1]
    v_gt  = z_gt[:, :, 1:]  - z_gt[:, :, :-1]

    return F.mse_loss(v_hat, v_gt, reduction=reduction)

def normalized_velocity_loss(z_hat, z_gt, eps=1e-6):
    v_hat = z_hat[:, :, 1:] - z_hat[:, :, :-1]
    v_gt  = z_gt[:, :, 1:]  - z_gt[:, :, :-1]

    scale = v_gt.std(dim=-1, keepdim=True) + eps
    return F.mse_loss(v_hat / scale, v_gt / scale)


def avg_pool_loss(z_hat, z_gt, pool_sizes=(4, 8, 16)):
    """
    pool_sizes: temporal pooling factors
    """
    loss = 0.0

    for k in pool_sizes:
        if z_hat.size(-1) < k:
            continue

        z_hat_p = F.avg_pool1d(z_hat, kernel_size=k, stride=k)
        z_gt_p  = F.avg_pool1d(z_gt,  kernel_size=k, stride=k)

        loss += F.mse_loss(z_hat_p, z_gt_p)

    return loss / len(pool_sizes)

def slow_loss(
    z_slow_hat,
    z_slow_target,
    lambda_vel=0.5,
):
    loss = 0.1 * mse_loss(z_slow_hat, z_slow_target)
    loss += lambda_vel * velocity_loss(z_slow_hat, z_slow_target)
    return loss

# Look into these more
def fast_zero_mean_loss(z_fast_hat):
    """
    Encourages residuals to be zero-mean over time
    """
    mean = z_fast_hat.mean(dim=2)
    return torch.mean(mean ** 2)

def fast_energy_loss(z_fast_hat, z_fast_target):
    """
    Matches temporal variance of residuals
    """
    var_hat = z_fast_hat.var(dim=2)
    var_tgt = z_fast_target.var(dim=2)
    return F.mse_loss(var_hat, var_tgt)

def fast_loss(
    z_fast_hat,
    z_fast_target,
    lambda_mean=0.1,
    lambda_energy=0.1,
):
    loss = mse_loss(z_fast_hat, z_fast_target)
    loss += lambda_mean * fast_zero_mean_loss(z_fast_hat) # optional
    loss += lambda_energy * fast_energy_loss(z_fast_hat, z_fast_target) # optional
    return loss



def variance_match_loss(mu, z_target):
    # match temporal variance per dimension
    var_mu = mu.var(dim=2, unbiased=False)       # [B,D]
    var_gt = z_target.var(dim=2, unbiased=False) # [B,D]
    return F.mse_loss(var_mu, var_gt)

def vae_prob_loss(z_slow, outputs, beta=1.0, w_nll=1.0, w_vel=1.0, w_var=0.0):
    z_hat, mu_z, logvar_z, mu_e, logvar_e = outputs

    # recon = w_mse * mse_loss(z_hat, z)
    # recon += w_vel * velocity_loss(z_hat, z)
    # recon += w_pool * avg_pool_loss(z_hat, z)
    nll = gaussian_nll_diag(z_slow, mu_z, logvar_z)
    vel = velocity_loss(mu_z, z_slow)
    var = variance_match_loss(mu_z, z_slow)
    kl = kl_divergence(mu_e, logvar_e)

    recon = w_nll * nll + w_vel * vel + w_var * var

    return  recon + beta * kl, recon, nll, vel, var, kl
