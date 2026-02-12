import sys
sys.path.append('../')

from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
from pathlib import Path
import os
import torch.nn as nn
import torch
import torchaudio
from torch.utils.tensorboard import SummaryWriter


from models.dataloaders import LatentTextureDataset_new
from models.multi_scale_rnn_vae import MultiScaleRNNVAE_slowRate
from models.loss_functions import vae_loss_mse, vae_slow_loss, vae_prob_loss
from scripts.config import Config
from utils.utilities import plot_latent_pca, lowpass_moving_average

from music2latent import EncoderDecoder


def build_training_objects(cfg, latent_files):
    # dataset = LatentTextureDataset(
    #     latent_files=latent_files,
    #     window_size=cfg.window_size
    # )

    dataset = LatentTextureDataset_new(
        latent_files=latent_files,
        window_size=cfg.window_size,
        stride=cfg.dataset_stride,
        kernel_size=cfg.K   
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=True
    )

    model = MultiScaleRNNVAE_slowRate(
        z_dim=cfg.latent_dim,
        e_dim=cfg.e_dim,
        enc_hidden_dim=cfg.enc_hidden_dim,
        slow_hidden_dim=cfg.slow_hidden_dim,
        context_window_size=cfg.context_window_size,
        context_window_hidden_dim=cfg.context_window_hidden_dim,
        context_encoder_pooling_rate=cfg.context_encoder_pooling_rate,
        target_window_size=cfg.target_window_size
    ).to(cfg.device)

    optimizer = Adam(model.parameters(), lr=cfg.lr)

    return model, optimizer, loader

def train(cfg, latent_dir, save_dir):

    writer = SummaryWriter(log_dir="./runs/test_run")

    os.makedirs(save_dir, exist_ok=True)

    model, optimizer, loader = build_training_objects(cfg, latent_dir)

    model.train()

    codec = EncoderDecoder()

    for epoch in range(cfg.epochs):
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        total_mu = 0.0
        total_std = 0.0

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{cfg.epochs}")

        for _, z_slow in pbar:
            z_slow_target = z_slow[:, :, cfg.context_window_size:]  # remove context from main input
            z_slow_ctx = z_slow[:, :, :cfg.context_window_size]

            optimizer.zero_grad()

            if epoch + 1 < 200:
                beta = 0.0
            else:
                beta = min(1.0, epoch / 500) * cfg.beta
            # beta = max(1e-4, min(1.0, epoch / 500)) * cfg.beta
            # beta = 0 

            outputs = model(z_slow_ctx)

            loss, recon, kl = vae_slow_loss(z_slow_target, outputs, beta=beta)

            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon.item()
            total_kl += kl.item()
            total_mu += outputs[1].mean().item()
            total_std += torch.exp(0.5 * outputs[2]).mean().item()

            pbar.set_postfix({
                "loss": loss.item(),
                "recon": recon.item(),
                "kl": kl.item()
            })

        avg_loss = total_loss / len(loader)
        avg_recon = total_recon / len(loader)
        avg_kl = total_kl / len(loader)
        avg_mu = total_mu / len(loader)
        avg_std = total_std / len(loader)

        print(
            f"[Epoch {epoch+1}] "
            f"Loss: {avg_loss:.4f} | "
            f"Recon: {avg_recon:.4f} | "
            f"KL: {avg_kl:.4f} | "
            f"Beta: {beta:.4f}"
        )


        if (epoch + 1) % 5 == 0:
            writer.add_scalar("Loss/Train", avg_loss, epoch+1)
            writer.add_scalar("Recon/Train", avg_recon, epoch+1)
            writer.add_scalar("KL/Train", avg_kl, epoch+1)
            writer.add_scalar("Mu/Train", avg_mu, epoch+1)
            writer.add_scalar("Std/Train", avg_std, epoch+1)

        if (epoch + 1) % 25 == 0:
            for _, z_val_slow in loader:
                z_slow_target = z_val_slow[:, :, cfg.context_window_size:]  # remove context from main input
                z_slow_ctx = z_val_slow[:, :, :cfg.context_window_size]
                z_hat_val, _, _ = model(z_slow_ctx)
                break  # take first batch only

            plot_latent_pca(z_slow_target[0:1].detach(), z_hat_val[0:1].detach(), epoch+1)
            if (epoch + 1) % 100 == 0:
                recon = codec.decode(z_hat_val)
                orig = codec.decode(z_slow_target)
                for i in range(recon.shape[0]):
                    writer.add_audio(f"Recon/GenLatent_{i}", recon[i:i+1], sample_rate=44100, global_step=epoch+1)
                    writer.add_audio(f"Recon/Orig_{i}", orig[i:i+1], sample_rate=44100, global_step=epoch+1)
                    torchaudio.save(f"./reconstructions/recon_{i}_{epoch+1}.wav", recon[i:i+1], sample_rate=44100)
                    torchaudio.save(f"./reconstructions/orig_{i}_{epoch+1}.wav", orig[i:i+1], sample_rate=44100)
        if (epoch + 1) % 500 == 0:
            torch.save(
                model.state_dict(),
                os.path.join(save_dir, f"model_epoch_{epoch+1}.pt")
            )

if __name__ == "__main__":


    data_dir = Path("/Users/adees/Code/multi-scale-rnn-vae/test_audio/latents")
    ext = ".pt"
    train_files = [str(p) for p in sorted(data_dir.glob(f"*{ext}"))]

    if not train_files:
        raise FileNotFoundError(f"No '{ext}' files found in {data_dir}")
    
    save_dir = "./checkpoints"

    cfg = Config()
    train(cfg, train_files, save_dir)
