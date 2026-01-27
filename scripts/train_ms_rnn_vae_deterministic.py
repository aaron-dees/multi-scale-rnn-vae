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


from models.dataloaders import LatentTextureDataset
from models.multi_scale_rnn_vae import MultiScaleRNNVAE_deterministic
from models.loss_functions import vae_loss_mse
from scripts.config import Config

from music2latent import EncoderDecoder


def build_training_objects(cfg, latent_files):
    dataset = LatentTextureDataset(
        latent_files=latent_files,
        window_size=cfg.window_size
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        pin_memory=True
    )

    model = MultiScaleRNNVAE_deterministic(
        z_dim=cfg.latent_dim,
        e_dim=cfg.e_dim,
        enc_hidden_dim=cfg.enc_hidden_dim,
        slow_hidden_dim=cfg.slow_hidden_dim,
        fast_hidden_dim=cfg.fast_hidden_dim,
        K=cfg.K
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

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{cfg.epochs}")

        for z in pbar:
            z = z.to(cfg.device)  # [B, D, T]

            optimizer.zero_grad()

            outputs = model(z, teacher_forcing=True)
            loss, recon, kl = vae_loss_mse(z, outputs, beta=cfg.beta)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon.item()
            total_kl += kl.item()

            pbar.set_postfix({
                "loss": loss.item(),
                "recon": recon.item(),
                "kl": kl.item()
            })

        avg_loss = total_loss / len(loader)
        avg_recon = total_recon / len(loader)
        avg_kl = total_kl / len(loader)

        print(
            f"[Epoch {epoch+1}] "
            f"Loss: {avg_loss:.4f} | "
            f"Recon: {avg_recon:.4f} | "
            f"KL: {avg_kl:.4f}"
        )


        if (epoch + 1) % 25 == 0:
            for z in loader:
                z_hat, _, _ = model(z, teacher_forcing=True)
                break  # take first batch only
            writer.add_scalar("Loss/Train", avg_loss, epoch+1)
            writer.add_scalar("Recon/Train", avg_recon, epoch+1)
            writer.add_scalar("KL/Train", avg_kl, epoch+1)
            # output_latent = model.reparameterize(outputs_recon[0], outputs_recon[1])
            recon = codec.decode(z_hat)
            orig = codec.decode(z)
            writer.add_audio("Recon/GenLatent", recon, sample_rate=44100, global_step=epoch+1)
            writer.add_audio("Recon/Orig", orig, sample_rate=44100, global_step=epoch+1)
            torchaudio.save(f"./reconstructions/recon_{epoch+1}.wav", recon, sample_rate=44100)
            torchaudio.save(f"./reconstructions/orig_{epoch+1}.wav", orig, sample_rate=44100)
            # torch.save(
            #     model.state_dict(),
            #     os.path.join(save_dir, f"model_epoch_{epoch+1}.pt")
            # )

if __name__ == "__main__":


    data_dir = Path("/Users/adees/Code/multi-scale-rnn-vae/test_audio/latents")
    ext = ".pt"
    train_files = [str(p) for p in sorted(data_dir.glob(f"*{ext}"))]

    if not train_files:
        raise FileNotFoundError(f"No '{ext}' files found in {data_dir}")
    
    save_dir = "./checkpoints"

    cfg = Config()
    train(cfg, train_files, save_dir)
