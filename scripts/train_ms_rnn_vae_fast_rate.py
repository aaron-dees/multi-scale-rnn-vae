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
from models.fast_model import ConditionalFastARGaussian
from models.loss_functions import fast_residual_nll_loss 
from scripts.fast_config import FastConfig
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

    slow_model = MultiScaleRNNVAE_slowRate(
        z_dim=cfg.latent_dim,
        e_dim=cfg.e_dim,
        enc_hidden_dim=cfg.enc_hidden_dim,
        slow_hidden_dim=cfg.slow_hidden_dim,
        context_window_size=cfg.context_window_size,
        context_window_hidden_dim=cfg.context_window_hidden_dim,
        context_encoder_pooling_rate=cfg.context_encoder_pooling_rate,
        target_window_size=cfg.target_window_size
    ).to(cfg.device).eval()

    fast_model = ConditionalFastARGaussian(
        z_dim=cfg.latent_dim,
        # only use e here if using e in training
        # e_dim=cfg.e_dim,
        hidden_dim=cfg.fast_hidden_dim,
        # num_layers=2
    )

    optimizer = Adam(fast_model.parameters(), lr=cfg.lr)

    return fast_model, slow_model, optimizer, loader

def train(cfg, latent_dir, save_dir):

    writer = SummaryWriter(log_dir="./runs/test_run")

    os.makedirs(save_dir, exist_ok=True)

    fast_model, slow_model, optimizer, loader = build_training_objects(cfg, latent_dir)

    fast_model.train()

    # load slow model
    state = torch.load(cfg.slow_model_checkpoint_path, map_location=cfg.device)

    # Accept either raw state_dict or a checkpoint dict containing 'model_state_dict'
    if isinstance(state, dict) and "model_state_dict" in state:
        state_dict = state["model_state_dict"]
    else:
        state_dict = state

    try:
        slow_model.load_state_dict(state_dict)
        print(f"Loaded checkpoint (strict) from {cfg.slow_model_checkpoint_path}")
    except RuntimeError:
        res = slow_model.load_state_dict(state_dict, strict=False)
        print(f"Loaded checkpoint (non-strict) from {cfg.slow_model_checkpoint_path}")
        print("Missing keys:", res.missing_keys)
        print("Unexpected keys:", res.unexpected_keys)
    
    slow_model.eval()

    codec = EncoderDecoder()

    for epoch in range(cfg.epochs):
        total_loss = 0.0

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{cfg.epochs}")

        for z, z_slow in pbar:
            z_slow_target = z_slow[:, :, cfg.context_window_size:]  # remove context from main input
            z_slow_ctx = z_slow[:, :, :cfg.context_window_size]

            z_fast = z - z_slow
            z_fast_target = z_fast[:, :, cfg.context_window_size:]  # remove context from main input
            z_fast_ctx = z_fast[:, :, :cfg.context_window_size]

            optimizer.zero_grad()

            z_hat_slow, _, _ = slow_model(z_slow_ctx)

            decay_epochs = 500
            teacher_forcing_prob = max(0.3, 1.0 - epoch / decay_epochs)
            teacher_forcing_prob = 1.0

            # Should i use predicted s_hat slow here or ground truth?
            z_hat_fast, mu_fast, logvar_fast = fast_model(z_hat_slow, z_fast_target, teacher_forcing_prob=teacher_forcing_prob, z0=None, sample_in_forward=False, temperature=1.0)

            loss, logs = fast_residual_nll_loss(z_fast_target, mu_fast, logvar_fast)

            loss.backward()

            # print(img)

            nn.utils.clip_grad_norm_(fast_model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

            pbar.set_postfix({
                "loss": loss.item(),
            })

        avg_loss = total_loss / len(loader)

        print(
            f"[Epoch {epoch+1}] "
            f"Loss: {avg_loss:.4f} | "
        )


        # if (epoch + 1) % 5 == 0:
        #     writer.add_scalar("Loss/Train", avg_loss, epoch+1)

        # if (epoch + 1) % 25 == 0:
        #     for _, z_val_slow in loader:
        #         z_slow_target = z_val_slow[:, :, cfg.context_window_size:]  # remove context from main input
        #         z_slow_ctx = z_val_slow[:, :, :cfg.context_window_size]
        #         z_hat_val, _, _ = model(z_slow_ctx)
        #         break  # take first batch only

        #     plot_latent_pca(z_slow_target[0:1].detach(), z_hat_val[0:1].detach(), epoch+1)
        #     if (epoch + 1) % 100 == 0:
        #         recon = codec.decode(z_hat_val)
        #         orig = codec.decode(z_slow_target)
        #         for i in range(recon.shape[0]):
        #             writer.add_audio(f"Recon/GenLatent_{i}", recon[i:i+1], sample_rate=44100, global_step=epoch+1)
        #             writer.add_audio(f"Recon/Orig_{i}", orig[i:i+1], sample_rate=44100, global_step=epoch+1)
        #             torchaudio.save(f"./reconstructions/recon_{i}_{epoch+1}.wav", recon[i:i+1], sample_rate=44100)
        #             torchaudio.save(f"./reconstructions/orig_{i}_{epoch+1}.wav", orig[i:i+1], sample_rate=44100)
        # if (epoch + 1) % 500 == 0:
        #     torch.save(
        #         model.state_dict(),
        #         os.path.join(save_dir, f"model_epoch_{epoch+1}.pt")
        #     )

if __name__ == "__main__":


    data_dir = Path("/Users/adees/Code/multi-scale-rnn-vae/test_audio/latents")
    ext = ".pt"
    train_files = [str(p) for p in sorted(data_dir.glob(f"*{ext}"))]

    if not train_files:
        raise FileNotFoundError(f"No '{ext}' files found in {data_dir}")
    
    save_dir = "./checkpoints"

    cfg = FastConfig()
    train(cfg, train_files, save_dir)
