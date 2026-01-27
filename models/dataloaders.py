import torch
from torch.utils.data import Dataset
import random

class LatentTextureDataset(Dataset):
    def __init__(self, latent_files, window_size):
        """
        latent_files: list of paths to .pt tensors, each [T_i, D]
        window_size: number of timesteps (e.g. 100 for 10s)
        """
        self.latent_files = latent_files
        self.window_size = window_size

        # Load all latents into memory (recommended if they fit)
        self.latents = [torch.load(f) for f in latent_files]

        # Filter out clips that are too short
        self.latents = [
            x for x in self.latents if x.shape[-1] >= window_size
        ]

    def __len__(self):
        # One sample per clip per epoch
        return len(self.latents)

    def __getitem__(self, idx):
        x = self.latents[idx]      # [1, D, T]
        T = x.shape[-1]

        start = random.randint(0, T - self.window_size)
        window = x[:,start:start + self.window_size]

        return window   # [window_size, D]
