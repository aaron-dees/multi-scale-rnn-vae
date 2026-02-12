import librosa
import soundfile as sf
import numpy as np
import os
import torchaudio
import torch
import torch.nn.functional as F

# audio_path = "/Users/adees/Code/neural_granular_synthesis/datasets/ESC-50_SeaWaves/audio/full/1-28135-B-11.wav"
# audio_path = "../test_audio/audio/2.wav"
# audio_path = "/Users/adees/Code/neural_granular_synthesis/datasets/seaWaveTests/seaWaves5/audio.wav"
# audio_path = "/Users/adees/Code/neural_granular_synthesis/datasets/thunderstorm_noisebandnet/audio/audio.wav"
audio_path = "/Users/adees/Code/neural_granular_synthesis/datasets/rain/audio.wav"
# audio_path_target = "/Users/adees/Code/neural_granular_synthesis/datasets/ES_Eval_Set/rain/segment.wav"
# audio_path_target = "/Users/adees/Downloads/drone2.wav"
# audio_path = "/Users/adees/Downloads/test.wav"
wv, sr = librosa.load(audio_path, sr=44100)
print("Audio shape: ", wv.shape)

wv = wv[44100:44100*9]
# wv_target, sr_target = librosa.load(audio_path_target, sr=44100)

print(f"Loaded audio with shape: {wv.shape}, sample rate: {sr}")

from music2latent import EncoderDecoder
encdec = EncoderDecoder()

def lowpass_moving_average(z, kernel_size):
    """
    z: [B, D, T]
    kernel_size: int (e.g. 25, 50, 100)
    """
    B, D, T = z.shape

    # reshape to [B*D, 1, T] for conv1d
    z_ = z.reshape(B * D, 1, T)

    kernel = torch.ones(1, 1, kernel_size, device=z.device) / kernel_size
    padding = kernel_size // 2

    z_lp = F.conv1d(z_, kernel, padding=padding)

    # back to [B, D, T]
    z_lp = z_lp.reshape(B, D, T)

    return z_lp


latent = encdec.encode(wv)
decoded_latent = encdec.decode(latent)
torchaudio.save(f"./reconstructions/decoded.wav", decoded_latent, sample_rate=44100)
print(f"Encoded latent shape: {latent.shape}")
# latent = latent[:,:56]
# low_pass = lowpass_moving_average(latent, kernel_size=9)[:,:,10:]
low_pass = lowpass_moving_average(latent, kernel_size=9)
# latent = latent[:,:,10:]

low_pass = low_pass[:,:,14:-14]
latent = latent[:,:,14:-14]

# PCA
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
# Use first batch of validation for plotting
target_seq = latent.permute(0,2,1)[0].cpu().numpy()
low_pass_seq = low_pass.permute(0,2,1)[0].cpu().numpy()
all_seq = np.vstack([target_seq])
pca = PCA(n_components=3)
Zp_target = pca.fit_transform(all_seq)
Zp_lowpass = pca.transform(low_pass_seq)
# compute per-sequence mean vectors (matching how PCA was fit: features are the latent dim)

# transform to PCA coords (reshape to (1, dim) for transform)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(Zp_target[:, 0], Zp_target[:, 1], '-o', markersize=3, alpha=0.6, label='Target')
plt.plot(Zp_lowpass[:, 0], Zp_lowpass[:, 1], '-o', markersize=3, alpha=0.6, label='Low-pass')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Latent Trajectories (Top 2 PCA)')
plt.legend()
plt.grid(True)
from mpl_toolkits.mplot3d import Axes3D
ax = plt.subplot(1, 2, 2, projection='3d')
ax.plot(Zp_target[:, 0], Zp_target[:, 1], Zp_target[:, 2], '-o', markersize=3, alpha=0.6, label='Target')
ax.plot(Zp_lowpass[:, 0], Zp_lowpass[:, 1], Zp_lowpass[:, 2], '-o', markersize=3, alpha=0.6, label='Low-pass')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.set_title('Latent Trajectories (Top 3 PCA)')
ax.legend()
plt.tight_layout()
plt.savefig(f"./latent_predictor_pca.png")
plt.close()

recon = encdec.decode(latent)
low_pass_recon = encdec.decode(low_pass)

torchaudio.save(f"./reconstructions/orig.wav", recon, sample_rate=44100)
torchaudio.save(f"./reconstructions/low_pass.wav", low_pass_recon, sample_rate=44100)
torchaudio.save(f"./reconstructions/orig_wv.wav", torch.from_numpy(wv).unsqueeze(0), sample_rate=44100)