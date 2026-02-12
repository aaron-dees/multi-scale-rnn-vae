import librosa
import soundfile as sf
import numpy as np
import os
import torchaudio
import torch

audio_path_target = "/Users/adees/Code/neural_granular_synthesis/datasets/ESC-50_SeaWaves/audio/full/1-28135-B-11.wav"
# audio_path = "/Users/adees/Code/neural_granular_synthesis/datasets/seaWaveTests/seaWaves5/audio.wav"
# audio_path = "/Users/adees/Code/neural_granular_synthesis/datasets/thunderstorm_noisebandnet/audio/audio.wav"
# audio_path = "/Users/adees/Code/neural_granular_synthesis/datasets/rain/audio.wav"
# audio_path_target = "/Users/adees/Code/neural_granular_synthesis/datasets/ES_Eval_Set/rain/segment.wav"
# audio_path_target = "/Users/adees/Downloads/drone2.wav"
audio_path = "/Users/adees/Downloads/test.wav"
wv, sr = librosa.load(audio_path, sr=44100)
wv_target, sr_target = librosa.load(audio_path_target, sr=44100)

print(f"Loaded audio with shape: {wv.shape}, sample rate: {sr}")

from music2latent import EncoderDecoder
encdec = EncoderDecoder()

def shift_towards_target(latent, target_latent, amount=0.1, normalize=True, use_mean=True):
    """
    Shift `latent` toward `target_latent` by adding a scaled direction vector to every frame.

    - latent, target_latent: np.ndarray or torch.Tensor with shape (channels, dim, seq_len)
    - amount: scalar multiplier (how far to move toward target). If normalize=True this is in units of
      the direction vector (1.0 = full mean->mean).
    - normalize: if True, normalize the direction to unit length per channel before scaling.
    - use_mean: if True use the mean over time as the summary vector; otherwise uses first frame.
    """
    is_torch = torch.is_tensor(latent) or torch.is_tensor(target_latent)

    # convert to numpy for arithmetic
    lat_np = latent.detach().cpu().numpy() if torch.is_tensor(latent) else np.asarray(latent)
    tgt_np = target_latent.detach().cpu().numpy() if torch.is_tensor(target_latent) else np.asarray(target_latent)

    if lat_np.ndim != 3 or tgt_np.ndim != 3:
        raise ValueError("Expected latents with shape (channels, dim, seq_len)")

    ch, dim, _ = lat_np.shape

    if use_mean:
        src_vec = lat_np.mean(axis=2, keepdims=True)   # (ch, dim, 1)
        tgt_vec = tgt_np.mean(axis=2, keepdims=True)
    else:
        src_vec = lat_np[:, :, 0:1]
        tgt_vec = tgt_np[:, :, 0:1]

    direction = tgt_vec - src_vec  # (ch, dim, 1)

    if normalize:
        # normalize per-channel to avoid scaling issues across channels
        flat = direction.reshape(ch, -1)
        norms = np.linalg.norm(flat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        direction = (direction.reshape(ch, -1) / norms).reshape(ch, dim, 1)

    shift = float(amount) * direction
    shifted_np = lat_np + shift  # add same shift to every time/frame -> preserves relative frame differences

    if is_torch:
        # preserve device & dtype if possible
        ref = latent if torch.is_tensor(latent) else target_latent
        return torch.from_numpy(shifted_np).to(ref.device).type(ref.dtype)
    return shifted_np

def shift_latent(latent, amount=0.1, direction=None, normalize=True):
    """Return latent + amount*direction.

    - latent: np.ndarray or torch.Tensor with shape (channels, dim, seq_len)
    - amount: scalar multiplier for the direction vector
    - direction: optional array/tensor shaped (channels, dim) or (channels, dim, 1). 
      If None, uses a unit vector along the first latent dimension.
    - normalize: if True, normalize the direction to unit norm before scaling
    """
    is_torch = torch.is_tensor(latent)
    if is_torch:
        lat_np = latent.detach().cpu().numpy()
    else:
        lat_np = np.asarray(latent)

    ch, dim, seq = lat_np.shape

    if direction is None:
        dir_np = np.zeros((ch, dim, 1), dtype=lat_np.dtype)
        # shift along the first latent dimension (you can change index 0)
        dir_np[:, 0, 0] = 1.0
    else:
        dir_np = np.asarray(direction)
        # allow (ch, dim) -> (ch, dim, 1)
        if dir_np.ndim == 2:
            dir_np = dir_np[:, :, None]
        # broadcast if user passed (dim,) or (1, dim, 1)
        if dir_np.shape[0] != ch:
            dir_np = np.broadcast_to(dir_np, (ch, dir_np.shape[1], 1))

    # normalize direction if requested
    if normalize:
        norm = np.linalg.norm(dir_np.reshape(-1))
        if norm > 0:
            dir_np = dir_np / norm

    shifted_np = lat_np + float(amount) * dir_np

    if is_torch:
        shifted = torch.from_numpy(shifted_np).to(latent.device).type(latent.dtype)
    else:
        shifted = shifted_np
    return shifted

def _to_numpy(lat):
    if torch.is_tensor(lat):
        return lat.detach().cpu().numpy()
    return np.asarray(lat)

def max_pairwise_distance(latent, target):
    """Max Euclidean distance between any frame in latent and any frame in target.
    Returns (max_distance, (idx_lat, idx_target))."""
    A = _to_numpy(latent)
    B = _to_numpy(target)
    if A.ndim != 3 or B.ndim != 3:
        raise ValueError("Expected shape (channels, dim, seq_len)")
    # frames -> (n_frames, channels*dim)
    A_f = A.transpose(2,0,1).reshape(A.shape[2], -1).astype(np.float64)
    B_f = B.transpose(2,0,1).reshape(B.shape[2], -1).astype(np.float64)
    # pairwise squared distances: ||a||^2 + ||b||^2 - 2 a.b
    aa = (A_f**2).sum(axis=1)[:, None]
    bb = (B_f**2).sum(axis=1)[None, :]
    d2 = aa + bb - 2.0 * (A_f @ B_f.T)
    # numerical safety
    d2 = np.maximum(d2, 0.0)
    idx = np.unravel_index(np.argmax(d2), d2.shape)
    return float(np.sqrt(d2[idx])), idx

latent = encdec.encode(wv)
target_latent = encdec.encode(wv_target)

print("Latent mean/std:", latent.mean().item(), latent.std().item())
print("Latent mean/std:", target_latent.mean().item(), target_latent.std().item())

dis = max_pairwise_distance(latent, target_latent)
print(f"Max pairwise distance: {dis[0]} (latent frame {dis[1][0]}, target frame {dis[1][1]})")

latent_shift = shift_latent(latent, amount=0.2)
latent_toward = shift_towards_target(latent, target_latent, amount=15.0, normalize=True, use_mean=True)
# latent_toward = shift_towards_target(latent, target_latent, amount=15.0, normalize=True, use_mean=True)
# latent has shape (batch_size/audio_channels, dim (64), sequence_length)
print(f"Encoded latent representation with shape: {latent.shape}")



# PCA
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
# Use first batch of validation for plotting
target_seq = latent.permute(0,2,1)[0].cpu().numpy()
pred_seq = target_latent.permute(0,2,1)[0].cpu().numpy()
shifted_seq = latent_toward.permute(0,2,1)[0].cpu().numpy()
all_seq = np.vstack([target_seq, pred_seq, shifted_seq])
pca = PCA(n_components=3)
Zp = pca.fit_transform(all_seq)
Zp_target = Zp[:target_seq.shape[0]]
Zp_pred = Zp[target_seq.shape[0]:target_seq.shape[0]+pred_seq.shape[0]]
Zp_shifted = Zp[target_seq.shape[0]+pred_seq.shape[0]:]
# compute per-sequence mean vectors (matching how PCA was fit: features are the latent dim)
src_vec = latent.mean(dim=2)[0].cpu().numpy()          # source sequence mean, shape (dim,)
tgt_vec = target_latent.mean(dim=2)[0].cpu().numpy()   # target sequence mean, shape (dim,)

# transform to PCA coords (reshape to (1, dim) for transform)
Zp_src = pca.transform(src_vec[None, :])   # shape (1, n_components)
Zp_tgt = pca.transform(tgt_vec[None, :])
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(Zp_target[:, 0], Zp_target[:, 1], '-o', markersize=3, alpha=0.6, label='Target')
plt.plot(Zp_pred[:, 0], Zp_pred[:, 1], '-o', markersize=3, alpha=0.6, label='Prediction')
plt.plot(Zp_shifted[:, 0], Zp_shifted[:, 1], '-o', markersize=3, alpha=0.6, label='Shifted')
plt.scatter(Zp_src[0, 0], Zp_src[0, 1], marker='X', s=100, c='black', label='Source mean')
plt.scatter(Zp_tgt[0, 0], Zp_tgt[0, 1], marker='D', s=100, c='magenta', label='Target mean')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Latent Trajectories (Top 2 PCA)')
plt.legend()
plt.grid(True)
from mpl_toolkits.mplot3d import Axes3D
ax = plt.subplot(1, 2, 2, projection='3d')
ax.plot(Zp_target[:, 0], Zp_target[:, 1], Zp_target[:, 2], '-o', markersize=3, alpha=0.6, label='Target')
ax.plot(Zp_pred[:, 0], Zp_pred[:, 1], Zp_pred[:, 2], '-o', markersize=3, alpha=0.6, label='Prediction')
ax.plot(Zp_shifted[:, 0], Zp_shifted[:, 1], Zp_shifted[:, 2], '-o', markersize=3, alpha=0.6, label='Shifted')
ax.scatter(Zp_src[0, 0], Zp_src[0, 1], Zp_src[0, 2], marker='X', s=80, c='black', label='Source mean')
ax.scatter(Zp_tgt[0, 0], Zp_tgt[0, 1], Zp_tgt[0, 2], marker='D', s=80, c='magenta', label='Target mean')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.set_title('Latent Trajectories (Top 3 PCA)')
ax.legend()
plt.tight_layout()
plt.savefig(f"./latent_predictor_pca.png")
plt.close()

wv_rec = encdec.decode(latent)
wv_rec_shift = encdec.decode(latent_shift)
wv_rec_toward = encdec.decode(latent_toward)
wv_rec_target = encdec.decode(target_latent)

wv_rec = wv_rec_target

print(f"Decoded waveform with shape: {wv_rec.shape}")
try:
    import matplotlib.pyplot as plt
    import librosa.display

    # convert wv_rec to 1D numpy mono signal
    if torch.is_tensor(wv_rec):
        wv_np = wv_rec.detach().cpu().numpy()
    else:
        wv_np = np.asarray(wv_rec)

    if wv_np.ndim == 2:
        # (channels, time) -> mix to mono
        if wv_np.shape[0] <= 2:
            mono = wv_np.mean(axis=0)
        else:
            mono = wv_np.reshape(-1)
    else:
        mono = wv_np

    mono = mono.astype("float32")

    # compute mel-spectrogram
    n_fft = 2048
    hop_length = 512
    n_mels = 128
    S = np.abs(librosa.stft(mono, n_fft=n_fft, hop_length=hop_length))**2
    S_db = librosa.power_to_db(S, ref=np.max)
    cent = librosa.feature.spectral_centroid(y=mono, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
    rolloff = librosa.feature.spectral_rolloff(y=mono, sr=sr, n_fft=n_fft, hop_length=hop_length, roll_percent=0.85)[0]
    band = librosa.feature.spectral_bandwidth(y=mono, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
    flat = librosa.feature.spectral_flatness(y=mono, n_fft=n_fft, hop_length=hop_length)[0]
    rms = librosa.feature.rms(y=mono, frame_length=n_fft, hop_length=hop_length)[0]
    zcr = librosa.feature.zero_crossing_rate(y=mono, frame_length=n_fft, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(cent)), sr=sr, hop_length=hop_length)
    def norm(x): return (x - x.min()) / (x.max() - x.min() + 1e-9)
    cent_n = norm(cent)
    roll_n = norm(rolloff)
    band_n = norm(band)
    flat_n = norm(flat)
    rms_n = norm(rms)
    zcr_n = norm(zcr)
    fig, ax = plt.subplots(figsize=(24,8))
    img = librosa.display.specshow(S_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='hz', cmap='magma', ax=ax)
    ax2 = ax.twinx()
    ax2.plot(times, cent_n, color='cyan', alpha=0.8, label='centroid (norm)')
    ax2.plot(times, roll_n, color='lime', alpha=0.8, label='rolloff (norm)')
    ax2.plot(times, band_n, color='yellow', alpha=0.7, label='bandwidth (norm)')
    ax2.plot(times, flat_n, color='magenta', alpha=0.7, label='flatness (norm)')
    ax2.plot(times, rms_n, color='orange', alpha=0.6, label='RMS (norm)')
    ax2.plot(times, zcr_n, color='silver', alpha=0.6, label='ZCR (norm)')
    ax2.legend(loc='upper right')
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    plt.title("Power spectrogram + texture features")
    # plt.tight_layout()
    plt.savefig('spec_with_features.png', dpi=150)
    plt.close(fig)

    # print(f"Saved spectrogram to: {img_path}")
except Exception as e:
    print("Failed to save spectrogram:", e)

wv_rec = encdec.decode(latent)
wv_rec_shift = encdec.decode(latent_shift)
wv_rec_toward = encdec.decode(latent_toward)
print(f"Decoded waveform with shape: {wv_rec.shape}")

out_path = "/Users/adees/Code/multi-scale-rnn-vae/decoded.wav"
out_path_shift = "/Users/adees/Code/multi-scale-rnn-vae/decoded_shift.wav"
out_path_toward = "/Users/adees/Code/multi-scale-rnn-vae/decoded_toward.wav"
out_path_target = "/Users/adees/Code/multi-scale-rnn-vae/target.wav"

# Ensure shape is (samples,) or (samples, channels)
# wv_out = np.asarray(wv_rec, dtype="float32")

# print(wv_rec)
# if wv_out.ndim == 2 and wv_out.shape[0] <= 2:
#     wv_out = wv_out.T

torchaudio.save(out_path, wv_rec, sr)
torchaudio.save(out_path_shift, wv_rec_shift, sr)
torchaudio.save(out_path_toward, wv_rec_toward, sr)
torchaudio.save(out_path_target, torch.from_numpy(wv_target).unsqueeze(0), sr)

# sf.write(out_path, wv_out, sr)
print(f"Saved decoded waveform to: {out_path}")