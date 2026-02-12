import librosa
import soundfile as sf
import numpy as np
import os
import torchaudio
import torch

audio_path_target = "/Users/adees/Code/neural_granular_synthesis/datasets/ESC-50_SeaWaves/audio/full/1-28135-B-11.wav"
# audio_path_target = "/Users/adees/Code/neural_granular_synthesis/datasets/seaWaveTests/seaWaves5/audio.wav"
# audio_path = "/Users/adees/Code/neural_granular_synthesis/datasets/thunderstorm_noisebandnet/audio/audio.wav"
# audio_path = "/Users/adees/Code/neural_granular_synthesis/datasets/rain/audio.wav"
# audio_path_target = "/Users/adees/Code/neural_granular_synthesis/datasets/ES_Eval_Set/rain/segment.wav"
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

dis = max_pairwise_distance(latent, target_latent)
print(f"Max pairwise distance: {dis[0]} (latent frame {dis[1][0]}, target frame {dis[1][1]})")

latent_shift = shift_latent(latent, amount=0.2)
latent_toward = shift_towards_target(latent, target_latent, amount=5.0, normalize=True, use_mean=True)
# latent has shape (batch_size/audio_channels, dim (64), sequence_length)
print(f"Encoded latent representation with shape: {latent.shape}")

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