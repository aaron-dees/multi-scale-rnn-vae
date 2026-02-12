import librosa
import soundfile as sf
import numpy as np
import os
import torchaudio
import torch

def interpolate_endpoints(latent_a, latent_b, n_steps=100):
	"""Return a new latent sequence that linearly interpolates between the
	first and last latent vectors from `latent`.

	Args:
		latent: array-like with shape (channels, dim, seq_len)
		n_steps: number of steps including endpoints

	Returns:
		np.ndarray of shape (channels, dim, n_steps)
	"""
	lat_a = np.asarray(latent_a)
	lat_b = np.asarray(latent_b)
	if lat_a.ndim != 3 or lat_b.ndim != 3:
		raise ValueError(f"Expected latent shape (channels, dim, seq_len), got {lat_a.shape} and {lat_b.shape}")

	start = lat_a[:,:, :40]
	end = lat_b[:,:, -40:]


	channels, dim, seq_len = lat_a.shape
	first = start[:, :, -1:]
	last = end[:, :, 0:1]
	# first = torch.randn((channels, dim, 1))*5
	# last = torch.randn((channels, dim, 1))

	frames = []
	for t in np.linspace(0.0, 1.0, n_steps):
		frames.append((1.0 - t) * first + t * last)
	frames = np.concatenate(frames, axis=2)	
	frames = np.concatenate([start, frames, end], axis=2)
	return frames

audio_path_a = "/Users/adees/Code/neural_granular_synthesis/datasets/ESC-50_SeaWaves/audio/full/1-28135-B-11.wav"
# audio_path_b = "/Users/adees/Code/neural_granular_synthesis/datasets/seaWaveTests/seaWaves5/audio.wav"
audio_path_b = "/Users/adees/Downloads/test.wav"
# audio_path_a = "/Users/adees/Code/neural_granular_synthesis/datasets/thunderstorm_noisebandnet/audio/audio.wav"
# audio_path_b = "/Users/adees/Code/neural_granular_synthesis/datasets/rain/audio.wav"
# audio_path_a = "/Users/adees/Code/neural_granular_synthesis/datasets/rain/audio.wav"
# audio_path_b = "/Users/adees/Code/neural_granular_synthesis/datasets/ES_Eval_Set/rain/segment.wav"
wv_a, sr = librosa.load(audio_path_a, sr=44100)
wv_b, sr = librosa.load(audio_path_b, sr=44100)

print(f"Loaded audio with shape: {wv_a.shape}, sample rate: {sr}")

from music2latent import EncoderDecoder
encdec = EncoderDecoder()

latent_a = encdec.encode(wv_a)
latent_b = encdec.encode(wv_b)
# latent has shape (batch_size/audio_channels, dim (64), sequence_length)
print(f"Encoded latent representation with shape: {latent_a.shape}")

# wv_rec = encdec.decode(latent_a)
# print(f"Decoded waveform with shape: {wv_rec.shape}")




# Example: create 100-step interpolation between first and last latent vectors
interp_latents = interpolate_endpoints(latent_a, latent_b, n_steps=40)
print(f"Created interp_latents with shape: {interp_latents.shape}")
interp_latents = torch.from_numpy(interp_latents.astype("float32"))

wv_rec = encdec.decode(interp_latents)

out_path = "/Users/adees/Code/multi-scale-rnn-vae/interpolate.wav"

# Ensure shape is (samples,) or (samples, channels)
# wv_out = np.asarray(wv_rec, dtype="float32")

# print(wv_rec)
# if wv_out.ndim == 2 and wv_out.shape[0] <= 2:
#     wv_out = wv_out.T

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
    

# Phase 
import numpy as np
import matplotlib.pyplot as plt
import librosa.display

# after you have `mono`, sr, n_fft, hop_length defined:
D = librosa.stft(mono, n_fft=n_fft, hop_length=hop_length)   # complex-valued STFT (freq_bins, time_frames)

# 1) wrapped phase (range -pi .. +pi)
phase = np.angle(D)  # shape (freq_bins, time_frames)

# 2) unwrap phase along time axis so phase evolution is continuous
phase_unwrapped = np.unwrap(phase, axis=1)

# 3) instantaneous frequency (Hz) from phase derivative over time
# formula: inst_freq = d(phase)/dt / (2*pi) ; with frames separated by hop_length samples:
phase_diff = np.diff(phase_unwrapped, axis=1)                # shape (freq_bins, time_frames-1)
inst_freq = phase_diff * (sr / (2.0 * np.pi * hop_length))   # shape (freq_bins, time_frames-1)


# time vectors for plotting
times = librosa.frames_to_time(np.arange(phase.shape[1]), sr=sr, hop_length=hop_length)
times_if = librosa.frames_to_time(np.arange(inst_freq.shape[1]), sr=sr, hop_length=hop_length)
freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

# Save wrapped phase image
plt.figure(figsize=(10, 4))
librosa.display.specshow(phase, sr=sr, hop_length=hop_length, x_axis='time', y_axis='hz', cmap='twilight')
plt.title("Wrapped phase (radians)")
plt.colorbar(format='%+2.0f')
plt.tight_layout()
plt.savefig("phase_wrapped.png", dpi=150)
plt.close()

# Save unwrapped-phase (visualize growth) - convert to degrees for readability
plt.figure(figsize=(10, 4))
librosa.display.specshow(np.degrees(phase_unwrapped), sr=sr, hop_length=hop_length, x_axis='time', y_axis='hz', cmap='magma')
plt.title("Unwrapped phase (degrees)")
plt.colorbar()
plt.tight_layout()
plt.savefig("phase_unwrapped.png", dpi=150)
plt.close()

# Save instantaneous frequency spectrogram (Hz)
plt.figure(figsize=(10, 4))
# guard against NaNs/Infs and clip extremes for better visualization
inst_freq = np.nan_to_num(inst_freq, nan=0.0, posinf=sr/2.0, neginf=-sr/2.0)
# choose symmetric vmin/vmax from percentile to avoid outlier domination
abs_max = np.percentile(np.abs(inst_freq), 99.0)
vmin, vmax = -abs_max, abs_max

librosa.display.specshow(inst_freq, sr=sr, hop_length=hop_length,
                         x_axis='time', y_axis='hz', cmap='viridis',
                         vmin=vmin, vmax=vmax)
plt.title("Instantaneous frequency (Hz)")
plt.colorbar(format='%+2.0f Hz')
plt.tight_layout()
plt.savefig("inst_freq.png", dpi=150)
plt.close()


torchaudio.save(out_path, wv_rec, sr)

# sf.write(out_path, wv_out, sr)
print(f"Saved decoded waveform to: {out_path}")