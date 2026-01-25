import librosa
import soundfile as sf
import numpy as np
import os
import torchaudio

# audio_path = "/Users/adees/Code/neural_granular_synthesis/datasets/ESC-50_SeaWaves/audio/full/1-28135-B-11.wav"
# audio_path = "/Users/adees/Code/neural_granular_synthesis/datasets/seaWaveTests/seaWaves5/audio.wav"
# audio_path = "/Users/adees/Code/neural_granular_synthesis/datasets/thunderstorm_noisebandnet/audio/audio.wav"
audio_path = "/Users/adees/Code/neural_granular_synthesis/datasets/rain/audio.wav"
wv, sr = librosa.load(audio_path, sr=44100)

print(f"Loaded audio with shape: {wv.shape}, sample rate: {sr}")

from music2latent import EncoderDecoder
encdec = EncoderDecoder()

latent = encdec.encode(wv)
# latent has shape (batch_size/audio_channels, dim (64), sequence_length)
print(f"Encoded latent representation with shape: {latent.shape}")

wv_rec = encdec.decode(latent)
print(f"Decoded waveform with shape: {wv_rec.shape}")

out_path = "/Users/adees/Code/music2latent_tests/decoded.wav"

# Ensure shape is (samples,) or (samples, channels)
# wv_out = np.asarray(wv_rec, dtype="float32")

# print(wv_rec)
# if wv_out.ndim == 2 and wv_out.shape[0] <= 2:
#     wv_out = wv_out.T

torchaudio.save(out_path, wv_rec, sr)

# sf.write(out_path, wv_out, sr)
print(f"Saved decoded waveform to: {out_path}")