import torch
import torchaudio
from pathlib import Path
from music2latent import EncoderDecoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_audio(path, target_sr):
    wav, sr = torchaudio.load(path)   # [C, N]

    # is this the correct way to mix to mono??
    wav = wav.mean(dim=0, keepdim=True)  # mono

    if sr != target_sr:
        wav = torchaudio.functional.resample(
            wav, orig_freq=sr, new_freq=target_sr
        )

    return wav.squeeze(0)  # [N]

@torch.no_grad()
def encode_audio_to_latents(
    encoder,
    wav,
    sr,
    frame_ms=90,
    batch_size=64,
    device=DEVICE
):
    wav = wav.to(device)

    # for i in range(0, wav.shape[0], batch_size):
    z = encoder.encode(wav)   # [1, 64, T]
    # latents.append(z.cpu())

    return z[0,:,:]  # [64, T]

def convert_audio_folder_to_latents(
    audio_dir,
    output_dir,
    encoder,
    sr=44100,
    device=DEVICE
):
    audio_dir = Path(audio_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # encoder = encoder.to(device)
    # encoder.eval()

    for audio_path in audio_dir.glob("*.wav"):
        print(f"Processing {audio_path.name}")

        wav = load_audio(audio_path, sr)
        latents = encode_audio_to_latents(
            encoder, wav, sr, device=device
        )

        out_path = output_dir / (audio_path.stem + ".pt")
        torch.save(latents, out_path)

        print(f"Saved {latents.shape} → {out_path}")



codec = EncoderDecoder()

convert_audio_folder_to_latents(
    audio_dir="../test_audio/audio/",
    output_dir="../test_audio/latents/",
    encoder=codec,
    sr=44100,
    device=DEVICE
)