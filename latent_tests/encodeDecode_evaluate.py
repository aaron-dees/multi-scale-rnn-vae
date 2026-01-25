import librosa
import soundfile as sf
import numpy as np
import os
import torchaudio
import torch
import auraloss
from frechet_audio_distance import FrechetAudioDistance
import torch.nn as nn
import glob
from typing import List, Tuple, Optional, Union

from torchaudio.transforms import Spectrogram, MelSpectrogram

def safe_log(x, eps=1e-7):
    return torch.log(x + eps)

class spectral_distances(nn.Module):
    def __init__(self,stft_scales=[2048, 1024, 512, 256, 128], mel_scales=[2048, 1024], spec_power=1, mel_dist=True, log_dist=0, sr=16000, device="cpu"):
        super(spectral_distances, self).__init__()
        self.stft_scales = stft_scales
        self.mel_scales = mel_scales
        self.mel_dist = mel_dist
        self.log_dist = log_dist
        T_spec = []
        for scale in stft_scales:
            T_spec.append(Spectrogram(n_fft=scale,hop_length=scale//4,window_fn=torch.hann_window,power=spec_power).to(device))
        self.T_spec = T_spec
        if mel_dist:
            # print("\n*** training with MelSpectrogram distance")
            T_mel = []
            for scale in mel_scales:
                T_mel.append(MelSpectrogram(n_fft=scale,hop_length=scale//4,window_fn=torch.hann_window,sample_rate=sr,f_min=50.,n_mels=scale//4,power=spec_power).to(device))
            self.T_mel = T_mel
    
    def forward(self,x_inp,x_tar):
        loss = 0
        n_scales = 0
        for i,scale in enumerate(self.stft_scales):
            S_inp,S_tar = self.T_spec[i](x_inp),self.T_spec[i](x_tar)
            stft_dist = (S_inp-S_tar).abs().mean()
            loss = loss+stft_dist
            n_scales += 1
            if self.log_dist>0:
                loss = loss+(safe_log(S_inp)-safe_log(S_tar)).abs().mean()*self.log_dist
                n_scales += self.log_dist
        if self.mel_dist:
            for i,scale in enumerate(self.mel_scales):
                M_inp,M_tar = self.T_mel[i](x_inp),self.T_mel[i](x_tar)
                mel_dist = (M_inp-M_tar).abs().mean()
                loss = loss+mel_dist
                n_scales += 1
                if self.log_dist>0:
                    loss = loss+(safe_log(M_inp)-safe_log(M_tar)).abs().mean()*self.log_dist
                    n_scales += self.log_dist
        return loss/n_scales

frechet = FrechetAudioDistance(
    model_name="vggish",
    # Do I need to resample these?
    sample_rate=16000,
    use_pca=False, 
    use_activation=False,
    verbose=False
)

def list_audio_files(folder: str, exts: Tuple[str, ...] = ('wav','mp3','flac','ogg','m4a')) -> List[str]:
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(folder, f'**/*.{ext}'), recursive=True))
    return sorted(files)

def load_audio_folder(folder: str,
                      sr: int = 44100,
                      mono: bool = True,
                      max_files: Optional[int] = None,
                      extensions: Tuple[str, ...] = ('wav','mp3','flac','ogg','m4a')
                      ) -> List[np.ndarray]:
    """
    Load all audio files in `folder` (recursively) and return a list of 1D numpy arrays.
    Each array has dtype float32 and is resampled to `sr`.
    """
    files = list_audio_files(folder, extensions)
    if max_files:
        files = files[:max_files]
    waves = []
    for p in files:
        y, _ = librosa.load(p, sr=sr, mono=mono)
        waves.append(y.astype('float32'))
    waves = np.stack(waves)
    waves = torch.from_numpy(waves)
    return waves, files

sr = 44100
# audio, file_list = load_audio_folder('/Users/adees/Code/neural_granular_synthesis/datasets/ESC-50_EvalSet/thunder', sr=sr)
audio, file_list = load_audio_folder('/Users/adees/Code/music2latent_tests/test_in', sr=sr)

from music2latent import EncoderDecoder
encdec = EncoderDecoder()

for i, src_path in enumerate(file_list):
    latent = encdec.encode(audio[i])  # unchanged
    wv_rec = encdec.decode(latent)

    base = os.path.splitext(os.path.basename(src_path))[0]
    # out_path = f"/Users/adees/Code/music2latent_tests/m2l_recon/thunder/{base}.wav"
    out_path = f"/Users/adees/Code/music2latent_tests/test_file.wav"

    torchaudio.save(out_path, wv_rec, sr)