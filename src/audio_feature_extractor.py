"""
Audio feature extraction utilities.

Usage example:
    extractor = AudioFeatureExtractor()
    mel_spec = extractor.compute_mel_spectrogram(audio_file_path)
"""

from pathlib import Path
from dataclasses import dataclass
import torch
import torchaudio


@dataclass
class AudioFeatureExtractor:
    """
    Extracts audio features (STFT and mel spectrogram) from a waveform tensor.

    Args:
        sample_rate: The sample rate of the input waveform.
        n_fft: FFT window size for STFT.
        hop_length: Number of samples between successive frames.
        n_mels: Number of mel filterbanks for mel spectrogram.
    """

    n_fft: int = 1024
    hop_length: int = 512
    n_mels: int = 80

    def compute_mel_spectrogram(self, audio_file_path: Path) -> torch.Tensor:
        """
        Computes the mel spectrogram of the input waveform.

        Args:
            audio_file_path: Path to the audio file.
        Returns:
            Mel spectrogram tensor (1, time, n_mels).
        """
        wave, sample_rate = torchaudio.load(str(audio_file_path))

        if wave.dim() == 1:
            wave = wave.unsqueeze(0)
        mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
        )
        mel_spec_1CT = mel_spectrogram_transform(wave)
        mel_spec_1TC = mel_spec_1CT.transpose(1, 2)
        # Standardize along the time axis for each mel bin
        mean = mel_spec_1TC.mean(dim=1, keepdim=True)
        std = mel_spec_1TC.std(dim=1, keepdim=True)
        mel_spec_1TC = (mel_spec_1TC - mean) / (std + 1e-8)
        return mel_spec_1TC
