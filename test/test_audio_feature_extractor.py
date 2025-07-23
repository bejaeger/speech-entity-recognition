"""
Usage:
    pytest test/test_audio_feature_extractor.py
"""

from pathlib import Path
import torch
from src.supporting_code.audio_feature_extractor import AudioFeatureExtractor

RESOURCE_FOLDER = Path("resource")
AUDIO_FILE_PATH = RESOURCE_FOLDER / "herr_kalinowski_ein_glas_wasser_getrunken.mp3"


def test_mel_spectrogram_output_type_and_shape():
    """
    Test that compute_mel_spectrogram returns a torch.Tensor of expected shape.
    """
    extractor = AudioFeatureExtractor()
    mel_spec_1TC = extractor.compute_mel_spectrogram(AUDIO_FILE_PATH)
    assert isinstance(mel_spec_1TC, torch.Tensor)

    assert mel_spec_1TC.dim() == 3
    assert mel_spec_1TC.shape[2] == extractor.N_MELS
