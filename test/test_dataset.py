"""
Usage:
    pytest test/test_dataset.py
"""

import torch
from pathlib import Path
from src.supporting_code.dataset import SpeechDataset, SpeechSample


def test_dummy_speech_dataset_basic():
    """Basic test to check DummySpeechDataset returns a SpeechSample with correct types."""
    dataset = SpeechDataset(data_folder=Path("resource"))
    assert len(dataset) > 1
    sample = dataset[0]
    assert isinstance(sample, SpeechSample)
    assert isinstance(sample.transcription, str)
    assert isinstance(sample.context, list)
    assert isinstance(sample.mel_features_BTC, torch.Tensor)
    assert sample.mel_features_BTC.shape[2] == 80
