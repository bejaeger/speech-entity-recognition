"""
Torch Dataset for the dummy data in the data/ folder.
Also does some basic preprocessing of the audio data to extract mel features.
"""

from dataclasses import dataclass
from pathlib import Path
from resource.text_data import TRAIN_DATA

import torch
from torch.utils.data import Dataset

from src.supporting_code.audio_feature_extractor import AudioFeatureExtractor


@dataclass
class SpeechSample:
    transcription: str
    context: list[str]
    mel_features_BTC: torch.Tensor


class SpeechDataset(Dataset):
    """Torch Dataset for dummy speech entity recognition data.

    Args:
        data_folder (Path): Path to the data folder containing audio files.
    """

    def __init__(self, data_folder: Path = Path("resource")) -> None:
        self.data_folder = data_folder
        self.data = TRAIN_DATA
        self.extractor = AudioFeatureExtractor()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> SpeechSample:
        audio_file, transcription, context = self.data[idx]
        audio_file_path = self.data_folder / audio_file
        mel_features_BTC = self.extractor.compute_mel_spectrogram(audio_file_path)

        return SpeechSample(
            transcription=transcription,
            context=context,
            mel_features_BTC=mel_features_BTC,
        )

    def collect_vocabulary(self) -> set[str]:
        """
        Collects all unique vocabulary words from the dataset's context and transcription fields.

        Returns:
            set[str]: Set of all unique lowercased words split by whitespace.
        """
        vocab = set()
        for i in range(len(self)):
            sample = self[i]
            for context_item in sample.context:
                vocab.update(context_item.lower().split())
            vocab.update(sample.transcription.lower().split())
        return vocab
