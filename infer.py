"""
Script to infer the transcription of an audio snippet using a trained transducer model.

Usage example:
    python infer.py --audio-file-path resource/herr_kalinowski_ein_glas_wasser_getrunken.mp3 \
        --context-texts "herr kalinowski" "frau berbel" [--device cuda]
"""

import argparse
from pathlib import Path
from resource.text_data import TEST_DATA, TRAIN_DATA

import torch

from src.audio_feature_extractor import AudioFeatureExtractor
from src.model import Transducer
from src.tokenizer import SimpleTokenizer

torch.manual_seed(42)
torch.cuda.manual_seed(42)


def collect_vocabulary() -> set[str]:
    """
    Collects all unique vocabulary words from both TRAIN_DATA and TEST_DATA.

    Returns:
        set[str]: Set of all unique lowercased words split by whitespace.
    """
    vocab = set()
    for dataset in (TRAIN_DATA, TEST_DATA):
        for _, transcription, context in dataset:
            vocab.update(transcription.lower().split())
            for context_item in context:
                vocab.update(context_item.lower().split())
    return vocab


def infer_transcription(
    model: Transducer,
    tokenizer: SimpleTokenizer,
    mel_features_BTC: torch.Tensor,
    context_texts: list[str],
    device: torch.device,
    max_length: int = 20,
) -> str:
    """
    Runs autoregressive greedy decoding to predict the transcription for the given audio and context.

    Args:
        model: The loaded Transducer model.
        tokenizer: The tokenizer instance.
        mel_features_BTC: Mel spectrogram tensor (1, T, C).
        context_texts: List of context strings.
        device: The torch device.
        max_length: Maximum length of the generated sequence.
    Returns:
        str: The predicted transcription.
    """
    model.eval()

    input_ids_context_BS = tokenizer.encode(" ".join(context_texts)).to(device)
    print(f"input_ids_context_BS: {input_ids_context_BS}")
    mel_features_BTC = mel_features_BTC.to(device)
    generated: list[int] = [tokenizer.sos_token_id]

    for _ in range(max_length):
        input_ids_transcription_BS = torch.tensor(
            generated,
            dtype=torch.long,
            device=device,
        )[None]
        with torch.no_grad():
            logits_SV = model(
                mel_features_BTC=mel_features_BTC,
                input_ids_transcription_BS=input_ids_transcription_BS,
                input_ids_context_BS=input_ids_context_BS,
            ).squeeze(0)
        next_token_logits = logits_SV[logits_SV.shape[0] - 1, :]
        next_token_id = int(next_token_logits.argmax().item())
        generated.append(next_token_id)

    return tokenizer.decode(generated)


def main(
    audio_file_path: Path,
    context_texts: list[str],
    max_num_tokens: int = 20,
    device: str | None = None,
) -> None:
    """
    Main inference logic.

    Args:
        audio_file_path: Path to the audio file to transcribe.
        context_texts: List of context strings.
        max_num_tokens: Maximum number of tokens to generate.
        device: Torch device string or None.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    vocab = collect_vocabulary()
    tokenizer = SimpleTokenizer(vocab=list(vocab))

    model = Transducer(
        vocab_size=len(tokenizer.vocab),
        max_seq_len=tokenizer.max_length,
        pad_token_id=tokenizer.pad_token_id,
    ).to(device)
    state = torch.load("transducer.pth", map_location=device)
    model.load_state_dict(state, strict=False)

    extractor = AudioFeatureExtractor()
    mel_features_BTC = extractor.compute_mel_spectrogram(audio_file_path)

    print(f"Context texts: {context_texts}")
    transcription = infer_transcription(
        model=model,
        tokenizer=tokenizer,
        mel_features_BTC=mel_features_BTC,
        context_texts=context_texts,
        device=device,
        max_length=max_num_tokens,
    )
    print(f"Predicted transcription: {transcription}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audio-file-path",
        type=Path,
        required=True,
        help="Path to the audio file to transcribe.",
    )
    parser.add_argument(
        "--context-texts",
        type=str,
        nargs="+",
        required=True,
        help="One or more context texts (as in resource/text_data.py)",
    )
    parser.add_argument(
        "--max-num-tokens",
        type=int,
        default=20,
        help="Maximum number of tokens to generate.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device string (e.g., 'cuda' or 'cpu').",
    )
    args = parser.parse_args()
    main(**vars(args))
