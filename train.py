"""
Script to train the transducer model.

Usage example:
    # trains until interrupted with Ctrl+C
    python train.py
"""

import argparse
import logging

import torch
import torchaudio
from torch.utils.data import DataLoader

from src.model import Transducer
from src.supporting_code.dataset import SpeechDataset
from src.supporting_code.tokenizer import SimpleTokenizer

logging.basicConfig(level=logging.INFO)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

BATCH_SIZE = 1
DEVICE = "cpu"  # only tested on cpu


def main(
    learning_rate: float,
    num_epochs: int,
) -> None:
    device = torch.device(DEVICE)

    train_dataset = SpeechDataset(split="train")

    all_vocabs = train_dataset.collect_vocabulary()
    logging.info(f"Found {len(all_vocabs)} unique words in the dataset.")
    tokenizer = SimpleTokenizer(vocab=sorted(list(all_vocabs)))

    if BATCH_SIZE != 1:
        raise ValueError(
            f"You chose a batch size of {BATCH_SIZE}. Currently, "
            "only a batch size of 1 is supported"
        )
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda x: x,
    )

    transducer = Transducer(
        vocab_size=len(tokenizer.vocab),
        max_seq_len=tokenizer.max_length,
        pad_token_id=tokenizer.pad_token_id,
        sos_token_id=tokenizer.sos_token_id,
    ).to(device)

    optimizer = torch.optim.AdamW(transducer.parameters(), lr=learning_rate)

    losses = []
    epoch = 0

    try:
        # Handle continuous training when num_epochs is -1
        while epoch < num_epochs or num_epochs == -1:
            transducer.train()
            for batch in train_loader:
                b = batch[0]
                mel_features_BTC = b.mel_features_BTC.to(device)
                input_ids_transcription_BS = tokenizer.encode(b.transcription).to(
                    device
                )
                input_ids_context_BS = tokenizer.encode(" ; ".join(b.context)).to(
                    device
                )

                logits_BTSV = transducer(
                    mel_features_BTC=mel_features_BTC,
                    input_ids_transcription_BS=input_ids_transcription_BS,
                    input_ids_context_BS=input_ids_context_BS,
                )

                target_lengths = torch.tensor(
                    [input_ids_transcription_BS.shape[1]], dtype=torch.int32
                )
                logit_lengths = torch.tensor([logits_BTSV.shape[1]], dtype=torch.int32)

                loss = torchaudio.functional.rnnt_loss(
                    logits_BTSV,
                    input_ids_transcription_BS.to(torch.int32),
                    logit_lengths=logit_lengths,
                    target_lengths=target_lengths,
                    blank=tokenizer.blank_token_id,
                    reduction="mean",
                )

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                losses.append(loss.item())

            if epoch % 10 == 0 or (num_epochs != -1 and epoch == num_epochs - 1):
                transducer.eval()
                with torch.no_grad():
                    pred_token_ids = transducer.inference(
                        mel_features_BTC=mel_features_BTC,
                        input_ids_context_BS=input_ids_context_BS,
                        blank_token_id=tokenizer.blank_token_id,
                        max_length=50,
                    )
                    transcription = tokenizer.decode(pred_token_ids)

                print("=" * 100)
                print(f"Epoch {epoch} Loss: {sum(losses) / len(losses):.3f}")
                print(f"Ground Truth: {b.transcription}")
                print(f"Biasing Context: {b.context}")
                print(f"Predicted Transcription: `{transcription}`")

                losses = []

            epoch += 1

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving model...")

    torch.save(transducer.state_dict(), "transducer.pth")
    print("Model saved to transducer.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-epochs", type=int, default=-1)
    args = parser.parse_args()
    main(**vars(args))
