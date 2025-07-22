import argparse
import logging

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.dataset import SpeechDataset
from src.model import Transducer
from src.tokenizer import SimpleTokenizer

logging.basicConfig(level=logging.INFO)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

NUM_EPOCHS = 100


def main(
    device: torch.device | str | None,
) -> None:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    train_dataset = SpeechDataset(split="train")
    test_dataset = SpeechDataset(split="test")

    all_vocabs = train_dataset.collect_vocabulary()
    all_vocabs.update(test_dataset.collect_vocabulary())
    logging.info(f"Found {len(all_vocabs)} unique words in the dataset.")
    tokenizer = SimpleTokenizer(vocab=list(all_vocabs))

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=lambda x: x,
    )
    # test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    transducer = Transducer(
        vocab_size=len(tokenizer.vocab),
        max_seq_len=tokenizer.max_length,
        pad_token_id=tokenizer.pad_token_id,
    ).to(device)

    optimizer = torch.optim.AdamW(transducer.parameters(), lr=5e-4)

    transducer.train()
    losses = []
    for epoch in range(NUM_EPOCHS):
        for batch in train_loader:
            b = batch[0]
            mel_features_BTC = b.mel_features_BTC.to(device)
            input_ids_transcription_BS = tokenizer.encode(b.transcription).to(device)
            input_ids_context_BS = tokenizer.encode(" ".join(b.context)).to(device)

            logits_BSV = transducer(
                mel_features_BTC=mel_features_BTC,
                input_ids_transcription_BS=input_ids_transcription_BS,
                input_ids_context_BS=input_ids_context_BS,
            )

            loss = F.cross_entropy(
                logits_BSV.transpose(1, 2),
                input_ids_transcription_BS,
                ignore_index=tokenizer.pad_token_id,
            )
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())

        if epoch % 10 == 0 or epoch == NUM_EPOCHS - 1:
            pred_token_ids = logits_BSV.argmax(dim=-1).squeeze(0).tolist()
            # Remove padding and blank tokens
            pred_token_ids = [
                tid
                for tid in pred_token_ids
                if tid not in {tokenizer.pad_token_id, tokenizer.blank_token_id}
            ]
            transcription = tokenizer.decode(pred_token_ids)

            print("=" * 100)
            print(f"Epoch {epoch} Loss: {sum(losses) / len(losses)}")
            print(f"Biasing Context: {b.context}")
            print(f"Transcription: {transcription}")

            losses = []

    torch.save(transducer.state_dict(), "transducer.pth")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    main(**vars(args))
