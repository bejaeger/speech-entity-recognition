"""
Module with the tokenizer for the project.
"""

import torch

from src.supporting_code.config import MAX_SEQ_LEN


class SimpleTokenizer:
    """A simple whitespace tokenizer with a fixed vocabulary."""

    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"
    BLANK_TOKEN = "<blank>"
    SOS_TOKEN = "<sos>"
    EOS_TOKEN = "<eos>"

    def __init__(
        self,
        vocab: list[str],
        max_length: int = MAX_SEQ_LEN,
    ):
        self.vocab = [
            self.BLANK_TOKEN,
            self.PAD_TOKEN,
            self.UNK_TOKEN,
            self.SOS_TOKEN,
            self.EOS_TOKEN,
        ] + vocab
        self.token_to_id = {token: idx for idx, token in enumerate(self.vocab)}
        self.max_length = max_length

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def pad_token_id(self) -> int:
        return self.token_to_id[self.PAD_TOKEN]

    @property
    def unk_token_id(self) -> int:
        return self.token_to_id[self.UNK_TOKEN]

    @property
    def blank_token_id(self) -> int:
        return self.token_to_id[self.BLANK_TOKEN]

    @property
    def sos_token_id(self) -> int:
        return self.token_to_id[self.SOS_TOKEN]

    @property
    def eos_token_id(self) -> int:
        return self.token_to_id[self.EOS_TOKEN]

    def encode(self, text: str) -> torch.Tensor:
        """
        Encodes a text string into a tensor of token ids.

        Args:
            text (str): The text string to encode.
        Returns:
            Tensor: A tensor of token ids of shape (1, seq_len).
        """
        tokens = text.lower().split()
        ids = [self.token_to_id.get(token, self.unk_token_id) for token in tokens]
        if len(ids) > self.max_length:
            raise ValueError(
                f"Text length {len(ids)} exceeds max_length {self.max_length}"
            )
        return torch.tensor([ids], dtype=torch.long)

    def decode(self, ids: list[int]) -> str:
        return " ".join(self.vocab[id] for id in ids)


__all__ = ["SimpleTokenizer"]
