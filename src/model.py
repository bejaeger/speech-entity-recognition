"""
Module with all the models for the project.
"""

import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class ContextEncoder(nn.Module):
    """ContextEncoder using a TransformerEncoder and an integrated tokenizer.

    Args:
        max_seq_len (int): Maximum sequence length.
        vocab_size (int): Size of the vocabulary.
        pad_token_id (int): Padding token id.
        embed_dim (int): Embedding dimension.
        num_heads (int): Number of attention heads.
        num_layers (int): Number of transformer layers.
        feedforward_dim (int): Feedforward network dimension.
        max_seq_len (int): Maximum sequence length.
    """

    EMBED_DIM = 64
    NUM_HEADS = 2
    NUM_LAYERS = 2
    FEEDFORWARD_DIM = 128
    OUTPUT_DIM = 128

    def __init__(
        self,
        max_seq_len: int,
        vocab_size: int,
        pad_token_id: int,
        embed_dim: int = EMBED_DIM,
        num_heads: int = NUM_HEADS,
        num_layers: int = NUM_LAYERS,
        feedforward_dim: int = FEEDFORWARD_DIM,
        output_dim: int = OUTPUT_DIM,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.feedforward_dim = feedforward_dim
        self.max_seq_len = max_seq_len
        self.output_dim = output_dim
        self.pad_token_id = pad_token_id
        self.embedding = nn.Embedding(
            self.vocab_size,
            self.embed_dim,
            padding_idx=self.pad_token_id,
        )
        encoder_layer = TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=self.num_heads,
            dim_feedforward=self.feedforward_dim,
            batch_first=True,
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layer, num_layers=self.num_layers
        )
        self.output_layer = nn.Linear(self.embed_dim, self.output_dim)

    def forward(self, input_ids_1S: torch.Tensor) -> torch.Tensor:
        """Tokenizes input text and returns the hidden representation from the Transformer encoder.

        Args:
            text (str): Input text string.
        Returns:
            Tensor: Hidden representation of shape (batch, seq_len, embed_dim).
        """

        # _BSC stands for dimensions (batch, seq_len, embed_dim)
        embedded_BSC = self.embedding(input_ids_1S)
        src_key_padding_mask = input_ids_1S == self.pad_token_id
        hidden_BSC = self.transformer_encoder(
            embedded_BSC, src_key_padding_mask=src_key_padding_mask
        )
        return self.output_layer(hidden_BSC)


class LSTMEncoder(nn.Module):
    """A simple LSTM encoder for mel features.

    Args:
        input_dim (int): Number of input features (C).
        hidden_dim (int): Number of hidden units in the LSTM.
        num_layers (int): Number of LSTM layers.
    Returns:
        Tensor: The final hidden state of the LSTM.
    """

    INPUT_DIM = 80
    HIDDEN_DIM = 128
    NUM_LAYERS = 2

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        hidden_dim: int = HIDDEN_DIM,
        num_layers: int = NUM_LAYERS,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

    def forward(self, mel_features_BSC: torch.Tensor) -> torch.Tensor:
        """Forward pass for LSTM encoder.

        Args:
            mel_features_BSC (Tensor): Input tensor of shape [B, S, C].
        Returns:
            Tensor: Final hidden state of shape [B, hidden_dim].
        """
        _, (hidden_LBC, _) = self.lstm(mel_features_BSC)
        return hidden_LBC[-1, ...]


class TransformerPredictionNetwork(nn.Module):
    """Decoder-only Transformer-based prediction network for transducer models.

    Args:
        vocab_size (int): Size of the output vocabulary.
        embed_dim (int): Embedding dimension.
        num_heads (int): Number of attention heads.
        num_layers (int): Number of transformer decoder layers.
        feedforward_dim (int): Feedforward network dimension.
        max_seq_len (int): Maximum sequence length.
    Returns:
        Tensor: Output hidden states of shape (batch, seq_len, embed_dim).
    """

    EMBED_DIM = 64
    NUM_HEADS = 2
    NUM_LAYERS = 2
    FEEDFORWARD_DIM = 256

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        embed_dim: int = EMBED_DIM,
        num_heads: int = NUM_HEADS,
        num_layers: int = NUM_LAYERS,
        feedforward_dim: int = FEEDFORWARD_DIM,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.feedforward_dim = feedforward_dim
        self.max_seq_len = max_seq_len
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = nn.Parameter(
            torch.zeros(1, max_seq_len, embed_dim),
            requires_grad=True,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers
        )

    def forward(
        self,
        input_ids_BS: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass for the prediction network.

        Args:
            input_ids_1S (Tensor): Input token ids of shape (batch, seq_len).
            tgt_mask (Tensor | None): Optional target mask for causal decoding.
            tgt_key_padding_mask (Tensor | None): Optional padding mask.
        Returns:
            Tensor: Outputs hidden states of shape (batch, seq_len, embed_dim).
        """
        batch_size, seq_len = input_ids_BS.shape
        embedded_BSC = (
            self.embedding(input_ids_BS) + self.positional_encoding[:, :seq_len, :]
        )
        # Causal mask if not provided
        if tgt_mask is None:
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(
                input_ids_BS.device
            )

        # Decoder-only: memory is not used, so pass a dummy tensor with correct batch size and zero sequence length
        memory = torch.zeros(batch_size, 0, self.embed_dim, device=input_ids_BS.device)
        hidden_BSC = self.transformer_decoder(
            tgt=embedded_BSC,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        return hidden_BSC


class JointNetwork(nn.Module):
    """Joint network for combining acoustic and prediction network outputs in a transducer model.

    Args:
        acoustic_dim (int): Dimensionality of the acoustic encoder output.
        prediction_dim (int): Dimensionality of the prediction network output.
        vocab_size (int): Size of the output vocabulary.
        joint_dim (int): Dimensionality of the joint hidden layer.
    Returns:
        Tensor: Hidden states of shape [B, S, joint_dim].
    """

    JOINT_DIM = 128

    def __init__(
        self,
        acoustic_dim: int,
        prediction_dim: int,
        vocab_size: int,
        joint_dim: int = JOINT_DIM,
    ):
        super().__init__()
        self.acoustic_dim = acoustic_dim
        self.prediction_dim = prediction_dim
        self.vocab_size = vocab_size
        self.joint_dim = joint_dim
        self.linear = nn.Linear(acoustic_dim + prediction_dim, joint_dim)
        self.activation = nn.ReLU()
        self.output = nn.Linear(joint_dim, joint_dim)

    def forward(
        self,
        acoustic_hidden_BC: torch.Tensor,
        prediction_hidden_BSC: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for the joint network.

        Args:
            acoustic_hidden_BC (Tensor): Acoustic encoder output of shape [B, C].
            prediction_hidden_BSC (Tensor): Prediction network output of shape [B, S, C2].
        Returns:
            Tensor: hidden states of shape [B, S, joint_dim].
        """
        _, seq_len, _ = prediction_hidden_BSC.shape
        acoustic_expanded_BSC = acoustic_hidden_BC.unsqueeze(1).expand(-1, seq_len, -1)
        joint_input_BSCC = torch.cat(
            [acoustic_expanded_BSC, prediction_hidden_BSC], dim=-1
        )
        joint_hidden_BSJ = self.activation(self.linear(joint_input_BSCC))
        joint_hidden_BSJ = self.output(joint_hidden_BSJ)
        return joint_hidden_BSJ


class ContextBiasingAttention(nn.Module):
    """Context biasing module for the transducer.

    Args:
        hidden_dim (int): Dimensionality of the context encoder output.
        num_heads (int): Number of attention heads.
    """

    NUM_HEADS = 4

    def __init__(self, hidden_dim: int, num_heads: int = NUM_HEADS):
        super().__init__()
        self.context_encoder_output_dim = hidden_dim
        self.context_cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.query_layer = nn.Linear(hidden_dim, hidden_dim)
        self.key_layer = nn.Linear(hidden_dim, hidden_dim)
        self.value_layer = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self, joint_hidden_BSC: torch.Tensor, context_hidden_BSC: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass for the context biasing module.

        Args:
            context_hidden_BSC (Tensor): Context encoder output of shape [B, C].
            joint_hidden_BSJ (Tensor): Joint network output of shape [B, S, C2].
        Returns:
            Tensor: Context-biased hidden states of shape [B, S, C].
        """
        query = self.query_layer(joint_hidden_BSC)
        key = self.key_layer(context_hidden_BSC)
        value = self.value_layer(context_hidden_BSC)
        context_hidden_BSC, _ = self.context_cross_attention(query, key, value)
        return context_hidden_BSC


class Transducer(nn.Module):
    """Transducer model for speech recognition.

    Args:
        max_seq_len (int): Maximum sequence length.
        vocab_size (int): Size of the output vocabulary.
        pad_token_id (int): Padding token id.
    """

    def __init__(
        self,
        max_seq_len: int,
        vocab_size: int,
        pad_token_id: int,
    ) -> None:
        super().__init__()
        self.context_encoder = ContextEncoder(
            vocab_size=vocab_size,
            pad_token_id=pad_token_id,
            max_seq_len=max_seq_len,
        )
        self.lstm_encoder = LSTMEncoder()
        self.transformer_prediction_network = TransformerPredictionNetwork(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
        )
        self.joint_network = JointNetwork(
            acoustic_dim=self.lstm_encoder.hidden_dim,
            prediction_dim=self.transformer_prediction_network.embed_dim,
            vocab_size=vocab_size,
        )

        self.context_biasing = ContextBiasingAttention(
            hidden_dim=self.context_encoder.output_dim
        )

        self.output_layer = nn.Linear(self.context_encoder.output_dim, vocab_size)

    def forward(
        self,
        mel_features_BTC: torch.Tensor,
        input_ids_transcription_BS: torch.Tensor,
        input_ids_context_BS: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for the transducer.

        Args:
            mel_features_BSC (Tensor): Input tensor of shape [B, S, C].
            input_ids_BS (Tensor): Input token ids of shape [B, S].
        Returns:
            Tensor: Output hidden states of shape [B, S, joint_dim].
        """

        # THIS IS ONLY THE LAST HIDDEN STATE WHICH IS NOT IDEAL
        lstm_hidden_BC = self.lstm_encoder(mel_features_BTC)
        prediction_hidden_BSC = self.transformer_prediction_network(
            input_ids_transcription_BS
        )
        joint_hidden_BSC = self.joint_network(
            acoustic_hidden_BC=lstm_hidden_BC,
            prediction_hidden_BSC=prediction_hidden_BSC,
        )

        context_hidden_BSC = self.context_encoder(input_ids_context_BS)
        joint_biased_hidden_BSC = self.context_biasing(
            joint_hidden_BSC=joint_hidden_BSC, context_hidden_BSC=context_hidden_BSC
        )

        logits_BSV = self.output_layer(joint_biased_hidden_BSC)
        return logits_BSV
