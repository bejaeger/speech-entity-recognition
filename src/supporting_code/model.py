"""
Module with all the models for the project.
"""

import math
import torch
from torch import nn


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
    HIDDEN_DIM = 64
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

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize LSTM parameters with orthogonal and Xavier initialization."""
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                # Input-to-hidden weights: Xavier uniform
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                # Hidden-to-hidden weights: Orthogonal
                nn.init.orthogonal_(param)
            elif "bias" in name:
                # Biases: zeros, except forget gate bias to 1
                nn.init.zeros_(param)
                # Set forget gate bias to 1 for better gradient flow
                hidden_size = param.size(0) // 4
                param.data[hidden_size : 2 * hidden_size].fill_(1.0)

    def forward(self, mel_features_BSC: torch.Tensor) -> torch.Tensor:
        """Forward pass for LSTM encoder.

        Args:
            mel_features_BSC (Tensor): Input tensor of shape [B, S, C].
        Returns:
            Tensor: Final hidden state of shape [B, T, hidden_dim].
        """
        output_BTC, _ = self.lstm(mel_features_BSC)
        return output_BTC


class TransformerPredictionNetwork(nn.Module):
    """Decoder-only Transformer-based prediction network for transducer models.

    Args:
        vocab_size (int): Size of the output vocabulary.
        sos_token_id (int): Start of sequence token id.
        max_seq_len (int): Maximum sequence length.
        output_dim (int): Output dimension.
        embed_dim (int): Embedding dimension.
        num_heads (int): Number of attention heads.
        num_layers (int): Number of transformer decoder layers.
        feedforward_dim (int): Feedforward network dimension.
        max_seq_len (int): Maximum sequence length.
    Returns:
        Tensor: Output hidden states of shape (batch, seq_len, embed_dim).
    """

    EMBED_DIM = 32
    OUTPUT_DIM = 32
    NUM_HEADS = 2
    NUM_LAYERS = 2
    FEEDFORWARD_DIM = 128

    def __init__(
        self,
        vocab_size: int,
        sos_token_id: int,
        max_seq_len: int,
        output_dim: int = OUTPUT_DIM,
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
        self.sos_token_id = sos_token_id
        self.max_seq_len = max_seq_len
        self.output_dim = output_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = nn.Parameter(
            torch.zeros(1, max_seq_len + 1, embed_dim),  # + 1 because of sos token
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
        self.output_layer = nn.Linear(embed_dim, output_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize parameters following best practices for transformer decoders."""
        # Initialize embedding with normal distribution
        nn.init.normal_(
            self.embedding.weight, mean=0.0, std=1.0 / math.sqrt(self.embed_dim)
        )

        # Initialize positional encoding with small random values
        nn.init.normal_(self.positional_encoding, mean=0.0, std=0.1)

        # Initialize output layer with Xavier uniform and smaller scale
        nn.init.xavier_uniform_(self.output_layer.weight, gain=0.5)
        nn.init.zeros_(self.output_layer.bias)

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
        batch_size = input_ids_BS.shape[0]
        sos_tokens = torch.full(
            (batch_size, 1),
            self.sos_token_id,
            dtype=input_ids_BS.dtype,
            device=input_ids_BS.device,
        )
        input_ids_BS = torch.cat([sos_tokens, input_ids_BS], dim=1)
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
        return self.output_layer(hidden_BSC)


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

    JOINT_DIM = 64

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

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize joint network parameters."""
        # Initialize linear layers with Xavier uniform
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        acoustic_hidden_BTC: torch.Tensor,
        prediction_hidden_BSC: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for the joint network.

        Args:
            acoustic_hidden_BTC (Tensor): Acoustic encoder output of shape [B, T, C].
            prediction_hidden_BSC (Tensor): Prediction network output of shape [B, S, C2].
        Returns:
            Tensor: hidden states of shape [B, S, joint_dim].
        """
        T = acoustic_hidden_BTC.shape[1]
        S = prediction_hidden_BSC.shape[1]

        acoustic_expanded_BTSC = acoustic_hidden_BTC.unsqueeze(2).expand(-1, T, S, -1)
        prediction_expanded_BTSC = prediction_hidden_BSC.unsqueeze(1).expand(
            -1, T, S, -1
        )

        joint_input_BTSC = torch.cat(
            [acoustic_expanded_BTSC, prediction_expanded_BTSC], dim=-1
        )

        joint_hidden_BTSC = self.activation(self.linear(joint_input_BTSC))
        joint_hidden_BTSC = self.output(joint_hidden_BTSC)
        return joint_hidden_BTSC
