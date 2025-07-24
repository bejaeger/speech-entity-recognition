import math

import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from src.supporting_code.model import (
    JointNetwork,
    LSTMEncoder,
    TransformerPredictionNetwork,
)


class ContextEncoder(nn.Module):
    """ContextEncoder using a TransformerEncoder.

    Takes a sequence of token ids and returns a hidden representation of the context.
    This is intended to be used in an ASR pipeline to bias the model towards a specific context,
    e.g. a list of entities. To this end, the output of the ContextEncoder can e.g. be
    attended to via cross-attention.

    Args:
        vocab_size (int): Size of the vocabulary.
        pad_token_id (int): Padding token id.
        output_dim (int): Output dimension.
        embed_dim (int): Embedding dimension.
        num_heads (int): Number of attention heads.
        num_layers (int): Number of transformer layers.
        feedforward_dim (int): Feedforward network dimension.
    """

    EMBED_DIM = 16
    NUM_HEADS = 2
    NUM_LAYERS = 2
    FEEDFORWARD_DIM = 64

    def __init__(
        self,
        vocab_size: int,
        pad_token_id: int,
        output_dim: int,
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

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize parameters following best practices for transformer models."""
        nn.init.normal_(
            self.embedding.weight, mean=0.0, std=1.0 / math.sqrt(self.embed_dim)
        )

        with torch.no_grad():
            self.embedding.weight[self.pad_token_id].fill_(0)

        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

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


class Transducer(nn.Module):
    """Transducer model for speech recognition.

    Args:
        max_seq_len (int): Maximum sequence length.
        vocab_size (int): Size of the output vocabulary.
        sos_token_id (int): Start of sequence token id.
        pad_token_id (int): Padding token id.
        context_attention_num_heads (int): Number of attention heads for context attention.
        use_encoder_context_attention (bool): Whether to bias the encoder with context attention.
    """

    CONTEXT_ATTENTION_NUM_HEADS = 2

    def __init__(
        self,
        max_seq_len: int,
        vocab_size: int,
        sos_token_id: int,
        pad_token_id: int,
        context_attention_num_heads: int = CONTEXT_ATTENTION_NUM_HEADS,
        use_encoder_context_attention: bool = False,
    ) -> None:
        super().__init__()
        self.lstm_encoder = LSTMEncoder()
        self.transformer_prediction_network = TransformerPredictionNetwork(
            vocab_size=vocab_size,
            sos_token_id=sos_token_id,
            max_seq_len=max_seq_len,
        )

        self.joint_network = JointNetwork(
            acoustic_dim=self.lstm_encoder.hidden_dim,
            prediction_dim=self.transformer_prediction_network.output_dim,
        )

        self.context_encoder = ContextEncoder(
            vocab_size=vocab_size,
            pad_token_id=pad_token_id,
            output_dim=self.transformer_prediction_network.output_dim,
        )

        self.predictor_context_attention = nn.MultiheadAttention(
            embed_dim=self.context_encoder.output_dim,
            num_heads=context_attention_num_heads,
            batch_first=True,
        )
        if use_encoder_context_attention:
            assert self.context_encoder.output_dim == self.lstm_encoder.hidden_dim, (
                "Output dimension of context encoder must match hidden dimension of LSTM encoder"
            )
            self.encoder_context_attention = nn.MultiheadAttention(
                embed_dim=self.context_encoder.output_dim,
                num_heads=context_attention_num_heads,
                batch_first=True,
            )
        else:
            self.encoder_context_attention = None

        self.output_layer = nn.Linear(self.joint_network.joint_dim, vocab_size)

        self._init_weights()

        print(
            f"Number of parameters in context encoder: {sum(p.numel() for p in self.context_encoder.parameters()):,}"
        )
        print(
            f"Number of parameters in lstm encoder: {sum(p.numel() for p in self.lstm_encoder.parameters()):,}"
        )
        print(
            f"Number of parameters in transformer prediction network: {sum(p.numel() for p in self.transformer_prediction_network.parameters()):,}"
        )
        print(
            f"Number of parameters in joint network: {sum(p.numel() for p in self.joint_network.parameters()):,}"
        )
        print(
            f"Number of parameters in output layer: {sum(p.numel() for p in self.output_layer.parameters()):,}"
        )
        print(
            f"Total number of parameters: {sum(p.numel() for p in self.parameters()):,}"
        )

    def _init_weights(self) -> None:
        """Initialize transducer-specific parameters."""
        nn.init.xavier_uniform_(self.predictor_context_attention.in_proj_weight)
        nn.init.zeros_(self.predictor_context_attention.in_proj_bias)
        nn.init.xavier_uniform_(self.predictor_context_attention.out_proj.weight)
        nn.init.zeros_(self.predictor_context_attention.out_proj.bias)

        # Initialize final output layer with smaller variance to prevent large initial logits
        nn.init.xavier_uniform_(self.output_layer.weight, gain=0.1)
        nn.init.zeros_(self.output_layer.bias)

    def forward(
        self,
        mel_features_BTC: torch.Tensor,
        input_ids_transcription_BS: torch.Tensor,
        input_ids_context_BS: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for the transducer.

        Args:
            mel_features_BTC (Tensor): Input tensor of shape [B, T, C].
            input_ids_transcription_BS (Tensor): Input token ids of shape [B, S].
            input_ids_context_BS (Tensor): Input token ids of shape [B, S'].
        Returns:
            Tensor: Output hidden states of shape [B, T, S, V], where V is the vocabulary size.
        """

        lstm_hidden_BTC = self.lstm_encoder(mel_features_BTC)

        prediction_hidden_BSC = self.transformer_prediction_network(
            input_ids_transcription_BS
        )

        context_hidden_BSC = self.context_encoder(input_ids_context_BS)

        prediction_biased_hidden_BSC, _ = self.predictor_context_attention(
            query=prediction_hidden_BSC,
            key=context_hidden_BSC,
            value=context_hidden_BSC,
        )
        if self.encoder_context_attention is not None:
            lstm_biased_hidden_BTC, _ = self.encoder_context_attention(
                query=lstm_hidden_BTC,
                key=context_hidden_BSC,
                value=context_hidden_BSC,
            )
        else:
            lstm_biased_hidden_BTC = lstm_hidden_BTC

        joint_hidden_BTSC = self.joint_network(
            acoustic_hidden_BTC=lstm_biased_hidden_BTC,
            prediction_hidden_BSC=prediction_biased_hidden_BSC,
        )

        logits_BTSV = self.output_layer(joint_hidden_BTSC)
        return logits_BTSV

    def inference(
        self,
        mel_features_BTC: torch.Tensor,
        input_ids_context_BS: torch.Tensor,
        blank_token_id: int,
        max_length: int = 50,
    ) -> list[int]:
        """Run greedy inference using the model's current weights.

        Args:
            mel_features_BTC: Audio features of shape (batch, time, channels)
            input_ids_context_BS: Context input IDs for biasing
            blank_token_id: ID of the blank token (no emission)
            max_length: Maximum output sequence length
        Returns:
            List of predicted token IDs (excluding blank tokens)
        """
        self.eval()
        device = mel_features_BTC.device
        batch_size, time_steps, _ = mel_features_BTC.shape

        if batch_size != 1:
            raise ValueError("Currently only supports batch size of 1")

        lstm_hidden_BTC = self.lstm_encoder(mel_features_BTC)

        context_hidden_BSC = self.context_encoder(input_ids_context_BS)

        predicted_tokens = []
        current_prediction_input = torch.tensor([[]], dtype=torch.long, device=device)

        for t in range(time_steps):
            current_acoustic_B1C = lstm_hidden_BTC[:, t : t + 1, :]

            # Current prediction network state
            if len(predicted_tokens) == 0:
                # Start with empty prediction (just SOS will be added internally)
                current_prediction_input = torch.tensor(
                    [[]], dtype=torch.long, device=device
                )
            else:
                current_prediction_input = torch.tensor(
                    [predicted_tokens], dtype=torch.long, device=device
                )

            prediction_hidden_BSC = self.transformer_prediction_network(
                current_prediction_input
            )

            prediction_biased_hidden_BSC, _ = self.predictor_context_attention(
                query=prediction_hidden_BSC,
                key=context_hidden_BSC,
                value=context_hidden_BSC,
            )

            if self.encoder_context_attention is not None:
                lstm_biased_hidden_BTC, _ = self.encoder_context_attention(
                    query=current_acoustic_B1C,
                    key=context_hidden_BSC,
                    value=context_hidden_BSC,
                )
            else:
                lstm_biased_hidden_BTC = current_acoustic_B1C

            # Get the last prediction state (most recent)
            if prediction_biased_hidden_BSC.shape[1] > 0:
                prediction_state_B1C = prediction_biased_hidden_BSC[:, -1:, :]
            else:
                # If empty, use the zero state from the prediction network
                prediction_state_B1C = prediction_biased_hidden_BSC

            # Apply joint network for this specific (time, sequence) pair
            joint_input_B1C = torch.cat(
                [
                    lstm_biased_hidden_BTC,
                    prediction_state_B1C,
                ],
                dim=-1,
            )

            joint_hidden_B1C = self.joint_network.activation(
                self.joint_network.linear(joint_input_B1C)
            )
            joint_hidden_B1C = self.joint_network.output(joint_hidden_B1C)

            logits_B1V = self.output_layer(joint_hidden_B1C)
            current_logits = logits_B1V.squeeze()

            predicted_token_id = torch.argmax(current_logits).item()

            # If it's not a blank token, emit it
            if predicted_token_id != blank_token_id:
                predicted_tokens.append(predicted_token_id)

                # Stop if we've reached max length
                if len(predicted_tokens) >= max_length:
                    break

            # If it's a blank, we continue to the next time step without emitting

        return predicted_tokens
