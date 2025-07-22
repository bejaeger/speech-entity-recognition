import torch

from src.model import (
    ContextEncoder,
    LSTMEncoder,
    TransformerPredictionNetwork,
    JointNetwork,
    ContextBiasingAttention,
    Transducer,
)


def test_context_encoder_forward():
    dummy_vocab = ["<pad>", "hello", "world"]
    max_seq_len = 5
    encoder = ContextEncoder(
        max_seq_len=max_seq_len,
        vocab_size=len(dummy_vocab),
        pad_token_id=0,
    )
    # Simulate a batch of 2 sequences, both padded to max_seq_len
    input_ids_1S = torch.tensor(
        [
            [1, 2, 0, 0, 0],  # 'hello world' + padding
            [2, 1, 0, 0, 0],  # 'world hello' + padding
        ]
    )
    output_2SC = encoder(input_ids_1S)
    assert output_2SC.shape == (2, max_seq_len, encoder.output_dim)


def test_simple_lstm_encoder_output_shape_and_type():
    """Test SimpleLSTMEncoder returns correct shape and type for dummy input."""
    batch_size = 1
    seq_len = 10
    model = LSTMEncoder()
    input_dim = model.input_dim
    dummy_input_BSC = torch.randn(batch_size, seq_len, input_dim)
    output_BH = model(dummy_input_BSC)
    assert isinstance(output_BH, torch.Tensor)
    assert output_BH.shape == (batch_size, model.hidden_dim)


def test_transformer_prediction_network_output_shape_and_type():
    """Test TransformerPredictionNetwork returns correct shape and type for dummy input."""

    batch_size = 2
    seq_len = 7
    vocab_size = 10
    max_seq_len = 12
    model = TransformerPredictionNetwork(vocab_size=vocab_size, max_seq_len=max_seq_len)
    dummy_input_1S = torch.randint(0, vocab_size, (batch_size, seq_len))
    output = model(dummy_input_1S)
    assert isinstance(output, torch.Tensor)
    assert output.shape == (batch_size, seq_len, model.embed_dim)


def test_joint_network_output_shape_and_type():
    """Test JointNetwork returns correct shape and type for dummy input."""
    batch_size = 2
    seq_len = 5
    acoustic_dim = 16
    prediction_dim = 32
    joint_dim = 64
    # Acoustic encoder output: [B, C]
    acoustic_hidden_BC = torch.randn(batch_size, acoustic_dim)
    # Prediction network output: [B, S, C2]
    prediction_hidden_BSC = torch.randn(batch_size, seq_len, prediction_dim)
    model = JointNetwork(
        acoustic_dim=acoustic_dim,
        prediction_dim=prediction_dim,
        vocab_size=100,
        joint_dim=joint_dim,
    )
    output = model(acoustic_hidden_BC, prediction_hidden_BSC)
    assert isinstance(output, torch.Tensor)
    assert output.shape == (batch_size, seq_len, joint_dim)


def test_context_biasing_attention_output_shape_and_type():
    """Test ContextBiasingAttention returns correct shape and type for dummy input."""
    batch_size = 2
    seq_len = 4
    seq_len_context = 3
    hidden_dim = 8
    # Joint network output: [B, S, C]
    joint_hidden_BSC = torch.randn(batch_size, seq_len, hidden_dim)
    # Context encoder output: [B, Sc, C]
    context_hidden_BSC = torch.randn(batch_size, seq_len_context, hidden_dim)
    model = ContextBiasingAttention(hidden_dim=hidden_dim, num_heads=2)
    output = model(joint_hidden_BSC, context_hidden_BSC)
    assert isinstance(output, torch.Tensor)
    assert output.shape == (batch_size, seq_len, hidden_dim)


def test_transducer_output_shape_and_type():
    """Test Transducer returns correct shape and type for dummy input."""
    batch_size = 2
    seq_len = 6
    seq_len_context = 3
    vocab = ["<pad>", "hello", "world"]
    vocab_size = len(vocab)
    pad_token_id = vocab.index("<pad>")
    max_seq_len = seq_len
    mel_feature_dim = 80  # LSTMEncoder default
    mel_features_BSC = torch.randn(batch_size, seq_len, mel_feature_dim)
    input_ids_transcription_BS = torch.randint(0, vocab_size, (batch_size, seq_len))
    input_ids_context_BS = torch.randint(0, vocab_size, (batch_size, seq_len_context))
    model = Transducer(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        pad_token_id=pad_token_id,
    )
    output = model(
        mel_features_BSC=mel_features_BSC,
        input_ids_transcription_BS=input_ids_transcription_BS,
        input_ids_context_BS=input_ids_context_BS,
    )
    assert isinstance(output, torch.Tensor)
    assert output.shape == (batch_size, seq_len, vocab_size)
