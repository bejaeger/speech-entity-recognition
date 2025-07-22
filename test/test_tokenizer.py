from src.tokenizer import SimpleTokenizer


def test_encode_decode():
    """Test that encoding and then decoding returns the original tokens for known vocab."""
    tokenizer = SimpleTokenizer(vocab=["hello", "world"])
    text = "hello world"
    encoded = tokenizer.encode(text)
    ids = encoded[0].tolist()
    decoded = tokenizer.decode(ids)
    assert decoded == text
