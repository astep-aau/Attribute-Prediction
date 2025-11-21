# tests/test_example.py

def test_basic_example():
    """A simple test to verify pytest is working"""
    result = 2 + 2
    assert result == 4

def test_string_operations():
    """Test string manipulation"""
    text = "hello"
    assert text.upper() == "HELLO"
    assert len(text) == 5
