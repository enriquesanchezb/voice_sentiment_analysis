from src.sentiment import analyze_sentiment


def test_analyze_sentiment_positives():
    # Positive sentiment
    text = "I love this movie!"
    expected_keys = ["love"]
    result = analyze_sentiment(text)
    assert all(key in result.keys() for key in expected_keys)


def test_analyze_sentiment_negatives():
    # Negative sentiment
    text = "I hate this product!"
    expected_keys = ["anger"]
    result = analyze_sentiment(text)
    assert all(key in result.keys() for key in expected_keys)


def test_analyze_sentiment_empty():
    # Negative sentiment
    text = ""
    expected_keys = ["neutral"]
    result = analyze_sentiment(text)
    assert all(key in result.keys() for key in expected_keys)


def test_analyze_sentiment_neutral():
    # Negative sentiment
    text = "hello how are you"
    expected_keys = ["neutral"]
    result = analyze_sentiment(text)
    assert all(key in result.keys() for key in expected_keys)
