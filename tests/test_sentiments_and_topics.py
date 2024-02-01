import os
import tempfile

import pytest

from src.generate_conversation import generate_conversation
from src.sentiment import analyze_sentiment
from src.summarization import topics_for_text

SENTIMENTS = ["neutral", "sadness", "neutral", "joy"]
TOPICS = ["ai", "soccer", "art", "science"]


@pytest.mark.parametrize("sentiment", SENTIMENTS)
@pytest.mark.parametrize("topic", TOPICS)
def test_generate_conversation(sentiment, topic):
    # Call the function
    conversation = generate_conversation(topic, sentiment, llm="llama2")

    new_sentiment = analyze_sentiment(conversation)
    # Assert that the conversation has a positive sentiment

    assert (
        sentiment in s for s in new_sentiment
    ), "Sentiment is not in the index of new_sentiment"

    # Save the conversation to a temporary text file
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as file:
        file.write(conversation)
        temp_filepath = file.name

    new_topics = topics_for_text(temp_filepath)
    os.remove(temp_filepath)
    assert any(topic in word for word in new_topics)
