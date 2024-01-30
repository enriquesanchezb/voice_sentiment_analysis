import os
import tempfile

from src.generate_conversation import generate_conversation
from src.sentiment import analyze_sentiment
from src.summarization import topics_for_text


def test_generate_conversation_positive_sentiment():
    # Define the input parameters
    topic = "ai"
    sentiment = "admiration"

    # Call the function
    conversation = generate_conversation(topic, sentiment)

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

    assert topic in new_topics, "Topic is not in the index of new_topics"
