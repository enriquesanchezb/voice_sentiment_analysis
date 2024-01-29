import os

from src.summarization import topics_for_text


def test_topics_for_text():
    # Define the input parameters

    file_conv = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "conversations",
            "technology_1706560176_9889.txt",
        )
    )

    # Call the function
    topics = topics_for_text(file_conv)

    # Assert the expected output
    assert isinstance(topics, list)
    assert "technology" in topics
