import os
from src.generate_audio import new_audio_conversation


def test_new_audio_conversation():
    topic = "technology"

    # Call the function
    file_name = new_audio_conversation(topic)

    # Assert that the file exists
    assert os.path.exists(file_name)

    # Assert that the file name matches the expected format
    assert file_name.startswith(f"conversations/{topic}_")
    assert file_name.endswith(".txt")
