import os

from src.transcribe import transcribe_audio


def test_transcribe_audio_success():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    audio_file = os.path.join(current_dir, "samplings/thisisatest.mp3")
    expected_text = "This is a test"
    result = transcribe_audio(audio_file).strip().rstrip(".")
    assert result == expected_text


def test_transcribe_audio_failure():
    # Transcription failure
    audio_file = "samplings/nonexistent_audio.wav"
    expected_text = ""
    result = transcribe_audio(audio_file)
    assert result == expected_text
