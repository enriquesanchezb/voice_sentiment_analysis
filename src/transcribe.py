"""
This module is used to transcribe the audio file and return the text
"""

from faster_whisper import WhisperModel


model = WhisperModel("small", device="cpu", compute_type="int8")


def transcribe_audio(audio_file: str) -> str:
    """Returns the transcribed text and the sentiment analysis results"""
    try:
        segments, _ = model.transcribe(audio_file, beam_size=5)
        text = list(segments)[0].text
        return text
    except Exception as e:
        print(f"Error in transcribe_audio: {e}")
        return ""
