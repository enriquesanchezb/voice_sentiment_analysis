import os
import random
import re
import shutil
import time

from pydub import AudioSegment

from .generate_conversation import generate_conversation


def _generate_audio_from_text(
    text_file, voice1="Daniel", voice2="Samantha", temp_dir="temp_audio"
):
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    with open(text_file, "r", encoding="utf-8") as file:
        lines = file.readlines()

    for i, line in enumerate(lines):
        output_file = os.path.join(temp_dir, f"{i+1}.aiff")
        voice = voice1 if i % 2 == 0 else voice2
        os.system(f'say -v {voice} "{line}" -o {output_file}')


def _combine_audio(audio_dir, output_file, silence_duration=500):
    chunks1 = []
    chunks2 = []
    i = 1
    for file in sorted(os.listdir(audio_dir)):
        if i % 2 == 0:
            chunks2.append(AudioSegment.from_file(os.path.join(audio_dir, file)))
        else:
            chunks1.append(AudioSegment.from_file(os.path.join(audio_dir, file)))
        i += 1
    combined = AudioSegment.silent(duration=0)
    for chunk1, chunk2 in zip(chunks1, chunks2):
        combined += (
            chunk1
            + AudioSegment.silent(duration=silence_duration)
            + chunk2
            + AudioSegment.silent(duration=silence_duration)
        )

    combined.export(output_file, format="mp3")


def _text_to_file(text, file_name):
    phrases = re.findall(r"\"(.*?)\"", text)

    with open(file_name, "w") as file:
        for phrase in phrases:
            file.write(phrase + "\n")


def new_audio_conversation(topic) -> str:
    shutil.rmtree("temp_audio", ignore_errors=True)
    topic = "technology"
    text_file = generate_conversation(topic)
    timestamp = str(int(time.time()))
    random_number = str(random.randint(1000, 9999))
    file_name = f"conversations/{topic}_{timestamp}_{random_number}.txt"

    _text_to_file(text_file, file_name)
    _generate_audio_from_text(file_name)
    _combine_audio(
        "temp_audio", f"conversations/{topic}_{timestamp}_{random_number}.mp3"
    )
    shutil.rmtree("temp_audio", ignore_errors=True)
    return file_name
