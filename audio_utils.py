import base64
import requests
from config import SARVAM_API_KEY
from pydub import AudioSegment
import os
import random

BULBUL_SPEAKERS = ["anushka", "manisha", "vidya", "arya", "abhilash", "karun", "hitesh"]

def save_audio_from_browser(audio_bytes, filename="input.wav"):
    with open(filename, "wb") as f:
        f.write(audio_bytes)

def validate_lang_code(lang_code):
    try:
        response = requests.post(
            "https://api.sarvam.ai/text-to-speech",
            headers={"api-subscription-key": SARVAM_API_KEY},
            json={"text": "test", "target_language_code": lang_code}
        )
        return lang_code if response.status_code == 200 else None
    except Exception as e:
        print(f"Lang code validation error: {e}")
        return None

def text_to_speech(text, lang_code, filename="output.wav", speaker=None):
    lang_code = validate_lang_code(lang_code)
    if not lang_code:
        print("Invalid lang code. Aborting.")
        return ""

    if not speaker:
        speaker = random.choice(BULBUL_SPEAKERS)

    text = text.replace('\n', ' ').replace("**", "").strip()
    chunk_size = 300
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    combined = AudioSegment.silent(duration=0)
    success_chunks = 0

    for i, chunk in enumerate(chunks):
        try:
            tts = requests.post(
                "https://api.sarvam.ai/text-to-speech",
                headers={"api-subscription-key": SARVAM_API_KEY},
                json={
                    "text": chunk,
                    "target_language_code": lang_code,
                    "model": "bulbul:v2",
                    "speaker": speaker
                }
            )
            tts.raise_for_status()
            audio_base64 = tts.json().get("audios", [None])[0]

            if audio_base64:
                chunk_path = f"chunk_{i}.wav"
                with open(chunk_path, "wb") as f:
                    f.write(base64.b64decode(audio_base64))
                segment = AudioSegment.from_wav(chunk_path)
                combined += segment
                os.remove(chunk_path)
                success_chunks += 1
        except Exception as e:
            print(f"Chunk {i} failed: {e}")
            continue

    if success_chunks == 0:
        return ""

    combined.export(filename, format="wav")
    with open(filename, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")
