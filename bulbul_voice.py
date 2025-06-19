import requests
from config import SARVAM_API_KEY

def transcribe_with_sarvam(audio_path):
    try:
        with open(audio_path, "rb") as audio_file:
            response = requests.post(
                "https://api.sarvam.ai/speech-to-text-translate",
                headers={"api-subscription-key": SARVAM_API_KEY},
                files={"file": ("audio.wav", audio_file, "audio/wav")}
            )

        if response.status_code != 200:
            print("Sarvam API Error:", response.text)
            return "", ""

        data = response.json()
        return data.get("transcript", ""), data.get("language_code", "en-IN")

    except Exception as e:
        print("Exception in Sarvam STT:", str(e))
        return "", ""