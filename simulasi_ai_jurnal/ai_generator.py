import os
import random
import pandas as pd
from groq import Groq
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(BASE_DIR, '..', '.env'))

USE_MOCK = False
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    USE_MOCK = True
    client = None
else:
    client = Groq(api_key=api_key)

def get_mock_response(emosi: str) -> str:
    """Fallback pesan jika API Groq tidak tersedia."""
    return f"Terima kasih sudah berbagi perasaanmu. Merasa {emosi} adalah hal yang manusiawi. Tetap kuat ya!"

def get_ai_response(journal_text: str, emotion_label: str = None) -> str:
    """
    Menghasilkan kalimat penyemangat (pop-up) berdasarkan isi jurnal dan label emosi.
    """
    emosi = str(emotion_label).strip() if emotion_label else "Tidak Diketahui"

    if USE_MOCK or not client:
        return get_mock_response(emosi)

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Tugasmu adalah membuat pesan pop-up dukungan psikologis untuk aplikasi kesehatan mental mahasiswa. "
                        "Berikan pesan maksimal 2 kalimat pendek yang hangat. "
                        "PENTING: Jangan gunakan kata ganti orang pertama (aku, saya, kami). "
                        "Validasi perasaan mahasiswa berdasarkan teks jurnal mereka dan label emosi yang diberikan. "
                        "Label emosi bisa bervariasi dari kategori Senang hingga Putus Asa (total 28 jenis perasaan). "
                        "Gunakan bahasa Indonesia yang empati, santai, dan tanpa format markdown."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Isi jurnal: '{journal_text}'. "
                        f"Label perasaan: {emosi}."
                    )
                }
            ],
            model=GROQ_MODEL,
            temperature=0.7,
        )
        return chat_completion.choices[0].message.content.strip()

    except Exception:
        return get_mock_response(emosi)

if __name__ == "__main__":
    # Test simulation
    print(f"AI Response: {get_ai_response('Hari ini terasa sangat berat karena banyak tugas', 'Lelah')}")