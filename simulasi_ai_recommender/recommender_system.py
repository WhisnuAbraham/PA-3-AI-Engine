import os
import random
import pandas as pd
from groq import Groq
from dotenv import load_dotenv
from utils.db_connector import get_database

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(BASE_DIR, '..', '.env'))

USE_MOCK = False
GROQ_MODEL = "llama-3.3-70b-versatile"

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    USE_MOCK = True
    client = None
else:
    try:
        client = Groq(api_key=api_key)
    except Exception:
        USE_MOCK = True
        client = None

def _get_mock_fallback(feeling: str) -> str:
    """Cadangan jika API error atau kuota habis."""
    return f"Tetap semangat! Meskipun kamu sedang merasa {feeling}, ingatlah bahwa setiap hari adalah kesempatan baru untuk memulai kembali."

# FUNGSI UTAMA — dipanggil oleh main.py
def get_recommendation(nim: str, feeling: str = None, mood: str = None) -> str:
    """
    Menghasilkan quote motivasi berdasarkan emosi (mood & feeling) mahasiswa.
    Jika feeling tidak diberikan, ambil dari riwayat terakhir di MongoDB.
    """
    emosi_label = feeling
    mood_label = mood

    if not emosi_label:
        try:
            db = get_database()
            last_entry = db.moods.find_one(
                {"username": nim},
                sort=[("created_at", -1)]
            )
            if not last_entry:
                last_entry = db.moods.find_one(
                    {"nim": nim},
                    sort=[("created_at", -1)]
                )

            if last_entry:
                emosi_label = last_entry.get('perasaan', 'Biasa Saja')
                mood_label = last_entry.get('mood_label', 'Netral')
            else:
                # Default fallback jika benar-benar tidak ada data
                emosi_label = "Biasa Saja"
                mood_label = "Netral"

        except Exception as e:
            print(f"Database error in recommender: {e}")
            emosi_label = "Biasa Saja"
            mood_label = "Netral"

    # Context untuk AI
    context_perasaan = f"Kategori Mood: {mood_label}, Perasaan Spesifik: {emosi_label}" if mood_label else f"Perasaan: {emosi_label}"

    # Generate via Groq
    if not USE_MOCK and client:
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Kamu adalah asisten psikologis virtual untuk aplikasi kesehatan mental mahasiswa. "
                            "Tugasmu memberikan pesan dukungan singkat (maksimal 2 kalimat yang padat dan bermakna). "
                            "DILARANG KERAS menggunakan kata ganti orang pertama (saya, aku, kami). "
                            "ATURAN FEEDBACK:\n"
                            "1. Validasi perasaannya secara spesifik berdasarkan input.\n"
                            "2. Jika emosinya negatif (Sedih, Marah, Takut, Lelah, dll), berikan empati yang dalam, validasi bahwa wajar merasa begitu, dan berikan kalimat menenangkan tanpa *toxic positivity*.\n"
                            "3. Jika emosinya positif (Senang, Antusias, Bangga, dll), ikut rayakan, apresiasi energi tersebut, dan dorong untuk dinikmati.\n"
                            "4. Gunakan gaya bahasa Indonesia yang hangat, personal, santai seperti seorang konselor sebaya."
                        )
                    },
                    {
                        "role": "user",
                        "content": f"Mahasiswa menginput Kategori Mood: '{mood_label}', dengan Perasaan Spesifik: '{emosi_label}'. Berikan pesan feedback langsung yang sangat relevan dengan emosi tersebut."
                    }
                ],
                model=GROQ_MODEL,
                temperature=0.8,
            )
            return chat_completion.choices[0].message.content.strip()
        except Exception as e:
            print(f"Groq error: {e}")
            return _get_mock_fallback(emosi_label)
    else:
        return _get_mock_fallback(emosi_label)

if __name__ == "__main__":
    # Test simulation
    print(f"Quote : {get_recommendation('MHS-001')}")