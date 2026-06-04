import os
from groq import Groq
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Path ke file .env di dalam folder pa3-ai-engine (prioritas utama)
env_path = os.path.join(BASE_DIR, '..', '.env')
if os.path.exists(env_path):
    load_dotenv(dotenv_path=env_path, override=True)
else:
    # Fallback jika tidak ketemu di ../.env, coba di root (jika dijalankan dari pa3-ai-engine)
    load_dotenv()

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
api_key    = os.getenv("GROQ_API_KEY")

def get_groq_client():
    """Lazy initialization of Groq client."""
    global api_key
    if not api_key:
        api_key = os.getenv("GROQ_API_KEY")
    
    if api_key and api_key.startswith("gsk_"):
        return Groq(api_key=api_key)
    return None

def _summarize_with_groq(semua_jurnal: list) -> str:
    """Kirim semua jurnal sekaligus ke Groq LLaMA untuk diringkas."""
    client = get_groq_client()
    if not client:
        return _fallback_summary(semua_jurnal)

    # Format jurnal dengan nomor urut
    formatted = "\n".join([
        f"Catatan {i + 1}: {text.strip()}"
        for i, text in enumerate(semua_jurnal)
    ])

    try:
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Tugasmu: Berikan ringkasan (summary) yang padat dan objektif dari kumpulan jurnal mahasiswa. "
                        "ATURAN KRITIS: "
                        "1. JANGAN MENGULANG atau menuliskan kembali isi jurnal secara harfiah. "
                        "2. JANGAN membuat daftar (list/bullet points). Gunakan format satu paragraf naratif yang mengalir. "
                        "3. Rangkum inti dari apa yang dialami, dilakukan, atau dirasakan mahasiswa secara umum. "
                        "4. JANGAN memberikan saran medis, psikologis, atau langkah intervensi. "
                        "5. Tulis dalam Bahasa Indonesia yang formal, padat, dan langsung ke inti. "
                        "6. WAJIB Sangat Singkat: Maksimal 2-3 kalimat saja."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Berikut adalah {len(semua_jurnal)} entri jurnal mahasiswa:\n\n"
                        f"{formatted}\n\n"
                        "Berikan ringkasan yang sangat padat (2-3 kalimat) mengenai inti dari seluruh jurnal ini."
                    )
                }
            ],
            model=GROQ_MODEL,
            temperature=0.5,
            max_tokens=150,
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"[ERROR] Groq API fail: {e}")
        return _fallback_summary(semua_jurnal)


def _fallback_summary(semua_jurnal: list) -> str:
    """Fallback sederhana jika Groq tidak tersedia."""
    count = len(semua_jurnal)
    cuplikan = " | ".join(semua_jurnal[:2])
    suffix   = "..." if count > 2 else ""
    return (
        f"Analisis otomatis saat ini tidak tersedia. Mahasiswa telah mencatat {count} jurnal. "
        f"Kecenderungan isi: {cuplikan}{suffix}"
    )

def proses_jurnal_sebulan(list_jurnal: list) -> str:
    """Meringkas seluruh jurnal."""
    if not list_jurnal:
        return "Belum ada data jurnal."

    client = get_groq_client()
    if not client:
        print("[WARNING] Groq API Key tidak ditemukan. Menggunakan fallback.")
        return _fallback_summary(list_jurnal)

    print(f"[AI] Menganalisis {len(list_jurnal)} jurnal menggunakan {GROQ_MODEL}...")
    return _summarize_with_groq(list_jurnal)


def generate_monthly_summary(nim: str, journal_texts: list) -> str:
    """Fungsi utama."""
    if not journal_texts:
        return "Mahasiswa ini belum memiliki catatan jurnal."

    return proses_jurnal_sebulan(journal_texts)



# =====================================================================
# SIMULASI — hanya berjalan saat file dijalankan langsung
# =====================================================================
if __name__ == "__main__":
    contoh_jurnal = [
        "Hari ini merasa sangat cemas karena deadline PKM dan tugas akhir makin dekat.",
        "Revisi sistem manajemen ternyata banyak banget. Rasanya capek dan ingin menyerah.",
        "Masih kepikiran soal revisi kemarin. Takut kalau sidang ditanya hal yang belum kupahami.",
        "Akhirnya proposal ACC! Lega banget bisa tidur nyenyak malam ini.",
    ]
    hasil = generate_monthly_summary("MHS-TEST", contoh_jurnal)
    print(f"\nHasil: {hasil}")