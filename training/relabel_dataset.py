import sys
import pandas as pd
import os

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.append(base_dir)

from utils.urgent_label import convert_to_pa3_levels

def assign_new_level(row):
    teks = str(row.get('text_clean', '')).lower()
    label_asli = str(row.get('label', '')).lower()

    # ──────────────────────────────────────────────────────────────────
    # LEVEL 3 — Krisis / Darurat (Rule-Based, Prioritas Tertinggi)
    # Berdasarkan Dataset - Dataset.csv kolom ke-3:
    #   - Ideasi bunuh diri eksplisit
    #   - Keinginan kekerasan terhadap diri/orang lain
    #   - Menyerah pada hidup secara eksplisit
    # ──────────────────────────────────────────────────────────────────
    if convert_to_pa3_levels(teks, label_asli) == 3:
        return 3

    # ──────────────────────────────────────────────────────────────────
    # LEVEL 2 — Perlu Perhatian Serius
    # Berdasarkan Dataset - Dataset.csv kolom ke-3:
    #   - Keputusasaan & kehilangan harapan berat
    #   - Insomnia parah / tidak bisa tidur bermalam-malam
    #   - Frustrasi berat, membenci diri sendiri
    #   - Merasa tidak berharga / tidak berguna
    #   - Pikiran kacau / tidak bisa berpikir jernih
    #   - Hidup/jiwa terasa kosong
    #   - Sudah tidak kuat / tidak sanggup
    # ──────────────────────────────────────────────────────────────────
    level_2_keywords = [
        # Keputusasaan & kehilangan harapan berat
        'putus asa', 'keputusasaan', 'tidak ada harapan', 'kehilangan harapan',
        'kehilangan harapan total', 'tidak kuat lagi', 'sudah tidak kuat',
        'sudah tidak sanggup', 'tidak sanggup lagi', 'mengakhiri semuanya',

        # Gangguan tidur parah
        'insomnia parah', 'tidak bisa tidur bermalam-malam',

        # Self-worth sangat rendah
        'membenci diri sendiri', 'merasa tidak berharga', 'merasa tidak berguna',
        'merasa gagal dalam hidup', 'frustrasi', 'frustasi',

        # Pikiran & kondisi mental berat
        'pikiran kacau', 'tidak bisa berpikir jernih',
        'hidup terasa kosong', 'jiwaku merasa kosong',
        'jiwa kosong', 'hidup kosong',

        # Emosi berat lainnya
        'sangat sedih', 'hancur', 'patah hati', 'terpuruk',
        'sangat cemas', 'ketakutan', 'anxiety parah',
        'burn out total', 'burnout total', 'habis tenaga',

        # Labeling diri negatif berat
        'saya orang gagal', 'saya tidak berguna', 'saya payah',
        'tidak ada jalan', 'tidak ada yang bisa saya lakukan',

        # Orientasi masa depan sangat gelap
        'masa depan saya suram', 'tidak melihat masa depan',
        'semua usaha sia-sia', 'sudah menyerah', 'tidak ada gunanya mencoba lagi',
        'capek hidup', 'jenuh dengan hidup', 'lelah jadi diri sendiri',
        'ingin istirahat selamanya',

        # Isolasi sosial berat
        'tidak ada yang peduli', 'terisolir', 'merasa jadi masalah buat orang sekitar',
        'saya menyusahkan orang lain',
    ]

    # ──────────────────────────────────────────────────────────────────
    # LEVEL 1 — Perlu Pemantauan
    # Berdasarkan Dataset - Dataset.csv kolom ke-3:
    #   - Emosi tidak stabil, sering menangis (tapi bukan krisis)
    #   - Tidak bisa fokus / konsentrasi
    #   - Kehilangan motivasi total
    #   - Hati kosong / hidup datar
    #   - Konflik sosial / berantem
    #   - Gangguan tidur ringan (sulit tidur, tidak tidur sama sekali)
    #   - Kelelahan, capek berat, stres ringan-sedang
    # ──────────────────────────────────────────────────────────────────
    level_1_keywords = [
        # Emosi tidak stabil / menangis (ringan)
        'emosi tidak stabil', 'sering menangis', 'menangis tanpa alasan',
        'nangis tanpa alasan', 'nangis terus', 'menangis',

        # Fokus & konsentrasi terganggu
        'tidak bisa fokus', 'tidak bisa konsentrasi',

        # Kehilangan motivasi
        'kehilangan motivasi total', 'tidak punya motivasi',

        # Hati/hidup terasa kosong & datar (ringan)
        'hati kosong', 'hidup terasa datar', 'tidak ada jiwa',

        # Konflik sosial
        'sering berantem', 'berantem',

        # Gangguan tidur ringan
        'tidak bisa tidur sama sekali', 'sulit tidur terus menerus',
        'tidak tidur bermalam-malam', 'insomnia', 'susah tidur',

        # Emosi campuran ringan
        'sedih', 'murung', 'galau', 'mellow', 'down',
        'khawatir', 'was-was', 'deg-degan', 'gugup', 'cemas',
        'overthinking', 'gelisah', 'kesal', 'sebel', 'jengkel',
        'lelah', 'capek', 'penat', 'exhausted', 'kelelahan', 'kecapekan',

        # Keraguan ringan
        'tidak tahu', 'belum bisa', 'tidak yakin', 'belum siap',
        'kurang mampu', 'masih kesulitan',

        # Sosial merenggang
        'merasa sendirian', 'tidak ada teman', 'kesepian', 'agak kesepian',
        'jarang kumpul', 'sedikit menyendiri',

        # Konteks mahasiswa
        'pusing', 'tugas', 'malas', 'susah', 'sulit', 'dosen',
        'mumet', 'revisi', 'burnout ringan', 'ngantuk',
    ]

    # Cek Level 2
    if any(keyword in teks for keyword in level_2_keywords):
        return 2

    # Cek Level 1
    elif any(keyword in teks for keyword in level_1_keywords):
        return 1

    else:
        # Fallback berdasarkan label sentiment
        if label_asli == 'negative':
            return 1
        else:
            return 0


def main():
    input_file = os.path.join(base_dir, 'data', 'train_final_ready.csv')
    output_file = os.path.join(base_dir, 'training', 'train_dataset_baru.csv')

    print(f"Membaca dataset lama dari: {input_file}...")

    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print("Data tidak ditemukan! Pastikan path benar.")
        return

    print("Memproses relabeling dataset menjadi Level 0, 1, 2, 3...")
    print("(Sesuai definisi dari Dataset - Dataset.csv kolom ke-3)")
    df['final_level'] = df.apply(assign_new_level, axis=1)

    os.makedirs(os.path.join(base_dir, 'training'), exist_ok=True)

    df.to_csv(output_file, index=False)

    print(f"\nRelabeling selesai!")
    print(f"Dataset baru tersimpan di: {output_file}")

    print("\n" + "=" * 55)
    print("Rangkuman Hasil Relabeling (sesuai Dataset - Dataset.csv)")
    print("=" * 55)
    summary = df['final_level'].value_counts().sort_index()
    label_names = {
        0: "Level 0 — Normal / Positif",
        1: "Level 1 — Perlu Pemantauan",
        2: "Level 2 — Perlu Perhatian Serius",
        3: "Level 3 — Krisis / Darurat (Rule-Based)"
    }
    for lvl, count in summary.items():
        print(f"  {label_names.get(lvl, f'Level {lvl}')}: {count} data")
    print("=" * 55)


if __name__ == "__main__":
    main()
