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
    
    #Cek Level 3
    #
    if convert_to_pa3_levels(teks, label_asli) == 3:
        return 3

    # Kata kunci Level 2 (Perlu Perhatian)
    level_2_keywords = [
        # Emosi & Kondisi Berat
        'sangat sedih', 'menangis', 'nangis terus', 'hancur', 'patah hati', 'putus asa', 'tidak kuat', 'menyesal', 'pilu', 'terpuruk',
        'sangat cemas', 'panik', 'ketakutan', 'tidak tenang', 'susah tidur karena pikiran', 'pikiran tidak bisa berhenti', 'anxiety parah',
        'marah', 'frustrasi', 'benci', 'muak', 'gerah', 'emosi', 'sakit hati', 'dendam', 'gondok',
        'sangat kelelahan', 'tidak ada tenaga', 'mati-matian tapi tidak ada hasil', 'habis tenaga', 'burn out total', 'burnout total',
        
        # Labeling Diri Negatif & Kemustahilan
        'saya ini bodoh', 'saya orang gagal', 'saya tidak berguna', 'saya payah', 'saya memang tidak bisa', 'saya lemah',
        'tidak bisa', 'tidak mampu', 'tidak sanggup', 'tidak mungkin', 'tidak ada harapan', 'tidak akan bisa', 'tidak ada jalan',
        'tidak bisa apa-apa', 'tidak ada yang bisa saya lakukan', 'kemampuan saya tidak cukup', 'saya tidak kompeten sama sekali',
        
        # Orientasi Masa Depan Gelap & Distorsi Kognitif
        'tidak yakin masa depan saya', 'tidak ada yang bisa diharapkan', 'masa depan saya suram', 'tidak melihat masa depan',
        'dari dulu selalu begini', 'saya sudah sering gagal', 'semua yang sudah saya usahakan tidak ada hasilnya', 'dulu juga begini',
        'sekarang tidak ada semangat sama sekali', 'hari-hari terasa berat', 'saat ini tidak bisa ngapa-ngapain', 'stuck banget',
        'selalu gagal', 'tidak pernah berhasil', 'semua usaha sia-sia', 'setiap kali pasti salah', 'tidak pernah benar',
        'saya memang orangnya begini', 'saya dari dulu selalu gagal', 'semua orang lebih baik dari saya', 'saya selalu kalah',
        'mustahil buat saya', 'tidak mungkin saya bisa', 'sudah terlambat untuk berubah', 'percuma dicoba',
        'sudah menyerah', 'tidak ada gunanya mencoba lagi', 'percuma dipikirin', 'pasrah saja', 'sudah capek berjuang',
        
        # Isolasi Sosial & Merasa Jadi Beban
        'tidak ada teman yang mengerti', 'merasa sendirian', 'tidak ada yang peduli', 'orang-orang tidak mau tahu', 'asing di lingkungan',
        'sendiri terus', 'tidak ada teman', 'merasa asing di antara orang banyak', 'tidak ada yang mengerti perasaan saya', 'terisolir',
        'saya menyusahkan orang lain', 'sering merepotkan teman', 'merasa jadi masalah buat orang sekitar',
        
        # Risiko Menghindar
        'ingin menghilang sejenak', 'ingin lari dari semua ini', 'tidak tahan lagi', 'ingin istirahat selamanya',
        'jenuh dengan hidup', 'capek hidup', 'lelah jadi diri sendiri',
        
        # Tambahan dari list sebelumnya
        'hampa', 'sendirian', 'berat banget', 'gak sanggup', 'depresi', 'kesepian'
    ]

    # Kata kunci Level 1 (Perlu Pemantauan)
    level_1_keywords = [
        # Emosi Campuran/Ringan
        'lumayan', 'cukup senang', 'sedikit lega', 'masih oke', 'semoga bisa', 'insya allah', 'boleh lah',
        'sedih', 'murung', 'galau', 'nelangsa', 'mellow', 'down', 'terpuruk sedikit', 'agak hancur',
        'khawatir', 'was-was', 'deg-degan', 'gugup', 'cemas', 'takut gagal', 'grogi', 'overthinking', 'gelisah',
        'kesal', 'sebel', 'jengkel', 'annoying', 'bt', 'tidak sabar', 'dongkol', 'cranky',
        'lelah', 'capek', 'penat', 'exhausted', 'kelelahan', 'kecapekan', 'ngantuk parah', 'burnout ringan', 'ngantuk',
        
        # Keraguan & Negasi Ringan
        'saya ini memang begini', 'saya kurang pintar', 'saya masih belajar',
        'tidak tahu', 'belum bisa', 'tidak yakin', 'kurang paham', 'belum siap', 'tidak terlalu', 'kurang begitu',
        'tidak begitu bisa', 'kurang mampu', 'masih kesulitan', 'belum cukup', 'kurang kompeten',
        'tidak tahu nanti bagaimana', 'belum ada rencana', 'mudah-mudahan bisa', 'entah nanti',
        
        # Waktu & Kognitif Ringan
        'kemarin gagal', 'minggu lalu tidak berhasil', 'dulu pernah susah juga',
        'sekarang agak bingung', 'saat ini lagi tidak fokus', 'hari ini capek banget', 'lagi kurang mood',
        'sering', 'kadang-kadang', 'biasanya', 'rata-rata', 'kebanyakan', 'sepertinya saya memang begini', 'mungkin saya kurang beruntung',
        'susah banget', 'tampaknya sulit', 'agak mustahil tapi dicoba', 'mungkin tidak bisa',
        'mungkin', 'sepertinya', 'kayaknya', 'entah', 'tidak begitu yakin', 'kurang yakin', 'ragu-ragu', 'sepertinya begitu',
        
        # Sosial Merenggang
        'teman ada tapi', 'kadang sendiri', 'kurang kumpul', 'jarang bareng', 'kurang dekat dengan orang-orang',
        'jarang kumpul belakangan', 'agak jauh', 'agak kesepian', 'kurang bersosialisasi', 'jarang ketemu orang belakangan', 'sedikit menyendiri',
        
        # Konteks Mahasiswa & Tambahan
        'pusing', 'tugas', 'malas', 'susah', 'sulit', 'dosen', 'mumet', 'revisi'
    ]
    
    # Cek Level 2
    if any(keyword in teks for keyword in level_2_keywords):
        return 2
        
    # Cek Level 1 
    elif any(keyword in teks for keyword in level_1_keywords):
        return 1

    else:
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

    print("Memproses relabeling dataset menjadi Kelas 0, 1, 2")
    df['final_level'] = df.apply(assign_new_level, axis=1)

    os.makedirs(os.path.join(base_dir, 'training'), exist_ok=True)

    df.to_csv(output_file, index=False)
    
    print(f"Relabeling selesai!")
    print(f"Dataset baru tersimpan di: {output_file}")

    print("\nRangkuman Hasil Relabeling")
    print(df['final_level'].value_counts().sort_index().rename(index={
        0: "Level 0 (Positif/Netral)", 
        1: "Level 1 (Stres Ringan)", 
        2: "Level 2 (Lebih Buruk - BUKAN RED FLAG)",
        3: "Level 3 (RED FLAG/Darurat/Rule-Based)"
    }))

if __name__ == "__main__":
    main()
