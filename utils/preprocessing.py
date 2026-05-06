import pandas as pd
import re
import textwrap
from utils.urgent_label import convert_to_pa3_levels
from utils.slang_mapping import apply_slang_mapping

# Load Dataset 
train_df = pd.read_csv('data/train_preprocess_ori.tsv', sep='\t', names=['text', 'label'], header=None, skiprows=[0])
valid_df = pd.read_csv('data/valid_preprocess.tsv', sep='\t', names=['text', 'label'], header=None, skiprows=[0])

def clean_noise_and_slang(text):
    text = str(text).lower() 
    text = re.sub(r'http\S+|@\S+|#\S+|[^\w\s]', '', text)
    text = apply_slang_mapping(text)
    return text.strip()

# Terapkan pembersihan
train_df['text_clean'] = train_df['text'].apply(clean_noise_and_slang)
valid_df['text_clean'] = valid_df['text'].apply(clean_noise_and_slang)

# Konversi ke PA3 Levels
train_df['final_level'] = train_df.apply(lambda x: convert_to_pa3_levels(x['text_clean'], x['label']), axis=1)
valid_df['final_level'] = valid_df.apply(lambda x: convert_to_pa3_levels(x['text_clean'], x['label']), axis=1)

# CEK DISTRIBUSI DATA
print("Distribusi Level pada Data Training:")
print(train_df['final_level'].value_counts())

print("\n" + "="*80)
print("CONTOH DATA - LEVEL 1 (BAIK) - 5 JUMLAH:")
print("="*80)
for idx, row in train_df[train_df['final_level'] == 1][['text_clean', 'final_level']].head(5).iterrows():
    print(f"Text: {textwrap.fill(row['text_clean'], width=80)}")
    print(f"Level: {row['final_level']}")
    print("-" * 40)

print("\n" + "="*80)
print("CONTOH DATA - LEVEL 2 (PERLU PEMANTAUAN) - 5 JUMLAH:")
print("="*80)
for idx, row in train_df[train_df['final_level'] == 2][['text_clean', 'final_level']].head(5).iterrows():
    print(f"Text: {textwrap.fill(row['text_clean'], width=80)}")
    print(f"Level: {row['final_level']}")
    print("-" * 40)

print("\n" + "="*80)
print("CONTOH DATA - LEVEL 3 (PERLU PENANGANAN) - 5 JUMLAH:")
print("="*80)
for idx, row in train_df[train_df['final_level'] == 3][['text_clean', 'final_level']].head(5).iterrows():
    print(f"Text: {textwrap.fill(row['text_clean'], width=80)}")
    print(f"Level: {row['final_level']}")
    print("-" * 40)

# BALANCING DATASET
# 1. Cek nilai asli label dari Kaggle
print("\n" + "="*80)
print("NILAI ASLI LABEL DARI KAGGLE:")
print("="*80)
print("Nilai asli label di dataset:")
print(train_df['label'].value_counts())

# 2. Undersampling (mengurangi Level 1)
# Mengambil hanya 1000 data dari Level 1 agar seimbang
level_1 = train_df[train_df['final_level'] == 1].sample(n=1000, random_state=42)
level_2 = train_df[train_df['final_level'] == 2]
level_3 = train_df[train_df['final_level'] == 3]

# 3. Gabungkan kembali
train_balanced = pd.concat([level_1, level_2, level_3])

print("\nDistribusi Baru setelah Balancing:")
print(train_balanced['final_level'].value_counts())

# INTEGRASIKAN DENGAN DATA RED FLAG
# 1. Baca file Red Flag yang sudah dibuat dari generator
red_flag_df = pd.read_csv('data/data Red Flag.csv')

# 2. Ambil sampel untuk menyeimbangkan (Undersampling)
# Kita ambil 1200 data dari Level 1 dan semua Level 2 yang tersedia
level_1_balanced = train_df[train_df['final_level'] == 1].sample(n=min(1200, len(train_df[train_df['final_level'] == 1])), random_state=42)
level_2_balanced = train_df[train_df['final_level'] == 2]

# 3. Ambil Level 3 (Hasil mapping otomatis + Hasil generator)
level_3_auto = train_df[train_df['final_level'] == 3]

# Siapkan data Red Flag untuk digabung
level_3_manual = red_flag_df.copy()
level_3_manual['text_clean'] = level_3_manual['text']
level_3_manual['final_level'] = 3
level_3_manual = level_3_manual[['text_clean', 'final_level']]

# 4. GABUNGKAN SEMUANYA
train_final = pd.concat([level_1_balanced, level_2_balanced, level_3_auto[['text_clean', 'final_level']], level_3_manual]).reset_index(drop=True)

# 5. Simpan dataset yang sudah siap untuk Training
train_final.to_csv('data/train_final_ready.csv', index=False)

print("\n" + "="*80)
print("DISTRIBUSI DATA AKHIR UNTUK TRAINING:")
print("="*80)
print(train_final['final_level'].value_counts())
print(f"\nTotal data siap pakai: {len(train_final)} baris")