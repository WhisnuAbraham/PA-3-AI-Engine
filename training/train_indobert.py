import pandas as pd
import torch
import torch.nn as nn
from collections import Counter
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

# ─────────────────────────────────────────────
# Konfigurasi
# ─────────────────────────────────────────────
MODEL_NAME   = "indobenchmark/indobert-base-p1"
CSV_PATH     = "train_dataset_baru.csv"
OUTPUT_DIR   = r"E:\kuliah\AI PA3\pa3_indobert_final_v2"
RESULTS_DIR  = r"E:\kuliah\AI PA3\training\results"
LOGS_DIR     = r"E:\kuliah\AI PA3\training\logs"
NUM_LABELS   = 4          # Level 0, 1, 2, 3
MAX_WEIGHT   = 30.0       # Batas atas bobot agar Level 3 tidak overfit

# ─────────────────────────────────────────────
# 1. Membaca Dataset
# ─────────────────────────────────────────────
print("1. Membaca Dataset...")
df = pd.read_csv(CSV_PATH)

# Gunakan semua level: 0 (Normal), 1 (Pemantauan), 2 (Perhatian Serius), 3 (Krisis)
texts  = df['text_clean'].astype(str).tolist()
labels = df['final_level'].astype(int).tolist()

print(f"   Total data (Label 0, 1, 2, 3): {len(texts)}")

# Tampilkan distribusi kelas
label_counts = Counter(labels)
print("\n   Distribusi kelas:")
level_names = {0: "Normal/Positif", 1: "Perlu Pemantauan", 2: "Perlu Perhatian", 3: "Krisis/Darurat"}
for lvl in sorted(label_counts):
    print(f"   Level {lvl} ({level_names[lvl]}): {label_counts[lvl]} data")

# ─────────────────────────────────────────────
# 2. Hitung Class Weights untuk Weighted Loss
# ─────────────────────────────────────────────
print("\n2. Menghitung Class Weights (Weighted Loss)...")
total_samples = len(labels)
class_weights = []

for lvl in range(NUM_LABELS):
    count = label_counts.get(lvl, 1)                          # hindari division by zero
    weight = total_samples / (NUM_LABELS * count)              # rumus invers frekuensi
    weight = min(weight, MAX_WEIGHT)                           # cap agar tidak terlalu ekstrem
    class_weights.append(weight)
    print(f"   Level {lvl}: {count} data → bobot = {weight:.2f}")

class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)
print(f"\n   Class weights tensor: {class_weights_tensor.tolist()}")

# ─────────────────────────────────────────────
# 3. Membagi Data Training & Validasi
# ─────────────────────────────────────────────
print("\n3. Membagi Data Training & Validasi (80:20)...")
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels  # stratify agar distribusi seimbang
)
print(f"   Training : {len(train_texts)} data")
print(f"   Validasi : {len(val_texts)} data")

# ─────────────────────────────────────────────
# 4. Load Tokenizer & Model IndoBERT
# ─────────────────────────────────────────────
print("\n4. Memuat Model IndoBERT...")
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model     = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)

# ─────────────────────────────────────────────
# 5. Tokenisasi Dataset
# ─────────────────────────────────────────────
print("5. Tokenisasi Teks...")
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
val_encodings   = tokenizer(val_texts,   truncation=True, padding=True, max_length=128)

class IndoBERTDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels    = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = IndoBERTDataset(train_encodings, train_labels)
val_dataset   = IndoBERTDataset(val_encodings,   val_labels)

# ─────────────────────────────────────────────
# 6. Weighted Trainer — Override compute_loss
# ─────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_weights_tensor = class_weights_tensor.to(device)

# Loss function dengan bobot per kelas
weighted_loss_fn = nn.CrossEntropyLoss(weight=class_weights_tensor)

class WeightedTrainer(Trainer):
    """
    Custom Trainer yang menggunakan Weighted CrossEntropyLoss.
    Kelas langka (Level 2 & 3) diberi penalty lebih besar saat salah prediksi,
    sehingga model lebih waspada terhadap kondisi krisis mahasiswa.
    """
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.get("labels")
        outputs = model(**inputs)
        logits  = outputs.get("logits")

        loss = weighted_loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss

# ─────────────────────────────────────────────
# 7. Konfigurasi Training
# ─────────────────────────────────────────────
import os
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

print("\n6. Menyiapkan Konfigurasi Pelatihan...")
training_args = TrainingArguments(
    output_dir                  = RESULTS_DIR,
    num_train_epochs            = 5,               # Dinaikkan dari 3 → 5 agar kelas langka lebih terpelajari
    per_device_train_batch_size = 16,
    per_device_eval_batch_size  = 16,
    warmup_steps                = 300,
    weight_decay                = 0.01,
    logging_dir                 = LOGS_DIR,
    logging_steps               = 50,
    evaluation_strategy         = "epoch",
    save_strategy               = "no",            # Hindari PermissionError rename checkpoint di Windows
    load_best_model_at_end      = False,            # Harus False jika save_strategy="no"
)

trainer = WeightedTrainer(
    model         = model,
    args          = training_args,
    train_dataset = train_dataset,
    eval_dataset  = val_dataset,
)

# ─────────────────────────────────────────────
# 8. Mulai Training
# ─────────────────────────────────────────────
print("7. Memulai Proses Training (dengan Weighted Loss)...")
print(f"   Device: {device}")
trainer.train()

# ─────────────────────────────────────────────
# 9. Simpan Model Final
# ─────────────────────────────────────────────
print(f"\n8. Menyimpan model final ke folder: {OUTPUT_DIR}")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("\n" + "=" * 55)
print("Selesai! Training dengan Weighted Loss berhasil.")
print(f"Model tersimpan di folder: {OUTPUT_DIR}")
print("Silakan pindahkan folder tersebut ke laptop Anda.")
print("=" * 55)
