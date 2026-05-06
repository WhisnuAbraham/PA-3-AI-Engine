import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

#Konfigurasi
MODEL_NAME = "indobenchmark/indobert-base-p1"
CSV_PATH = "train_dataset_baru.csv" 
OUTPUT_DIR = "./pa3_indobert_final_v2"

print("1. Membaca Dataset...")
df = pd.read_csv(CSV_PATH)

df = df[df['final_level'] != 3]

texts = df['text_clean'].astype(str).tolist()
labels = df['final_level'].astype(int).tolist()

print(f"Total data (Label 0, 1, 2): {len(texts)}")

print("2. Membagi Data Training & Validasi")
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# Load Tokenizer & Model IndoBERT
print("3. Memuat Model IndoBERT")
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

#okenisasi Dataset
print("4. Tokenisasi Teks")
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)

class IndoBERTDateset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = IndoBERTDateset(train_encodings, train_labels)
val_dataset = IndoBERTDateset(val_encodings, val_labels)

print("5. Menyiapkan Konfigurasi Pelatihan")
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,             
    per_device_train_batch_size=16,  
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    evaluation_strategy="epoch",     
    save_strategy="epoch",
    load_best_model_at_end=True     
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

print("6. Memulai Proses Training")
trainer.train()

print(f"7. Menyimpan model final ke folder: {OUTPUT_DIR}")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Selesai! Silakan download folder pa3_indobert_final_v2 ke laptop Anda.")
