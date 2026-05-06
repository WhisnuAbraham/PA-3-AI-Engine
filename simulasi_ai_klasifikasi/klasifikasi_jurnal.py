import os
import sys
import re
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, BertForSequenceClassification

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

from utils.urgent_label import convert_to_pa3_levels

print("Memuat hasil training model")
model_path = os.path.join(base_dir, "pa3_indobert_final")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model     = BertForSequenceClassification.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
print(f"IndoBERT siap di device: {device}")

level_map = {
    0: "Level 0 (Positif / Baik)",
    1: "Level 1 (Ekspresi Emosi Ringan)",
    2: "Level 2 (Perlu Pemantauan)"
}

red_flags = [
    # Keputusasaan
    "merasa putus asa", "kehilangan harapan", "tidak ada harapan",
    "tidak kuat lagi", "sudah tidak sanggup",
    # Gangguan tidur / fisik berat
    "tidak bisa tidur", "insomnia", "tidak tidur sama sekali",
    "sulit tidur terus menerus",
    # Kondisi mental berat
    "merasa kosong", "hati kosong", "hidup terasa kosong", "jiwa kosong",
    "pikiran kacau", "tidak bisa berpikir jernih",
    "emosi tidak stabil", "sering menangis",
    "berantem",
    # Kehilangan fungsi
    "tidak bisa fokus", "tidak bisa konsentrasi",
    "kehilangan motivasi total", "tidak bisa menjalani aktivitas",
    "tidak punya energi",
    # Self-worth rendah
    "membenci diri sendiri", "merasa tidak berharga",
    "merasa gagal", "merasa tidak berguna", "frustrasi", "frustasi",
    # Bahaya (PALING KRITIS)
    "ingin menyerah", "putus asa", "ingin mati",
    "bunuh diri", "mau mati", "mending mati", "lebih baik mati",
    "mampus", "mau mampus", "sekalian mati",
    "insomnia parah", "tidak bisa tidur bermalam-malam",
    "merasa frustrasi", "merasa frustasi"
]

def _is_negated(text: str, phrase: str) -> bool:
    pattern = rf"\b(tidak|bukan|ga|gak)\b(?:\W+\w+){{0,3}}\W+{re.escape(phrase)}\b"
    return re.search(pattern, text) is not None


#fungsi utama
def classify_text(text: str) -> dict:
    """
    Klasifikasi teks jurnal ke level kesehatan mental (1–3).

    Returns:
        dict: { level, label, confidence, red_flag }
    """
    text_lower = str(text).lower()

    # Cek red flag terlebih dahulu (rule-based, prioritas tertinggi)
    found_flags = []
    for flag in red_flags:
        if flag in text_lower and not _is_negated(text_lower, flag):
            found_flags.append(flag)
            
    if found_flags:
        return {
            "level":      3,
            "label":      "Level 3 (Krisis / Butuh Penanganan Cepat)",
            "confidence": 100.0,
            "red_flag":   ", ".join(found_flags)
        }

    # Jika tidak ada red flag, gunakan IndoBERT
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs      = model(**inputs)
        probs        = F.softmax(outputs.logits, dim=-1)
        confidence   = torch.max(probs).item() * 100
        pred_class   = torch.argmax(outputs.logits, dim=-1).item()

    return {
        "level":      pred_class, # Sekarang output kelas IndoBERT murni 0, 1, atau 2
        "label":      level_map[pred_class],
        "confidence": round(confidence, 2),
        "red_flag":   None
    }


def prediksi_mental(kalimat: str):
    """Wrapper untuk testing lokal — cetak hasil ke console."""
    result = classify_text(kalimat)
    print("\n" + "=" * 50)
    print(f"Teks      : '{kalimat}'")
    print(f"Analisis  : {result['label']}")
    if result['red_flag']:
        print(f"Yakin     : 100.00% (Red Flag Detected)")
        print(f"Red Flag  : '{result['red_flag']}'")
    else:
        print(f"Yakin     : {result['confidence']:.2f}%")
    print("=" * 50)

#simulasi
if __name__ == "__main__":
    tes_1 = "Wah, hebat banget ya dosen ngasih tugas segini banyak, sekalian aja cabut nyawa saya."
    tes_2 = "Tugas banyak, mau nyerah aja tapi masih kuat"
    tes_3 = "Proposal anjing, mampus aja sekalian mau mati cok gabisa ngerjain nya banyak revisi"

    prediksi_mental(tes_1)
    prediksi_mental(tes_2)
    prediksi_mental(tes_3)