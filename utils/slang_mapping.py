"""
Slang Mapping Module
Mengkonversi kata-kata gaul/informal/dialek lokal ke kata baku
Tujuan: Standardisasi teks sebelum masuk ke model BERT
"""

# Kamus kata gaul (Slang Mapping)
# Tambahkan sesuai kebutuhan mahasiswa lokal + dialek Toba/Batak
slang_dict = {
    # ===== UMUM =====
    "gk": "tidak", 
    "gak": "tidak", 
    "udh": "sudah", 
    "sdh": "sudah", 
    "bgt": "banget",
}


def apply_slang_mapping(text):
    """
    Terapkan slang mapping ke teks
    
    Args:
        text (str): Teks yang akan dimapping
    
    Returns:
        str: Teks setelah slang mapping diterapkan
    """
    words = text.split()
    fixed_words = [slang_dict.get(w, w) for w in words]
    return " ".join(fixed_words).strip()
