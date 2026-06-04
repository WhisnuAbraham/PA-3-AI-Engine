import re

def is_negated(text, phrase):
    pattern = rf"\b(tidak|bukan|ga|gak)\b(?:\W+\w+){{0,3}}\W+{re.escape(phrase)}\b"
    return re.search(pattern, text) is not None


def convert_to_pa3_levels(text, label):
    """
    Rule-based level detection berdasarkan Dataset - Dataset.csv (kolom ke-3).

    Level 3 — Krisis / Darurat:
        - Ideasi bunuh diri eksplisit
        - Keinginan kekerasan terhadap diri sendiri atau orang lain
        - Menyerah pada hidup secara eksplisit

    Level 2 — Perlu Perhatian Serius:
        - Keputusasaan & kehilangan harapan berat
        - Gangguan tidur parah (insomnia)
        - Self-worth sangat rendah
        - Pikiran kacau / tidak bisa berpikir jernih
        - Perasaan hampa / hidup kosong / tidak berharga

    Level 1 — Perlu Pemantauan:
        - Emosi tidak stabil, sering menangis
        - Tidak bisa fokus / konsentrasi
        - Kehilangan motivasi total
        - Hati kosong / hidup datar
        - Konflik sosial / berantem
        - Gangguan tidur ringan

    Level 0 — Normal / Positif:
        - Lelah biasa, sedikit sedih, capek tugas
        - Ekspresi positif: damai, bersyukur, bahagia, semangat, produktif
    """
    text = str(text).lower()

    # ──────────────────────────────────────────────────────────────────────
    # LEVEL 3: Ideasi bunuh diri, kekerasan, menyerah pada hidup
    # ──────────────────────────────────────────────────────────────────────
    red_flags_level3 = [
        # Ideasi bunuh diri eksplisit
        "ingin mati",
        "mau mati",
        "mending mati",
        "lebih baik mati",
        "sekalian mati",
        "mampus",
        "mau mampus",
        "bunuh diri",
        "tidak mau hidup lagi",
        "mengakhiri hidup",
        "mengakhiri semuanya",

        # Menyerah pada hidup secara eksplisit
        "ingin menyerah pada hidup",
        "tidak ada gunanya bertahan",
        "tidak ada gunanya hidup",
        "menyerah pada hidup",

        # Keinginan kekerasan / menyakiti
        "ingin menyakiti",
        "tidak bisa mengontrol keinginan untuk berbuat kekerasan",
        "akan benar-benar melukai seseorang",
        "melukai seseorang",
        "berbuat kekerasan",
    ]

    for flag in red_flags_level3:
        if flag in text and not is_negated(text, flag):
            return 3

    # ──────────────────────────────────────────────────────────────────────
    # LEVEL 2: Keputusasaan berat, self-worth sangat rendah, insomnia parah
    # ──────────────────────────────────────────────────────────────────────
    red_flags_level2 = [
        # Keputusasaan & kehilangan harapan berat
        "tidak ada harapan",
        "kehilangan harapan total",
        "keputusasaan",
        "tidak kuat lagi",
        "sudah tidak kuat",
        "sudah tidak sanggup",
        "tidak sanggup lagi",

        # Self-worth sangat rendah
        "membenci diri sendiri",
        "merasa tidak berharga",
        "merasa tidak berguna",
        "merasa gagal dalam hidup",
        "merasa gagal",
        "frustasi",
        "frustrasi",

        # Gangguan tidur parah
        "insomnia parah",
        "tidak bisa tidur bermalam-malam",

        # Perasaan hampa / hidup kosong
        "hidup terasa kosong",
        "jiwaku merasa kosong",
        "pikiran kacau",
        "tidak bisa berpikir jernih",
    ]

    for flag in red_flags_level2:
        if flag in text and not is_negated(text, flag):
            return 2

    # ──────────────────────────────────────────────────────────────────────
    # LEVEL 1: Emosi tidak stabil, gangguan tidur ringan, motivasi turun
    # ──────────────────────────────────────────────────────────────────────
    flags_level1 = [
        # Emosi tidak stabil / menangis
        "emosi tidak stabil",
        "sering menangis",
        "menangis tanpa alasan",
        "nangis tanpa alasan",

        # Fokus & konsentrasi terganggu
        "tidak bisa fokus",
        "tidak bisa konsentrasi",

        # Kehilangan motivasi
        "kehilangan motivasi total",
        "tidak punya motivasi",

        # Hati / hidup terasa kosong (lebih ringan dari Level 2)
        "hati kosong",
        "hidup terasa datar",
        "tidak ada jiwa",

        # Konflik sosial
        "sering berantem",
        "berantem",

        # Gangguan tidur ringan
        "tidak bisa tidur sama sekali",
        "sulit tidur terus menerus",
        "tidak tidur bermalam-malam",
    ]

    for flag in flags_level1:
        if flag in text and not is_negated(text, flag):
            return 1

    # ──────────────────────────────────────────────────────────────────────
    # Fallback berdasarkan label sentiment asli
    # ──────────────────────────────────────────────────────────────────────
    if label == 'negative':
        return 1

    return 0
