import re
def is_negated(text, phrase):
    pattern = rf"\b(tidak|bukan|ga|gak)\b(?:\W+\w+){{0,3}}\W+{re.escape(phrase)}\b"
    return re.search(pattern, text) is not None


def convert_to_pa3_levels(text, label):
    text = str(text).lower()

    red_flags = [
    #KRISIS/IDEASI BUNUH DIRI
    "ingin mati",
    "bunuh diri",
    "mau mati",
    "mending mati",
    "lebih baik mati",
    "mampus",
    "mau mampus",
    "sekalian mati",
    "tidak mau hidup lagi",

    #SELF-WORTH SANGAT RENDAH
    "merasa tidak berharga",

    #GANGGUAN TIDUR
    "tidak bisa tidur",
    "insomnia",
    "insomnia parah",
    "tidak tidur sama sekali",
    "sulit tidur terus menerus",
    "tidak bisa tidur bermalam-malam",

    #GANGGUAN EMOSI & KOGNITIF
    "emosi tidak stabil",
    "tidak bisa fokus",
    "tidak bisa konsentrasi",

    #GANGGUAN EMOSI BERKELANJUTAN
    "menangis terus setiap hari",
    "menangis setiap hari",
    "nangis terus setiap hari",
    "nangis setiap hari",
]
    for flag in red_flags:
        if flag in text and not is_negated(text, flag):
            return 3

    if label == 'negative':
        return 2

    return 1
