from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel  
from simulasi_ai_jurnal.ai_generator import get_ai_response          
from simulasi_ai_klasifikasi.klasifikasi_jurnal import classify_text                                   
from simulasi_ai_recommender.recommender_system import get_recommendation 
from simulasi_ai_predictive.predictive_analytics import run_clinical_analysis, evaluate_predictive_risk  
from summary.summarizer_bulanan import generate_monthly_summary       

app = FastAPI(title="Del Care AI Engine", version="1.0")

class MoodFeelingEntry(BaseModel):
    mood: str
    feeling: str

class JournalInput(BaseModel):
    nim: str
    text: Optional[str] = ""
    mood_history: Optional[list[MoodFeelingEntry]] = []
    days_since_last_journal: Optional[int] = 0

class EmotionInput(BaseModel):
    nim: str
    text: str
    emotion: Optional[str] = None   

class UserInput(BaseModel):
    nim: str

class SummarizeInput(BaseModel):
    nim: str
    journal_texts: list[str] 

@app.post("/api/classify")
async def classify_journal(data: JournalInput):
    """
    Input : nim, text (isi jurnal), mood_history (list of mood/feeling names), days_since_last_journal
    Output: level (0-3), label, confidence, red_flag
    """
    # 1. Base Classification (Text)
    if data.text and data.text.strip():
        result = classify_text(data.text)
    else:
        # Default jika mahasiswa tidak kirim teks
        result = {
            "level": 0,
            "label": "Level 0 (Aman / Tidak ada Jurnal)",
            "confidence": 100.0,
            "red_flag": "Mahasiswa tidak menulis jurnal hari ini"
        }
        
    # 2. Predictive Analytics Check (History of Mood/Feelings)
    # Convert mood_history (names) to scores using logic from predictive_analytics.py
    from simulasi_ai_predictive.predictive_analytics import get_combined_score, evaluate_predictive_risk
    
    recent_scores = [get_combined_score(entry.mood, entry.feeling) for entry in data.mood_history]
    
    is_journal_empty = not (data.text and data.text.strip())
    predictive_alert = evaluate_predictive_risk(recent_scores, data.days_since_last_journal, is_journal_empty)
    
    if predictive_alert["is_high_risk"]:
        # Logic Layered: Jika tren menunjukkan krisis (Level 3), override level jurnal
        result["level"] = 3
        result["label"] = "Level 3 (Krisis / Deteksi Tren Menurun)"
        
        # Gabungkan alasan deteksi ke dalam red_flag
        existing_flags = result.get("red_flag", "")
        new_flag = f"[PREDICTIVE AI]: {predictive_alert['reason']}"
        
        if existing_flags and existing_flags != "None":
             result["red_flag"] = f"{existing_flags} | {new_flag}"
        else:
             result["red_flag"] = new_flag

    return {"status": "success", "nim": data.nim, "data": result}


@app.post("/api/generate-popup")
async def generate_popup(data: EmotionInput):
    """
    Input : nim, text (isi jurnal), emotion (opsional)
    Output: reply — pesan penyemangat 1-2 kalimat
    """
    reply = get_ai_response(data.text, data.emotion)
    return {"status": "success", "nim": data.nim, "reply": reply}

@app.post("/api/recommend")
async def recommend_quote(data: UserInput):
    """
    Input : nim
    Output: quote — pesan motivasi berdasarkan emosi terakhir mahasiswa
    """
    quote = get_recommendation(data.nim)
    return {"status": "success", "nim": data.nim, "quote": quote}


@app.get("/api/predictive-radar")
async def predictive_radar():
    """
    Input : -
    Output: alerts — list status semua mahasiswa (HIGH_RISK / MODERATE_RISK / NORMAL)
    """
    alerts = run_clinical_analysis()
    return {"status": "success", "alerts": alerts}


@app.post("/api/summarize")
async def monthly_summary(data: SummarizeInput):
    """
    Input : nim, journal_texts[] — teks jurnal dikirim dari Laravel (hasil query DB)
    Output: summary — ringkasan kondisi mahasiswa berdasarkan jurnal asli dari database
    """
    summary = generate_monthly_summary(data.nim, data.journal_texts)
    return {"status": "success", "nim": data.nim, "summary": summary}


@app.get("/api/ping")
async def cek_koneksi():
    return {"status": "success", "message": "Halo Laravel! Mesin AI Python sudah menyala dan siap menerima perintah."}