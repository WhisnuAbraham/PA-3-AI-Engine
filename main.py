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

class RecommendInput(BaseModel):
    nim: str
    mood: Optional[str] = None
    feeling: Optional[str] = None

class SummarizeInput(BaseModel):
    nim: str
    journal_texts: list[str] 

@app.post("/api/classify")
async def classify_journal(data: JournalInput):
    """
    Input : nim, text (isi jurnal), mood_history (list of mood/feeling names), days_since_last_journal
    Output: level (0-3), label, confidence, red_flag
    """
    if data.text and data.text.strip():
        result = classify_text(data.text)
    else:
        result = {
            "level": 0,
            "label": "Level 0 (Aman / Tidak ada Jurnal)",
            "confidence": 100.0,
            "red_flag": "Mahasiswa tidak menulis jurnal hari ini"
        }
        
    from simulasi_ai_predictive.predictive_analytics import get_combined_score, evaluate_predictive_risk
    
    recent_scores = [get_combined_score(entry.mood, entry.feeling) for entry in data.mood_history]
    
    is_journal_empty = not (data.text and data.text.strip())
    predictive_alert = evaluate_predictive_risk(recent_scores, data.days_since_last_journal, is_journal_empty)
    
    pred_level = predictive_alert.get("predictive_level", 0)
    current_level = result.get("level", 0)
    
    if pred_level > current_level:
        result["level"] = pred_level
        
        if pred_level == 3:
            result["label"] = "Level 3 (Krisis / Deteksi Tren Menurun)"
        elif pred_level == 2:
            result["label"] = "Level 2 (Perhatian Serius / Deteksi Tren Menurun)"
        elif pred_level == 1:
            result["label"] = "Level 1 (Pemantauan / Deteksi Tren Menurun)"
        
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
async def recommend_quote(data: RecommendInput):
    """
    Input : nim, mood (opsional), feeling (opsional)
    Output: quote — pesan motivasi berdasarkan emosi yang dikirim atau terakhir mahasiswa
    """
    quote = get_recommendation(data.nim, data.feeling, data.mood)
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