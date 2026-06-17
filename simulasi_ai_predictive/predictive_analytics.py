import os
import pandas as pd
from datetime import datetime, timedelta
from utils.db_connector import get_database

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MOOD_SCORES = {
    'Senang': 1,
    'Antusias': 1,
    'Biasa': 2,
    'Netral': 2,
    'Terkejut': 3,
    'Jijik': 4,
    'Takut': 4,
    'Sedih': 5,
    'Marah': 5
}

FEELING_SCORES = {
    'Gembira': 1, 'Bangga': 1, 'Bersyukur': 1, 'Ceria': 1, 
    'Semangat': 1, 'Energik': 1, 'Kagum': 1, 'Bergairah': 1,
    'Antusias': 1,
    'Biasa Saja': 2, 'Stabil': 2, 'Tenang': 2, 'Santai': 2, 'Biasa': 2,
    'Tercengang': 3, 'Penasaran': 3, 'Tertarik': 3, 'Gelagapan': 3, 'Bosan': 3,
    'Kesal': 4, 'Jengkel': 4, 'Benci': 4, 'Kecewa': 4, 'Lelah': 4, 'Stres': 4,
    'Pilu': 5, 'Depresi': 5, 'Kesepian': 5, 'Putus Asa': 5,
    'Cemas': 5, 'Khawatir': 5, 'Panik': 5, 'Gelisah': 5
}

def get_combined_score(mood: str, feeling: str) -> float:
    m_str = str(mood).strip()
    f_str = str(feeling).strip()
    
    m_score = 3
    for k, v in MOOD_SCORES.items():
        if k.lower() == m_str.lower():
            m_score = v
            break
            
    f_score = 3
    for k, v in FEELING_SCORES.items():
        if k.lower() == f_str.lower():
            f_score = v
            break

    return (m_score + f_score) / 2

def run_clinical_analysis() -> list:
    """
    Menganalisis kondisi mental seluruh mahasiswa berdasarkan data daily_checkins dari MongoDB.
    Menggunakan jendela waktu 14 hari terakhir.
    """
    alerts = []

    try:
        db = get_database()
        
        mood_map = {str(m['mood_id']): m['mood_name'] for m in db.moods.find({"mood_id": {"$exists": True}})}
        feel_map = {str(f['id']): f['feeling_name'] for f in db.feelings.find({"id": {"$exists": True}})}

        cutoff_date = datetime.now() - timedelta(days=30)
        
        checkins_cursor = db.daily_checkins.find({
            "created_at": {"$gte": cutoff_date}
        }).sort([("nim", 1), ("created_at", 1)])
        
        df = pd.DataFrame(list(checkins_cursor))
        
        if df.empty:
            return [{"message": "Tidak ada data check-in terbaru di database."}]

        df['mood_name'] = df['mood_id'].apply(lambda x: mood_map.get(str(x), 'Netral'))
        df['feeling_name'] = df['feeling_id'].apply(lambda x: feel_map.get(str(x), 'Biasa Saja'))

        for nim in df['nim'].unique():
            user_data = df[df['nim'] == nim].copy()
            total_hari = len(user_data)
            
            user_data['daily_score'] = user_data.apply(
                lambda row: get_combined_score(row.get('mood_name'), row.get('feeling_name')), axis=1
            )
            
            two_weeks_data = user_data.tail(14)
            avg_score = round(two_weeks_data['daily_score'].mean(), 2)

            alert = {
                "nim": nim,
                "total_hari": total_hari,
                "avg_score": avg_score,
            }

            if total_hari >= 14 and avg_score >= 4.0:
                alert["status"] = "HIGH_RISK"
                alert["message"] = "Kondisi mental menurun konsisten selama 2 minggu (Terdeteksi Level 3)"
                alert["action"] = "WAJIB dirujuk ke Konselor (Tingkat Krisis)"

            elif total_hari >= 3 and all(x >= 3.5 for x in two_weeks_data['daily_score'].tail(3)):
                alert["status"] = "MODERATE_RISK"
                alert["message"] = "Tren mood & perasaan menurun drastis dalam 3 hari terakhir"
                alert["action"] = "Pantau lebih intensif oleh Konselor"

            else:
                alert["status"] = "NORMAL"
                alert["message"] = "Mahasiswa terpantau stabil/normal"
                alert["action"] = None

            alerts.append(alert)

    except Exception as e:
        alerts.append({"error": f"Gagal mengambil data dari MongoDB: {str(e)}"})

    return alerts

def evaluate_predictive_risk(recent_emotions: list[float], days_since_last_journal: int, is_journal_empty: bool) -> dict:
    """
    Evaluasi prediktif mengecek riwayat emosi dan rentang pengisian jurnal.
    Meresolusi level prediktif (0-3) berdasarkan riwayat mood.
    """
    alert = {
        "predictive_level": 0,
        "reason": ""
    }
    
    total_hari = len(recent_emotions)
    
    if total_hari > 0:
        avg_score = sum(recent_emotions) / total_hari
    else:
        avg_score = 0.0

    if is_journal_empty and days_since_last_journal >= 14 and avg_score >= 4.0:
        alert["predictive_level"] = 3
        alert["reason"] = f"Pasif {days_since_last_journal} hari tanpa jurnal dengan riwayat mood menurun (avg: {avg_score:.1f})."
        return alert

    if total_hari >= 14 and avg_score >= 4.0:
        alert["predictive_level"] = 3
        alert["reason"] = f"Penurunan mood menetap berdasarkan riwayat rentang panjang (avg score: {avg_score:.1f})."
        return alert
        
    if total_hari >= 3:
        last_3 = recent_emotions[-3:]
        avg_3 = sum(last_3) / 3
        
        if avg_3 >= 4.0:
            alert["predictive_level"] = 2
            alert["reason"] = f"Mood & perasaan sangat memburuk dalam 3 hari terakhir (avg skor: {avg_3:.1f})."
        elif avg_3 >= 3.5:
            alert["predictive_level"] = 2
            alert["reason"] = f"Mood & perasaan cukup memburuk dalam 3 hari terakhir (avg skor: {avg_3:.1f})."
        elif avg_3 >= 3.0:
            alert["predictive_level"] = 1
            alert["reason"] = f"Mood & perasaan kurang stabil dalam 3 hari terakhir (avg skor: {avg_3:.1f})."

    return alert

if __name__ == "__main__":
    print("=" * 70)
    print("AI CLINICAL PREDICTIVE (MONGODB): 14-DAY MENTAL HEALTH MONITORING")
    print("=" * 70)

    hasil = run_clinical_analysis()
    for item in hasil:
        if "error" in item:
            print(f"Error: {item['error']}")
            continue
        if "nim" not in item:
            print(item["message"])
            continue
            
        print(f"[{item['nim']}] Hari: {item['total_hari']} | Avg Score: {item['avg_score']}")
        print(f"  Status  : {item['status']}")
        print(f"  Pesan   : {item['message']}")
        if item['action']:
            print(f"  Tindakan: {item['action']}")
        print("-" * 70)