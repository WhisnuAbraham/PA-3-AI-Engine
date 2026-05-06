import os
from pymongo import MongoClient
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, '.env'))

def get_db_client():
    uri = os.getenv("DB_URI")
    if not uri:
        raise ValueError("DB_URI tidak ditemukan di .env")
    return MongoClient(uri)

def get_database():
    client = get_db_client()
    db_name = os.getenv("DB_NAME", "monitoring")
    return client[db_name]

if __name__ == "__main__":
    try:
        db = get_database()
        print(f"Berhasil terhubung ke database: {db.name}")
        # Test query to a collection (e.g., daily_checkins)
        count = db.daily_checkins.count_documents({})
        print(f"Jumlah dokumen di daily_checkins: {count}")
    except Exception as e:
        print(f"Koneksi gagal: {e}")
