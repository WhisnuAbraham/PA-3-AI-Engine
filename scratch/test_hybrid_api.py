import requests
import json

url = "http://127.0.0.1:8001/api/classify"

# Test Case 1: Aman (Journal Aman + Mood Aman)
payload_aman = {
    "nim": "TEST-001",
    "text": "Hari ini saya merasa baik-baik saja dan senang bisa belajar hal baru.",
    "mood_history": [
        {"mood": "Senang", "feeling": "Bersemangat"},
        {"mood": "Biasa", "feeling": "Tenang"}
    ],
    "days_since_last_journal": 0
}

# Test Case 2: Krisis Jurnal (Text contains red flag)
payload_krisis_teks = {
    "nim": "TEST-002",
    "text": "Saya merasa putus asa dan ingin mati saja.",
    "mood_history": [],
    "days_since_last_journal": 0
}

# Test Case 3: Krisis Tren (Journal Empty + Mood History Bad for 14 days)
# Note: In real logic, we need 14 days of bad scores for Level 3
bad_history = [{"mood": "Sedih", "feeling": "Letih"}] * 14
payload_krisis_tren = {
    "nim": "TEST-003",
    "text": "",
    "mood_history": bad_history,
    "days_since_last_journal": 14
}

def test_api(name, payload):
    print(f"Testing {name}...")
    try:
        response = requests.post(url, json=payload)
        print(f"Result: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error: {e}")
    print("-" * 50)

if __name__ == "__main__":
    test_api("CASE 1: AMAN", payload_aman)
    test_api("CASE 2: KRISIS TEKS", payload_krisis_teks)
    test_api("CASE 3: KRISIS TREN (PREDICTIVE)", payload_krisis_tren)
