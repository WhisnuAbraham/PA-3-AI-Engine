import sys
import os

# Tambah root dir ke path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulasi_ai_predictive.predictive_analytics import evaluate_predictive_risk

def test_predictive():
    print("=== TESTING PREDICTIVE RISK LOGIC ===")
    
    # 1. Test 3-day severe bad mood (avg_score >= 4.0)
    # Harus bernilai Level 2 sekarang (sebelumnya Level 3)
    emotions_3d_severe = [4.5, 4.0, 4.5]
    res_3d_severe = evaluate_predictive_risk(emotions_3d_severe, 0, False)
    print(f"3-Day Severe (avg >= 4.0) -> Level: {res_3d_severe['predictive_level']}, Reason: {res_3d_severe['reason']}")
    assert res_3d_severe['predictive_level'] == 2, f"Expected Level 2, got {res_3d_severe['predictive_level']}"
    
    # 2. Test 3-day moderate bad mood (avg_score >= 3.5)
    emotions_3d_mod = [3.5, 3.5, 3.5]
    res_3d_mod = evaluate_predictive_risk(emotions_3d_mod, 0, False)
    print(f"3-Day Moderate (avg >= 3.5) -> Level: {res_3d_mod['predictive_level']}, Reason: {res_3d_mod['reason']}")
    assert res_3d_mod['predictive_level'] == 2, f"Expected Level 2, got {res_3d_mod['predictive_level']}"

    # 3. Test 14-day severe bad mood (avg_score >= 4.0)
    # Harus tetap Level 3
    emotions_14d_severe = [4.0] * 14
    res_14d_severe = evaluate_predictive_risk(emotions_14d_severe, 0, False)
    print(f"14-Day Severe (avg >= 4.0) -> Level: {res_14d_severe['predictive_level']}, Reason: {res_14d_severe['reason']}")
    assert res_14d_severe['predictive_level'] == 3, f"Expected Level 3, got {res_14d_severe['predictive_level']}"

    # 4. Test 14-day passive without journal and bad historical mood (avg_score >= 4.0)
    # Harus tetap Level 3
    res_14d_passive = evaluate_predictive_risk([4.0], 14, True)
    print(f"14-Day Passive (avg >= 4.0) -> Level: {res_14d_passive['predictive_level']}, Reason: {res_14d_passive['reason']}")
    assert res_14d_passive['predictive_level'] == 3, f"Expected Level 3, got {res_14d_passive['predictive_level']}"

    print("ALL TESTS PASSED!")

if __name__ == "__main__":
    test_predictive()
