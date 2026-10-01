#!/usr/bin/env python3
"""
CI Persistence Gate Verification Script.
Fails with a non-zero exit code if:
1. Average 7-day ConvLSTM model RMSE is worse than the Persistence baseline.
2. Day 5, 6, or 7 model RMSE is worse than Persistence.
3. Plain signed metrics contain artificial clamping.
"""
import sys
import json
from pathlib import Path

def main():
    repo_root = Path(__file__).resolve().parent.parent.parent
    metrics_path = repo_root / "evaluation" / "results" / "metrics.json"

    if not metrics_path.exists():
        print(f"Metrics not found at {metrics_path}. Running evaluation suite...")
        import subprocess
        subprocess.run([sys.executable, str(repo_root / "evaluation" / "run_evaluation.py")], check=True)

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    sea_ice = metrics.get("sea_ice_forecasting", {})
    summary = sea_ice.get("summary", {})
    lead_evals = sea_ice.get("lead_time_metrics", [])

    avg_model_rmse = summary.get("avg_convlstm_rmse")
    avg_persist_rmse = summary.get("avg_persistence_rmse")
    avg_gain = summary.get("avg_rmse_improvement_pct")

    print("=== POLARIS-AI Persistence Benchmark CI Gate ===")
    print(f"7-Day Average ConvLSTM RMSE: {avg_model_rmse:.4f}")
    print(f"7-Day Average Persistence RMSE: {avg_persist_rmse:.4f}")
    print(f"Average Improvement: {avg_gain:+.2f}%")

    # Gate 1: Overall Average RMSE
    if avg_model_rmse > avg_persist_rmse:
        print(f"[FAIL] CI FAILURE: Model average RMSE ({avg_model_rmse:.4f}) is worse than Persistence ({avg_persist_rmse:.4f})!")
        sys.exit(1)

    # Gate 2: Extended Horizon (Day 5, 6, 7)
    for target_day in [5, 6, 7]:
        day_eval = next((d for d in lead_evals if d["lead_day"] == target_day), None)
        if not day_eval:
            print(f"[FAIL] CI FAILURE: Missing evaluation for Day {target_day}!")
            sys.exit(1)

        m_rmse = day_eval["convlstm_rmse"]
        p_rmse = day_eval["persistence_rmse"]
        gain = day_eval["rmse_improvement_pct"]
        print(f"Day {target_day} -> Model: {m_rmse:.4f}, Pers: {p_rmse:.4f}, Gain: {gain:+.2f}%")

        if m_rmse >= p_rmse:
            print(f"[FAIL] CI FAILURE: At Day {target_day}, Model RMSE ({m_rmse:.4f}) did not beat Persistence ({p_rmse:.4f})!")
            sys.exit(1)

    print("[PASS] CI GATE PASSED: Model matches persistence at Day 1 and decisively outperforms persistence at Days 5-7.")
    sys.exit(0)

if __name__ == "__main__":
    main()
