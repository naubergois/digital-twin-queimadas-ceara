"""Testes do agente explicador (sem rede)."""

from src.compare_case_explainer_agent import slim_case_for_llm


def test_slim_case_for_llm_keeps_core_metrics():
    row = {
        "date": "2024-01-03",
        "iou": 0.5,
        "precision": 0.8,
        "recall": 0.4,
        "tp": 4,
        "fp": 1,
        "fn": 6,
        "n_focos": 20,
        "figure": "/tmp/x.png",
        "adaptive_threshold_meta": {"mode": "adaptive_grid", "chosen": 0.3, "noise": "x" * 500},
    }
    s = slim_case_for_llm(row)
    assert s["date"] == "2024-01-03"
    assert s["tp"] == 4
    assert "figure" not in s
    assert "noise" not in s["adaptive_threshold_meta"]
