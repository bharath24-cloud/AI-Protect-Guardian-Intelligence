import os
import pytest
from src.memory.scientific_memory import ScientificMemory

def test_scientific_memory(tmp_path):
    db_file = str(tmp_path / "test_memory.db")
    kg_file = str(tmp_path / "test_kg.json")
    
    mem = ScientificMemory(db_path=db_file, kg_path=kg_file)
    
    hyp = {
        "hypothesis_id": "hyp-test-01",
        "statement": "Electronegativity delta increases band gap.",
        "variables": ["electronegativity_delta", "band_gap"],
        "expected_relationship": "monotonic_correlation",
        "confidence": 0.65,
        "novelty_score": 0.8,
        "testability_score": 1.0,
        "status": "proposed"
    }
    
    # Store
    assert mem.store_hypothesis(hyp)
    # Check deduplication
    assert mem.is_hypothesis_redundant(hyp["statement"], hyp["variables"])
    
    # Check Knowledge Graph
    mem.record_discovery_cycle(
        material_family="Perovskite Oxides",
        property_name="band_gap",
        hypothesis=hyp,
        experiment={
            "experiment_id": "exp-test-01",
            "hypothesis_id": "hyp-test-01",
            "experiment_type": "cross_validation_regression",
            "parameters": {},
            "random_seed": 42,
            "computational_cost": 1.0,
            "status": "completed",
            "metrics": {"r2": 0.62}
        },
        evidence={
            "evidence_id": "evi-test-01",
            "hypothesis_id": "hyp-test-01",
            "experiment_id": "exp-test-01",
            "metric_name": "r2",
            "metric_value": 0.62,
            "p_value": 0.001,
            "effect_size": 0.85,
            "support_verdict": "supported",
            "updated_confidence": 0.90
        }
    )
    
    stats = mem.get_summary_statistics()
    assert stats["total_hypotheses"] == 1
    assert stats["total_experiments_executed"] == 1
    assert stats["knowledge_graph_nodes"] >= 5
