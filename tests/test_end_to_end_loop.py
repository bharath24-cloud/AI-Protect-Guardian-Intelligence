import pytest
from src.datasets.materials_data import MaterialsScienceDataset
from src.loop.discovery_loop import AutonomousDiscoveryLoop
from src.memory.scientific_memory import ScientificMemory

def test_autonomous_discovery_end_to_end(tmp_path):
    db_file = str(tmp_path / "e2e_memory.db")
    kg_file = str(tmp_path / "e2e_kg.json")
    journal_file = str(tmp_path / "e2e_journal.jsonl")
    
    mem = ScientificMemory(db_path=db_file, kg_path=kg_file)
    dataset = MaterialsScienceDataset(num_samples=300, random_seed=42)
    
    loop = AutonomousDiscoveryLoop(
        dataset=dataset,
        memory=mem,
        journal_path=journal_file,
        budget_limit=15.0,
        random_seed=42
    )
    
    # Execute 3 autonomous discovery cycles
    cycles = loop.run_cycles(num_cycles=3)
    
    assert len(cycles) == 3
    for c in cycles:
        assert "hypothesis" in c
        assert "r2_score" in c
        assert "verdict" in c
        assert "reward" in c
        assert c["verdict"] in ["supported", "weak evidence", "contradicted", "rejected", "requires more experiments"]
        
    stats = mem.get_summary_statistics()
    assert stats["total_hypotheses"] >= 1
    assert stats["total_experiments_executed"] >= 1
    assert stats["knowledge_graph_nodes"] >= 5
