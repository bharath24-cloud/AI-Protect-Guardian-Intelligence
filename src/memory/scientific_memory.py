from pathlib import Path
from typing import Dict, Any, List, Optional

from src.memory.hypothesis_store import HypothesisStore
from src.memory.experiment_store import ExperimentStore
from src.memory.evidence_store import EvidenceStore
from src.memory.knowledge_graph import ScientificKnowledgeGraph

class ScientificMemory:
    """
    Unified Persistent Scientific Memory subsystem.
    Prevents redundant experiments and manages hypotheses, experiments, evidence, and knowledge graphs.
    """
    def __init__(
        self,
        db_path: str = "data/cache/scientific_memory.db",
        kg_path: str = "data/cache/knowledge_graph.json"
    ):
        self.db_path = db_path
        self.kg_path = kg_path
        self.hypothesis_store = HypothesisStore(db_path)
        self.experiment_store = ExperimentStore(db_path)
        self.evidence_store = EvidenceStore(db_path)
        self.knowledge_graph = ScientificKnowledgeGraph(kg_path)

    def is_hypothesis_redundant(self, statement: str, variables: List[str]) -> bool:
        """Returns True if an equivalent hypothesis has already been proposed or tested."""
        return self.hypothesis_store.exists(statement, variables)

    def store_hypothesis(self, hypothesis: Dict[str, Any]) -> bool:
        return self.hypothesis_store.store_hypothesis(hypothesis)

    def store_experiment(self, experiment: Dict[str, Any]) -> bool:
        return self.experiment_store.store_experiment(experiment)

    def store_evidence(self, evidence: Dict[str, Any]) -> bool:
        return self.evidence_store.store_evidence(evidence)

    def record_discovery_cycle(
        self,
        material_family: str,
        property_name: str,
        hypothesis: Dict[str, Any],
        experiment: Dict[str, Any],
        evidence: Dict[str, Any]
    ):
        """Atomically persists all artifacts from a single discovery cycle into SQLite & Knowledge Graph."""
        self.store_hypothesis(hypothesis)
        self.store_experiment(experiment)
        self.store_evidence(evidence)
        
        # Update hypothesis status based on evidence
        self.hypothesis_store.update_status(
            hypothesis["hypothesis_id"],
            evidence.get("support_verdict", "supported"),
            confidence=evidence.get("updated_confidence", hypothesis.get("confidence", 0.5))
        )
        
        # Link in knowledge graph
        self.knowledge_graph.link_discovery_cycle(
            material_family=material_family,
            property_name=property_name,
            hypothesis_id=hypothesis["hypothesis_id"],
            hypothesis_text=hypothesis["statement"],
            experiment_id=experiment["experiment_id"],
            evidence_id=evidence["evidence_id"],
            verdict=evidence.get("support_verdict", "supported")
        )

    def get_summary_statistics(self) -> Dict[str, Any]:
        """Returns high-level statistics of stored scientific knowledge."""
        all_hypotheses = self.hypothesis_store.get_all()
        all_experiments = self.experiment_store.get_all()
        
        status_counts = {}
        for h in all_hypotheses:
            st = h["status"]
            status_counts[st] = status_counts.get(st, 0) + 1
            
        return {
            "total_hypotheses": len(all_hypotheses),
            "status_breakdown": status_counts,
            "total_experiments_executed": len(all_experiments),
            "knowledge_graph_nodes": len(self.knowledge_graph.nodes),
            "knowledge_graph_edges": len(self.knowledge_graph.edges)
        }
