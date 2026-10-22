from typing import Dict, Any, List, Optional
from src.memory.scientific_memory import ScientificMemory

class MemoryAgent:
    """Coordinates memory persistence and guards against duplicate or redundant investigations."""
    def __init__(self, memory: Optional[ScientificMemory] = None):
        self.memory = memory or ScientificMemory()

    def filter_redundant_hypotheses(self, candidate_hypotheses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filters out hypotheses already tested or stored in memory."""
        novel_hypotheses = []
        for h in candidate_hypotheses:
            if not self.memory.is_hypothesis_redundant(h["statement"], h.get("variables", [])):
                novel_hypotheses.append(h)
        return novel_hypotheses

    def commit_cycle(
        self,
        material_family: str,
        property_name: str,
        hypothesis: Dict[str, Any],
        experiment: Dict[str, Any],
        evidence: Dict[str, Any]
    ):
        """Persists the complete discovery chain into memory."""
        self.memory.record_discovery_cycle(
            material_family=material_family,
            property_name=property_name,
            hypothesis=hypothesis,
            experiment=experiment,
            evidence=evidence
        )
