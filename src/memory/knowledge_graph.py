import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Tuple

class ScientificKnowledgeGraph:
    """
    Persistent Knowledge Graph representing the scientific discovery hierarchy:
    Material -> Property -> Hypothesis -> Experiment -> Evidence -> Conclusion
    """
    def __init__(self, json_path: str = "data/cache/knowledge_graph.json"):
        self.json_path = json_path
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.edges: List[Dict[str, Any]] = []
        self._load()

    def _load(self):
        p = Path(self.json_path)
        if p.exists():
            try:
                with open(p, "r") as f:
                    data = json.load(f)
                    self.nodes = data.get("nodes", {})
                    self.edges = data.get("edges", [])
            except Exception:
                self.nodes = {}
                self.edges = []
        else:
            p.parent.mkdir(parents=True, exist_ok=True)
            self._save()

    def _save(self):
        p = Path(self.json_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump({"nodes": self.nodes, "edges": self.edges}, f, indent=2)

    def add_node(self, node_id: str, node_type: str, label: str, metadata: Optional[Dict[str, Any]] = None):
        """Adds or updates a node in the knowledge graph."""
        self.nodes[node_id] = {
            "id": node_id,
            "type": node_type, # 'Material', 'Property', 'Hypothesis', 'Experiment', 'Evidence', 'Conclusion'
            "label": label,
            "metadata": metadata or {}
        }
        self._save()

    def add_edge(self, source_id: str, target_id: str, relation: str, metadata: Optional[Dict[str, Any]] = None):
        """Adds a directed edge between two scientific entities."""
        # Avoid duplicate edges
        for e in self.edges:
            if e["source"] == source_id and e["target"] == target_id and e["relation"] == relation:
                return
                
        self.edges.append({
            "source": source_id,
            "target": target_id,
            "relation": relation,
            "metadata": metadata or {}
        })
        self._save()

    def link_discovery_cycle(
        self,
        material_family: str,
        property_name: str,
        hypothesis_id: str,
        hypothesis_text: str,
        experiment_id: str,
        evidence_id: str,
        verdict: str
    ):
        """Convenience method to link an entire closed-loop discovery chain."""
        mat_node_id = f"mat_{material_family.lower().replace(' ', '_')}"
        prop_node_id = f"prop_{property_name.lower().replace(' ', '_')}"
        hyp_node_id = hypothesis_id
        exp_node_id = experiment_id
        evi_node_id = evidence_id
        conc_node_id = f"conc_{evidence_id}"

        self.add_node(mat_node_id, "Material", material_family)
        self.add_node(prop_node_id, "Property", property_name)
        self.add_node(hyp_node_id, "Hypothesis", hypothesis_text[:30] + "...")
        self.add_node(exp_node_id, "Experiment", f"Exp: {experiment_id}")
        self.add_node(evi_node_id, "Evidence", f"Evidence: {verdict}")
        self.add_node(conc_node_id, "Conclusion", f"Status: {verdict}")

        self.add_edge(mat_node_id, prop_node_id, "exhibits_property")
        self.add_edge(prop_node_id, hyp_node_id, "investigated_by")
        self.add_edge(hyp_node_id, exp_node_id, "tested_by")
        self.add_edge(exp_node_id, evi_node_id, "produces_evidence")
        self.add_edge(evi_node_id, conc_node_id, "leads_to_conclusion")

    def to_dict(self) -> Dict[str, Any]:
        """Exports graph in D3/Cytoscape format for the frontend."""
        nodes_list = list(self.nodes.values())
        return {
            "nodes": nodes_list,
            "edges": self.edges,
            "stats": {
                "total_nodes": len(nodes_list),
                "total_edges": len(self.edges),
                "node_types": list(set(n["type"] for n in nodes_list))
            }
        }
