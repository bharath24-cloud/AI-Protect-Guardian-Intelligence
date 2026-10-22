import json
import sqlite3
from pathlib import Path
from typing import Dict, Any, List, Optional

class EvidenceStore:
    """Persistent store for empirical statistical evidence linking experiments to hypotheses."""
    def __init__(self, db_path: str = "data/cache/scientific_memory.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS evidence (
                    evidence_id TEXT PRIMARY KEY,
                    hypothesis_id TEXT NOT NULL,
                    experiment_id TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL,
                    p_value REAL,
                    effect_size REAL,
                    confidence_interval_low REAL,
                    confidence_interval_high REAL,
                    support_verdict TEXT,
                    critique_notes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def store_evidence(self, evidence_record: Dict[str, Any]) -> bool:
        try:
            with sqlite3.connect(self.db_path) as conn:
                ci = evidence_record.get("confidence_interval", [0.0, 0.0])
                ci_low = float(ci[0]) if len(ci) > 0 else 0.0
                ci_high = float(ci[1]) if len(ci) > 1 else 0.0

                conn.execute("""
                    INSERT OR REPLACE INTO evidence (
                        evidence_id, hypothesis_id, experiment_id, metric_name,
                        metric_value, p_value, effect_size, confidence_interval_low,
                        confidence_interval_high, support_verdict, critique_notes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    evidence_record["evidence_id"],
                    evidence_record["hypothesis_id"],
                    evidence_record["experiment_id"],
                    evidence_record.get("metric_name", "r2_score"),
                    float(evidence_record.get("metric_value", 0.0)),
                    float(evidence_record.get("p_value", 1.0)),
                    float(evidence_record.get("effect_size", 0.0)),
                    ci_low,
                    ci_high,
                    evidence_record.get("support_verdict", "supported"),
                    evidence_record.get("critique_notes", "")
                ))
                conn.commit()
            return True
        except Exception as e:
            print(f"[EvidenceStore Error] {e}")
            return False

    def get_by_hypothesis(self, hypothesis_id: str) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute("SELECT * FROM evidence WHERE hypothesis_id = ? ORDER BY created_at DESC", (hypothesis_id,))
            rows = cur.fetchall()
            
        results = []
        for r in rows:
            results.append({
                "evidence_id": r["evidence_id"],
                "hypothesis_id": r["hypothesis_id"],
                "experiment_id": r["experiment_id"],
                "metric_name": r["metric_name"],
                "metric_value": r["metric_value"],
                "p_value": r["p_value"],
                "effect_size": r["effect_size"],
                "confidence_interval": [r["confidence_interval_low"], r["confidence_interval_high"]],
                "support_verdict": r["support_verdict"],
                "critique_notes": r["critique_notes"]
            })
        return results
