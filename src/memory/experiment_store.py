import json
import sqlite3
from pathlib import Path
from typing import Dict, Any, List, Optional

class ExperimentStore:
    """Persistent store for all computational experiment configurations, seeds, and execution results."""
    def __init__(self, db_path: str = "data/cache/scientific_memory.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY,
                    hypothesis_id TEXT NOT NULL,
                    experiment_type TEXT NOT NULL,
                    parameters_json TEXT,
                    random_seed INTEGER,
                    computational_cost REAL,
                    status TEXT,
                    metrics_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def store_experiment(self, experiment_record: Dict[str, Any]) -> bool:
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO experiments (
                        experiment_id, hypothesis_id, experiment_type, parameters_json,
                        random_seed, computational_cost, status, metrics_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    experiment_record["experiment_id"],
                    experiment_record["hypothesis_id"],
                    experiment_record.get("experiment_type", "computational_model"),
                    json.dumps(experiment_record.get("parameters", {})),
                    experiment_record.get("random_seed", 42),
                    float(experiment_record.get("computational_cost", 1.0)),
                    experiment_record.get("status", "completed"),
                    json.dumps(experiment_record.get("metrics", {}))
                ))
                conn.commit()
            return True
        except Exception as e:
            print(f"[ExperimentStore Error] {e}")
            return False

    def get_all(self) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute("SELECT * FROM experiments ORDER BY created_at DESC")
            rows = cur.fetchall()
            
        results = []
        for r in rows:
            results.append({
                "experiment_id": r["experiment_id"],
                "hypothesis_id": r["hypothesis_id"],
                "experiment_type": r["experiment_type"],
                "parameters": json.loads(r["parameters_json"] or "{}"),
                "random_seed": r["random_seed"],
                "computational_cost": r["computational_cost"],
                "status": r["status"],
                "metrics": json.loads(r["metrics_json"] or "{}"),
                "created_at": r["created_at"]
            })
        return results
