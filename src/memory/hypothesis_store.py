import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Dict, Any, List, Optional

class HypothesisStore:
    """Persistent store for candidate and validated hypotheses with SHA-256 deduplication."""
    def __init__(self, db_path: str = "data/cache/scientific_memory.db"):
        self.db_path = db_path
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS hypotheses (
                    hypothesis_id TEXT PRIMARY KEY,
                    statement TEXT NOT NULL,
                    variables TEXT NOT NULL,
                    expected_relationship TEXT NOT NULL,
                    evidence_json TEXT,
                    confidence REAL,
                    novelty_score REAL,
                    testability_score REAL,
                    status TEXT,
                    hash_key TEXT UNIQUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def _compute_hash(self, statement: str, variables: List[str]) -> str:
        var_str = ",".join(sorted(variables))
        content = f"{statement.strip().lower()}:{var_str}"
        return hashlib.sha256(content.encode()).hexdigest()

    def exists(self, statement: str, variables: List[str]) -> bool:
        h_key = self._compute_hash(statement, variables)
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute("SELECT 1 FROM hypotheses WHERE hash_key = ?", (h_key,))
            return cur.fetchone() is not None

    def store_hypothesis(self, hypothesis: Dict[str, Any]) -> bool:
        h_key = self._compute_hash(hypothesis["statement"], hypothesis.get("variables", []))
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO hypotheses (
                        hypothesis_id, statement, variables, expected_relationship,
                        evidence_json, confidence, novelty_score, testability_score, status, hash_key
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    hypothesis["hypothesis_id"],
                    hypothesis["statement"],
                    json.dumps(hypothesis.get("variables", [])),
                    hypothesis.get("expected_relationship", "unknown"),
                    json.dumps(hypothesis.get("evidence", [])),
                    float(hypothesis.get("confidence", 0.0)),
                    float(hypothesis.get("novelty_score", 0.0)),
                    float(hypothesis.get("testability_score", 1.0)),
                    hypothesis.get("status", "proposed"),
                    h_key
                ))
                conn.commit()
            return True
        except Exception as e:
            print(f"[HypothesisStore Error] {e}")
            return False

    def get_all(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            if status:
                cur.execute("SELECT * FROM hypotheses WHERE status = ? ORDER BY confidence DESC", (status,))
            else:
                cur.execute("SELECT * FROM hypotheses ORDER BY confidence DESC")
            rows = cur.fetchall()
            
        results = []
        for r in rows:
            results.append({
                "hypothesis_id": r["hypothesis_id"],
                "statement": r["statement"],
                "variables": json.loads(r["variables"]),
                "expected_relationship": r["expected_relationship"],
                "evidence": json.loads(r["evidence_json"] or "[]"),
                "confidence": r["confidence"],
                "novelty_score": r["novelty_score"],
                "testability_score": r["testability_score"],
                "status": r["status"]
            })
        return results

    def update_status(self, hypothesis_id: str, new_status: str, confidence: Optional[float] = None) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            if confidence is not None:
                conn.execute(
                    "UPDATE hypotheses SET status = ?, confidence = ?, updated_at = CURRENT_TIMESTAMP WHERE hypothesis_id = ?",
                    (new_status, confidence, hypothesis_id)
                )
            else:
                conn.execute(
                    "UPDATE hypotheses SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE hypothesis_id = ?",
                    (new_status, hypothesis_id)
                )
            conn.commit()
        return True
