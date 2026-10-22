import sqlite3
from pathlib import Path

db_path = Path("backend/data/aiprotect.db")
if db_path.exists():
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(evaluation_results)")
    existing = [c[1] for c in cursor.fetchall()]

    new_cols = [
        ("status", "VARCHAR(32) DEFAULT 'evaluated'"),
        ("status_detail", "TEXT"),
        ("evaluation_type", "VARCHAR(32) DEFAULT 'classifier'"),
        ("prompt_template", "VARCHAR(128)")
    ]

    for col, dtype in new_cols:
        if col not in existing:
            cursor.execute(f"ALTER TABLE evaluation_results ADD COLUMN {col} {dtype}")
            print(f"Added column {col} to evaluation_results")
        else:
            print(f"Column {col} already exists")

    conn.commit()
    conn.close()
    print("Database schema updated successfully.")
else:
    print(f"Database at {db_path} does not exist yet.")
