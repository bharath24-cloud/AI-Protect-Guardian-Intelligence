import asyncio
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sqlalchemy import inspect
from backend.app.database import engine, Base
# Import all models so metadata is complete
import backend.app.db_models.models  # noqa: F401
import backend.app.memory.models     # noqa: F401

logger = logging.getLogger("guardian.migrations")

async def run_migrations() -> None:
    """Creates Guardian Intelligence memory tables if they do not exist."""
    print("[Migrations] Inspecting database tables for Guardian Intelligence...")
    async with engine.begin() as conn:
        def check_and_create(sync_conn):
            inspector = inspect(sync_conn)
            existing_tables = inspector.get_table_names()
            target_tables = [
                "guardian_plans",
                "guardian_approvals",
                "guardian_executions",
                "guardian_evidence",
            ]
            missing = [t for t in target_tables if t not in existing_tables]
            if missing:
                print(f"[Migrations] Creating missing Guardian tables: {missing}")
                Base.metadata.create_all(sync_conn)
                print("[Migrations] Guardian memory tables created successfully.")
            else:
                print("[Migrations] All Guardian tables already exist. Schema up to date.")

        await conn.run_sync(check_and_create)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_migrations())
