import os
import sys
from pathlib import Path
import uvicorn

# Ensure the project root directory is in sys.path and PYTHONPATH
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

os.environ["PYTHONPATH"] = str(ROOT_DIR) + os.pathsep + os.environ.get("PYTHONPATH", "")

if __name__ == "__main__":
    uvicorn.run(
        "backend.app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        app_dir=str(ROOT_DIR)
    )
