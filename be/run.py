"""
Backend Runner for CITD ML Meta-Labeling Demo
Starts FastAPI on http://localhost:8000
"""
import uvicorn
import os
import sys
from pathlib import Path

# Add project root to sys.path so we can import packages if needed
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "127.0.0.1")
    print(f"==================================================")
    print(f" CITD ML LAB: Meta-Labeling CatBoost Backend API")
    print(f" Running at: http://{host}:{port}")
    print(f" Swagger Docs: http://{host}:{port}/docs")
    print(f"==================================================")
    uvicorn.run("be.app.main:app", host=host, port=port, reload=True)
