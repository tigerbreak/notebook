"""Startup script: pre-load model then run uvicorn."""
import os, sys, subprocess

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# Pre-load shared embedder (avoids hanging inside uvicorn lifespan)
print("🔄 Pre-loading embedding model...")
from engine import _get_embedder
_ = _get_embedder()
print("✅ Embedding model loaded.")

# Pre-load reranker
print("🔄 Pre-loading reranker model...")
try:
    from reranker import Reranker
    r = Reranker()
    r.load()
    if r.is_ready:
        print("✅ Reranker model loaded.")
    else:
        print("⚠️ Reranker not available (will be lazy-loaded on search).")
except Exception as e:
    print(f"⚠️ Reranker pre-load skipped: {e}")

# Start uvicorn (models already cached, lifespan will reuse them)
print("🚀 Starting API server...")
subprocess.run([
    sys.executable, "-m", "uvicorn", "app:app",
    "--host", "0.0.0.0", "--port", "8000",
], check=True)
