@echo off
cd /d "%~dp0"
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
set DEEPSEEK_API_KEY=sk-your-key-here
echo Pre-loading embedding model...
python -c "from engine import _get_embedder; _get_embedder(); print('Embedding OK')"
echo Starting API server...
python -m uvicorn app:app --host 0.0.0.0 --port 8000
