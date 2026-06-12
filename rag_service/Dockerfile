FROM python:3.11-slim

WORKDIR /app

# Use Aliyun PyPI mirror for faster installs in China
COPY requirements.txt .
# Install torch (CPU-only) separately to avoid downloading 2GB of CUDA libs
RUN pip install --no-cache-dir torch==2.12.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
