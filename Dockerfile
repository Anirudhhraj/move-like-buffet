# ============================================================
# move-like-buffet — production image
# ============================================================

# ── Stage 1: Build frontend ─────────────────────────────────
FROM node:20-alpine AS frontend
WORKDIR /build
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --ignore-scripts
COPY frontend/ ./
RUN npm run build


# ── Stage 2: Backend runtime ────────────────────────────────
FROM python:3.11-slim
WORKDIR /app

# Upgrade pip first — newer pip handles retries and timeouts much better
RUN pip install --upgrade pip

# ── Layer 1: CPU-only PyTorch ──
RUN pip install --no-cache-dir --timeout 600 --retries 10 \
    torch --index-url https://download.pytorch.org/whl/cpu

# ── Layer 2: sentence-transformers (no-deps) + its light deps ──
RUN pip install --no-cache-dir --timeout 600 --retries 10 --no-deps \
    sentence-transformers
RUN pip install --no-cache-dir --timeout 600 --retries 10 \
    transformers huggingface-hub tokenizers safetensors \
    tqdm scipy Pillow

# ── Layer 3: Heavy science stack (cached separately) ──
RUN pip install --no-cache-dir --timeout 600 --retries 10 \
    numpy pandas scikit-learn

# ── Layer 4: App deps (all lightweight, fast) ──
RUN pip install --no-cache-dir --timeout 600 --retries 10 \
    fastapi "uvicorn[standard]" pydantic python-dotenv \
    openai faiss-cpu yfinance

# ── Layer 5: Pre-download embedding model ──
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# ── Copy entire backend as-is ──
COPY backend/ ./

# ── Copy built frontend from stage 1 ──
COPY --from=frontend /build/dist ./static/

ENV PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]