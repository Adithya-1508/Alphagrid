FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy project definition for dependency layer caching
COPY pyproject.toml README.md ./
COPY config/ config/
COPY src/ src/

# Install Python dependencies with uv cache mount
RUN --mount=type=cache,target=/root/.cache/uv uv pip install --system .

# Copy remaining files
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit dashboard
CMD ["streamlit", "run", "src/alphagrid/dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
