FROM python:3.12-slim

ARG PORT=8051

WORKDIR /app

# Install system dependencies for various packages
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Copy the MCP server files
COPY . .

# Install packages directly to the system (no virtual environment)
# Combining commands to reduce Docker layers
# Install base dependencies and all optional dependencies
RUN uv pip install --system -e ".[all]" && \
    crawl4ai-setup && \
    # Download NLTK data
    python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger')"

# Set environment variables for better compatibility
ENV PYTHONUNBUFFERED=1
ENV TOKENIZERS_PARALLELISM=false

EXPOSE ${PORT}

# Command to run the MCP server
CMD ["python", "src/crawl4ai_mcp.py"]
