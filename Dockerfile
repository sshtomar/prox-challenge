FROM python:3.12-slim

# Install system dependencies for pdf2image (poppler) and Node.js (for claude CLI)
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    curl \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && npm install -g @anthropic-ai/claude-code \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-process: ensure knowledge base exists (pages, markdown, JSON)
# If knowledge_base is already built locally and committed, this is a no-op
RUN python -c "from pathlib import Path; assert Path('knowledge_base/specs.json').exists(), 'knowledge_base not found'"

EXPOSE 8000

CMD ["python", "server.py"]
