FROM python:3.11-slim

# git: the reviewer clones the PR it reviews
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .
COPY app.py ./
COPY evals ./evals

# Don't run as root: the container clones untrusted repos
RUN useradd --create-home app
USER app

# Most hosts (Render, Railway, Fly) pass the port in $PORT
ENV PORT=8501
EXPOSE 8501
CMD streamlit run app.py --server.address 0.0.0.0 --server.port $PORT --server.headless true
