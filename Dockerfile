# The builder image, used to build the virtual environment
FROM python:3.11-slim-buster as builder

RUN apt-get update && apt-get install -y git

WORKDIR /app

# The runtime image
FROM python:3.11-slim-buster as runtime

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH="/home/user/.local/bin:$PATH" \
    VIRTUAL_ENV=/app/.venv \
    LISTEN_PORT=8000 \
    HOST=0.0.0.0 \
    PYTHONUNBUFFERED=1

WORKDIR $HOME/app

# Copy application files
COPY --chown=user ./app ./app
COPY --chown=user ./.chainlit ./.chainlit
COPY --chown=user chainlit.md ./
COPY --chown=user requirements.txt ./

# Install dependencies
RUN pip install -r requirements.txt

EXPOSE 8000

# Run Chainlit with proper host binding
CMD ["chainlit", "run", "app/spark.py", "--host", "0.0.0.0", "--port", "8000"]