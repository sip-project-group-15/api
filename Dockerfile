FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# opencv-python-headless drops the GUI stack but still links against libglib.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copied on its own so the dependency layer is only rebuilt when pins change.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# Uploads are written at runtime; owned by the app user so the drop below works.
RUN useradd --create-home --uid 10001 kifaru \
    && mkdir -p /app/uploads \
    && chown -R kifaru:kifaru /app
USER kifaru

# 7000 is the port the katisha proxy expects (proxy/conf.d/kifaru-api.conf).
EXPOSE 7000

# urllib rather than curl — it is already in the image, so no extra apt layer.
HEALTHCHECK --interval=10s --timeout=5s --start-period=20s --retries=5 \
    CMD python -c "import urllib.request, sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:7000/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7000"]
