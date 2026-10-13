FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY webapp/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# detection pipeline lives one level up
COPY detect.py audio_bpm.py bounce_phase.py pose_worker.py audio_worker.py ./
COPY webapp ./webapp

EXPOSE 8123
CMD ["sh", "-c", "python -m uvicorn webapp.app:app --host 0.0.0.0 --port ${PORT:-7860}"]
