# Groove Check — web app

Upload a freestyle video (or paste an Instagram reel link) and get back
bounce / rock / body-wave detection plus down/up/double bounce timing
against the track's beat grid.

## Run locally

```bash
cd ~/workspace/dance-detection
venv/bin/python -m uvicorn webapp.app:app --host 0.0.0.0 --port 8123
```

Then open http://localhost:8123 (or the machine's LAN IP from your phone).

## How it works

- `POST /api/jobs/upload` — multipart video file (.mp4/.mov, 250 MB max)
- `POST /api/jobs/instagram` — `{"url": "https://www.instagram.com/reel/…"}`;
  tries yt-dlp, returns 422 with a friendly message when Instagram blocks it
  (common — in that case save the reel and upload the file)
- `GET /api/jobs/{id}` — poll for status/progress/result
- `GET /api/jobs/{id}/video` — annotated H.264 video, `…/plot` — score plot

The pipeline (`webapp/pipeline.py`) reuses the detection modules at the repo
root: pose tracking (MediaPipe), bounce/rock/wave scores (`detect.py`),
BPM + beat times + kick parity (`audio_bpm.py`), down/up/double phase
(`bounce_phase.py`). Two jobs run at most in parallel.

## Deploy publicly

```bash
cd ~/workspace/dance-detection
docker build -t groove-check .
docker run -p 8123:8123 groove-check
```

For a public URL, push the image to Render / Railway / Fly.io (all take a
Dockerfile directly), or to Hugging Face Spaces (Docker mode).
