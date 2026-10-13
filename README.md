# Groove Check

Upload a hip-hop freestyle video, get back bounce / rock / body-wave
detection plus down / up / double bounce timing measured against the
track's beat grid — in milliseconds.

Built for dancers who want an honest answer to *"am I on beat?"*

## What it detects

- **Bounce** — vertical hip oscillation with 0.8–3 Hz spectral concentration
- **Rock** — rhythmic lateral hip motion (fires independently; it layers on bounce)
- **Body wave** — shoulder/hip phase-lagged cross-correlation
- **Deep vs basic bounce** — whether the chest accents the bounce (shoulder
  amplitude clearly above hip amplitude, in phase, on a solid bounce)
- **Down / up / double** — hip-dip phase vs librosa beat times; the kick drum
  identifies beats 1 & 3 so half-time "dip on 1" isn't confused with an up bounce

## Web app

```bash
pip install -r webapp/requirements.txt
python -m uvicorn webapp.app:app --port 8123
```

Open `http://localhost:8123`: upload a video (or paste an Instagram reel
link — Instagram usually blocks server-side downloads, so uploading the file
is the reliable path). You get an annotated video, an on-beat verdict, per-segment
timing, and a rhythm plot.

Deploy with Docker: `docker build -t groove-check .` (see `render.yaml` for
Render, or use any Docker host / Hugging Face Spaces).

## CLI

```bash
python detect.py <video.mp4> <output-prefix>   # pose + scores + annotated video
python audio_bpm.py                            # BPM / beat grid for the batch
python plot_detect.py                          # "how the detectors see it" figure
```

## How it works

MediaPipe Pose → torso-normalized hip trajectories → per-window FFT band
concentration (2 s windows) → thresholded segments → depth from
shoulder/hip amplitude ratio → phase from cross-correlation against the
audio beat grid. Validated on synthetic bounces and hand-labeled clips.
