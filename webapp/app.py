"""Groove Check web app: upload a dance video (or an Instagram link) and get
bounce / rock / body-wave / on-beat analysis back."""
import json
import os
import re
import shutil
import subprocess
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

BASE = os.path.dirname(os.path.abspath(__file__))
JOBS = os.path.join(BASE, "jobs")
os.makedirs(JOBS, exist_ok=True)

MAX_BYTES = 250 * 1024 * 1024
JOB_RE = re.compile(r"^[0-9a-f]{32}$")
IG_RE = re.compile(r"^https?://(www\.)?instagram\.com/(reel|p)/[A-Za-z0-9_\-]+/?")

app = FastAPI(title="Groove Check")
pool = ThreadPoolExecutor(max_workers=2)
jobs = {}
# Serialize analyses: the free-tier box fits one job's peak comfortably,
# not two at once.
job_lock = threading.Lock()


def _job_dir(job_id):
    if not JOB_RE.match(job_id or ""):
        raise HTTPException(404, "unknown job")
    d = os.path.join(JOBS, job_id)
    if not os.path.isdir(d):
        raise HTTPException(404, "unknown job")
    return d


def _run(job_id, video_path):
    job_dir = os.path.join(JOBS, job_id)
    try:
        from webapp.pipeline import run_job

        def progress(stage, frac):
            jobs[job_id].update(status="running", stage=stage, progress=frac)

        with job_lock:
            result = run_job(job_dir, video_path, progress)
        jobs[job_id].update(status="done", stage="Done", progress=1.0,
                            result=result)
    except Exception as e:
        jobs[job_id].update(status="error", stage="Failed",
                            error=f"{type(e).__name__}: {e}")


def _submit(job_id, video_path):
    jobs[job_id] = {"status": "queued", "stage": "Queued", "progress": 0.0}
    pool.submit(_run, job_id, video_path)
    return {"job_id": job_id}


@app.post("/api/jobs/upload")
async def upload(file: UploadFile = File(...)):
    name = (file.filename or "").lower()
    if not name.endswith((".mp4", ".mov", ".m4v")):
        raise HTTPException(400, "please upload an .mp4 or .mov video")
    job_id = uuid.uuid4().hex
    job_dir = os.path.join(JOBS, job_id)
    os.makedirs(job_dir)
    dest = os.path.join(job_dir, "input.mp4")
    size = 0
    with open(dest, "wb") as f:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > MAX_BYTES:
                f.close()
                shutil.rmtree(job_dir, ignore_errors=True)
                raise HTTPException(400, "video is too large (250 MB max)")
            f.write(chunk)
    return _submit(job_id, dest)


@app.post("/api/jobs/instagram")
async def from_instagram(body: dict):
    url = (body.get("url") or "").strip()
    if not IG_RE.match(url):
        raise HTTPException(400, "that doesn't look like an Instagram reel/post link")
    job_id = uuid.uuid4().hex
    job_dir = os.path.join(JOBS, job_id)
    os.makedirs(job_dir)
    dest = os.path.join(job_dir, "input.mp4")
    try:
        subprocess.run(
            ["yt-dlp", "--no-check-certificate", "--no-playlist",
             "-f", "b[ext=mp4]/b", "--max-filesize", "80m",
             "-o", dest, url],
            check=True, capture_output=True, timeout=180)
    except Exception:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(
            422, "instagram_blocked: Instagram wouldn't hand over that video. "
                 "Open the reel, tap Share > Download (or screen-record it), "
                 "then upload the file here instead.")
    if not os.path.exists(dest) or os.path.getsize(dest) == 0:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(422, "instagram_blocked: couldn't fetch that reel. "
                                 "Please upload the video file directly.")
    return _submit(job_id, dest)


@app.get("/api/jobs/{job_id}")
async def job_status(job_id: str):
    _job_dir(job_id)
    return jobs.get(job_id, {"status": "unknown"})


@app.get("/api/jobs/{job_id}/video")
async def job_video(job_id: str):
    d = _job_dir(job_id)
    p = os.path.join(d, "annotated.mp4")
    if not os.path.exists(p):
        raise HTTPException(404, "video not ready yet")
    return FileResponse(p, media_type="video/mp4")


@app.get("/api/jobs/{job_id}/plot")
async def job_plot(job_id: str):
    d = _job_dir(job_id)
    p = os.path.join(d, "plot.png")
    if not os.path.exists(p):
        raise HTTPException(404, "plot not ready yet")
    return FileResponse(p, media_type="image/png")


@app.get("/", response_class=HTMLResponse)
async def index():
    with open(os.path.join(BASE, "templates", "index.html")) as f:
        return f.read()
