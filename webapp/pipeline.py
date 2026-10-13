"""Job pipeline for the Groove Check web app.

Runs the full dance-detection stack on an uploaded video:
pose tracking -> bounce/rock/wave scores -> audio BPM/beats/kick ->
down/up/double phase -> annotated video (H.264) -> score plot -> result.json
"""
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import detect
from audio_bpm import analyze_audio, movement_peak_hz
from bounce_phase import classify_segment, hip_signal


def _lock(bpm, pose_npz):
    """half-time / single / double-time / off-beat / None, like audio_bpm."""
    conc, peak = movement_peak_hz(pose_npz)
    if not bpm or conc <= 0.45 or peak <= 0:
        return None, conc, peak
    beat_hz = bpm / 60
    best_k, best_r = None, 1e9
    for k, f in {"double-time": 2 * beat_hz, "single": beat_hz,
                 "half-time": beat_hz / 2}.items():
        r = abs(peak - f) / f
        if r < best_r:
            best_k, best_r = k, r
    return (best_k if best_r < 0.15 else "off-beat"), conc, peak


def _h264(src, dst):
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", src,
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
         "-movflags", "+faststart", dst], check=True)


def _plot(scores_npz, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = np.load(scores_npz)
    bounce, rock, fps = d["bounce"], d["rock"], float(d["fps"])
    t = np.arange(len(bounce)) / fps
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.plot(t, bounce, color="green", lw=1.4, label="bounce")
    ax.plot(t, rock, color="magenta", lw=1.4, label="rock")
    ax.axhline(0.45, color="black", ls="--", lw=1, alpha=0.5)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("score")
    ax.set_title("Rhythmic energy over time (dashed = detection threshold)",
                 loc="left", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def run_job(job_dir, video_path, progress):
    """progress(stage: str, frac: float). Raises on failure; writes result.json."""
    prefix = os.path.join(job_dir, "work")
    pose_npz = f"{prefix}_pose.npz"
    scores_npz = f"{prefix}_scores.npz"

    progress("Tracking body landmarks", 0.05)
    n, fps = detect.extract_pose(video_path, pose_npz)

    progress("Measuring bounce, rock and wave", 0.55)
    bounce, wave, rock, dratio, dcorr, dframe, fps, cov = \
        detect.compute_scores(pose_npz, scores_npz)

    progress("Listening to the beat", 0.65)
    au = analyze_audio(video_path)
    lock, conc, peak = _lock(au["bpm"], pose_npz)

    progress("Checking on-beat timing", 0.72)
    bsegs = detect.segments(bounce, fps, detect.BOUNCE_TH)
    wsegs = detect.segments(wave, fps, detect.WAVE_TH)
    rsegs = detect.segments(rock, fps, 0.45)
    bsegs = [[s, e, p] + list(detect.depth_of((s, e), fps, dratio, dcorr, dframe))
             for s, e, p in bsegs]
    kick_par = au["kick_parity"] if au["kick_ratio"] >= 1.5 else None
    if lock in ("single", "half-time", "double-time"):
        pt, pdet, _ = hip_signal(pose_npz)
        beats = au["beats"]
        typed = []
        for s, e, p, lab, r, c in bsegs:
            typ, lag, _ = classify_segment(pt, pdet, (s, e), beats,
                                           kick_parity=kick_par)
            typed.append([s, e, p, lab, r, c, typ, lag])
        bsegs = typed
    else:
        bsegs = [seg + ["", 0.0] for seg in bsegs]

    progress("Rendering annotated video", 0.82)
    # stash phase for the renderer, then render + re-encode to H.264
    _d = dict(np.load(scores_npz))
    _d["seg_bounds"] = np.array([[s, e] for s, e, *_ in bsegs])
    _d["seg_phase"] = np.array([s[6] for s in bsegs], dtype="U16")
    np.savez(scores_npz, **_d)
    raw_mp4 = os.path.join(job_dir, "annotated_raw.mp4")
    detect.render(video_path, scores_npz, raw_mp4)
    out_mp4 = os.path.join(job_dir, "annotated.mp4")
    _h264(raw_mp4, out_mp4)
    os.remove(raw_mp4)

    progress("Drawing score plot", 0.95)
    _plot(scores_npz, os.path.join(job_dir, "plot.png"))

    dur = len(bounce) / fps
    bcov = sum(e - s for s, e, *_ in bsegs) / dur if dur else 0
    rcov = sum(e - s for s, e, _ in rsegs) / dur if dur else 0
    resolved = [s for s in bsegs if s[6] in ("down", "up", "double")]
    lags = [abs(s[7]) for s in resolved]
    verdict = None
    if resolved:
        tight = sum(1 for s in resolved if abs(s[7]) <= 75)
        verdict = ("On beat" if tight / len(resolved) >= 0.5
                   else "Loose timing")
    result = {
        "duration_s": round(dur, 1),
        "pose_coverage": round(cov, 2),
        "bpm": round(au["bpm"], 1) if au["bpm"] else None,
        "bpm_lock": lock,
        "bounce_coverage": round(bcov, 2),
        "rock_coverage": round(rcov, 2),
        "bounce_segments": [
            {"start": round(s, 1), "end": round(e, 1),
             "type": typ or None, "lag_ms": lag if typ else None,
             "depth": lab}
            for s, e, p, lab, r, c, typ, lag in bsegs],
        "n_deep": sum(1 for s in bsegs if s[3] == "deep"),
        "verdict": verdict,
        "mean_abs_lag_ms": round(float(np.mean(lags)), 0) if lags else None,
    }
    with open(os.path.join(job_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=1)
    progress("Done", 1.0)
    return result
