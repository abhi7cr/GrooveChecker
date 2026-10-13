"""Detect audio BPM per video and compare against the movement's
dominant bounce frequency (single vs double bounce vs the kick)."""
import json, os, subprocess, sys
import numpy as np
import librosa

from detect import movement_peak_hz

VIDEOS = {
    "footwork-combo": ("sample.mp4", "pose_tracks.npz"),
    "hiphopindia-choreo": ("videos/hiphopindia-choreo.mp4", "results/hiphopindia-choreo_pose.npz"),
    "hiphopindia-footwork": ("videos/hiphopindia-footwork.mp4", "results/hiphopindia-footwork_pose.npz"),
    "runnin": ("videos/runnin.mp4", "results/runnin_pose.npz"),
    "untitled": ("videos/untitled.mp4", "results/untitled_pose.npz"),
}
SH_L, SH_R, HIP_L, HIP_R = 11, 12, 23, 24


def audio_bpm(src):
    wav = "/tmp/_bpm.wav"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", src, "-ac", "1",
                    "-ar", "22050", wav], check=True)
    y, sr = librosa.load(wav, sr=22050, mono=True)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    return float(np.atleast_1d(tempo)[0])


def beat_times(src):
    """Beat positions in seconds, for phase analysis."""
    wav = "/tmp/_beats.wav"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", src, "-ac", "1",
                    "-ar", "22050", wav], check=True)
    y, sr = librosa.load(wav, sr=22050, mono=True)
    _, beats = librosa.beat.beat_track(y=y, sr=sr, units="time")
    return [round(float(b), 3) for b in beats]


def kick_parity(src, beats):
    """Which beat-index parity (0/1) is kick-heavy, i.e. beats 1&3 vs 2&4.
    Returns (parity, ratio); ratio near 1 = unclear."""
    from scipy.signal import butter, filtfilt
    wav = "/tmp/_kick.wav"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", src, "-ac", "1",
                    "-ar", "22050", wav], check=True)
    y, sr = librosa.load(wav, sr=22050, mono=True)
    b, a = butter(4, 120 / (sr / 2), btype="low")
    kl = filtfilt(b, a, y)
    on = librosa.onset.onset_strength(y=kl, sr=sr)
    times = librosa.frames_to_time(np.arange(len(on)), sr=sr)
    beats = np.asarray(beats)
    if len(beats) < 8:
        return None, 1.0
    e0 = float(np.mean([on[np.argmin(np.abs(times - x))] for x in beats[0::2]]))
    e1 = float(np.mean([on[np.argmin(np.abs(times - x))] for x in beats[1::2]]))
    if min(e0, e1) < 1e-9:
        return None, 1.0
    return (0 if e0 >= e1 else 1), round(max(e0, e1) / min(e0, e1), 2)


def analyze_audio(vid):
    """BPM + beat times + kick parity for one video file.
    Returns dict with bpm, beats (list of seconds), kick_parity, kick_ratio."""
    try:
        bpm = audio_bpm(vid)
    except Exception:
        bpm = None
    try:
        beats = beat_times(vid)
    except Exception:
        beats = []
    try:
        kp, kr = kick_parity(vid, beats)
    except Exception:
        kp, kr = None, 1.0
    return {"bpm": bpm, "beats": beats, "kick_parity": kp, "kick_ratio": kr}


def _batch_main():
    out = {}
    beats_out = {}
    kick_out = {}
    for name, (vid, npz) in VIDEOS.items():
        try:
            bpm = audio_bpm(vid)
        except Exception as e:
            bpm = None
            print(f"{name}: audio failed ({type(e).__name__})", flush=True)
        try:
            beats_out[name] = beat_times(vid)
        except Exception as e:
            beats_out[name] = []
            print(f"{name}: beat times failed ({type(e).__name__})", flush=True)
        try:
            kp, kr = kick_parity(vid, beats_out[name])
            kick_out[name] = {"parity": kp, "ratio": kr}
        except Exception as e:
            kick_out[name] = {"parity": None, "ratio": 1.0}
            print(f"{name}: kick parity failed ({type(e).__name__})", flush=True)
        conc, peak = movement_peak_hz(npz) if os.path.exists(npz) else (0.0, 0.0)
        beat_hz = bpm / 60 if bpm else None
        lock = None
        if beat_hz and conc > 0.45 and peak > 0:
            cands = {"double-time": 2 * beat_hz, "single": beat_hz, "half-time": beat_hz / 2}
            best_k, best_r = None, 1e9
            for k, f in cands.items():
                r = abs(peak - f) / f
                if r < best_r:
                    best_k, best_r = k, r
            lock = best_k if best_r < 0.15 else "off-beat"
        out[name] = {"bpm": round(bpm, 1) if bpm else None,
                     "beat_hz": round(beat_hz, 2) if beat_hz else None,
                     "move_peak_hz": round(peak, 2), "move_conc": round(conc, 2),
                     "lock": lock}
        print(json.dumps({name: out[name]}), flush=True)

    with open("results/audio_bpm.json", "w") as f:
        json.dump(out, f, indent=1)
    with open("results/audio_beats.json", "w") as f:
        json.dump(beats_out, f)
    with open("results/audio_kick.json", "w") as f:
        json.dump(kick_out, f, indent=1)


if __name__ == "__main__":
    _batch_main()
