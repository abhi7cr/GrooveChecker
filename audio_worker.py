"""Memory-isolated audio worker for the Groove Check web app.

Runs librosa BPM / beat-grid / kick-drum analysis in a child process so the
heavy audio stack never lives in the web server process.
Usage: python audio_worker.py <video.mp4> <audio.json>
Writes the analysis as JSON and prints a summary line on stdout.
"""
import json
import sys

import numpy as np

from audio_bpm import analyze_audio


def _conv(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


def main():
    vid, out_json = sys.argv[1], sys.argv[2]
    au = analyze_audio(vid)
    with open(out_json, "w") as f:
        json.dump({k: _conv(v) for k, v in au.items()}, f)
    print(json.dumps({"bpm": _conv(au["bpm"]),
                      "n_beats": len(au["beats"])}), flush=True)


if __name__ == "__main__":
    main()
