"""Memory-isolated pose worker for the Groove Check web app.

Runs MediaPipe pose extraction + bounce/rock/wave scoring in a child
process so the heavy TFLite model never lives in the web server process.
Usage: python pose_worker.py <video.mp4> <pose.npz> <scores.npz>
Prints a JSON summary line on stdout.
"""
import json
import sys

import detect


def main():
    src, pose_npz, scores_npz = sys.argv[1], sys.argv[2], sys.argv[3]
    # optional: model_complexity max_width frame_step (speed levers)
    kw = {}
    if len(sys.argv) > 4:
        kw["model_complexity"] = int(sys.argv[4])
    if len(sys.argv) > 5:
        kw["max_width"] = int(sys.argv[5])
    if len(sys.argv) > 6:
        kw["frame_step"] = int(sys.argv[6])
    if len(sys.argv) > 7:
        kw["min_tracking_confidence"] = float(sys.argv[7])

    def on_progress(f):
        print(f"PROGRESS {f:.3f}", flush=True)

    n, fps = detect.extract_pose(src, pose_npz, progress=on_progress, **kw)
    bounce, wave, rock, dratio, dcorr, dframe, fps, cov = \
        detect.compute_scores(pose_npz, scores_npz)
    print(json.dumps({"n_frames": n, "fps": fps,
                      "pose_coverage": round(cov, 3)}), flush=True)


if __name__ == "__main__":
    main()
