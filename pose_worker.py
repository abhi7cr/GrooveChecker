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
    n, fps = detect.extract_pose(src, pose_npz)
    bounce, wave, rock, dratio, dcorr, dframe, fps, cov = \
        detect.compute_scores(pose_npz, scores_npz)
    print(json.dumps({"n_frames": n, "fps": fps,
                      "pose_coverage": round(cov, 3)}), flush=True)


if __name__ == "__main__":
    main()
