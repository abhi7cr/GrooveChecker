"""Rule-based hip-hop move detectors on MediaPipe pose tracks.
Detectors: bounce (rhythmic hip oscillation), rock (lateral hip motion),
body wave (phase-lagged torso segments), and bounce depth (basic = knees
only, deep = chest involved too).

Usage: python detect.py <src.mp4> <out_prefix>
Writes <out_prefix>_pose.npz, <out_prefix>_scores.npz,
<out_prefix>_annotated.mp4, <out_prefix>_summary.json
"""
import json, os, sys
import cv2
import numpy as np

# MediaPipe pose landmark ids
NOSE = 0
SH_L, SH_R = 11, 12
HIP_L, HIP_R = 23, 24
KNEE_L, KNEE_R = 25, 26
ANK_L, ANK_R = 27, 28

BOUNCE_BAND = (0.8, 3.0)      # Hz where hip-hop bounce lives
WIN_S = 2.0                   # sliding window seconds
WAVE_MIN_CORR = 0.60
WAVE_MIN_GAIN = 0.15          # corr(lag*) - corr(0)
AMP_MIN = 0.020               # torso-lengths, noise floor
BOUNCE_TH = 0.45
WAVE_TH = 0.55
# Deep bounce: chest is involved too, not just the knees. Measured as
# shoulder vertical amplitude / hip vertical amplitude, with the two moving
# in phase. (His definition: "Basic Bounce is knees bent but a deep bounce
# involves chest too", 2026-10-11.)
# The line sits at 1.0: below it the hips dominate the motion (knees driving,
# chest riding along); above it the shoulders move MORE than the hips, i.e.
# the chest is adding motion of its own.
DEEP_RATIO_MIN = 1.15
DEEP_CORR_MIN = 0.40


def extract_pose(src, npz_path, model_complexity=1, max_width=0, frame_step=1,
                 min_tracking_confidence=0.5, progress=None):
    """Track pose landmarks.

    Speed levers (for slow CPUs): model_complexity=0 (lite model),
    max_width (downscale frames wider than this), frame_step=2 (track
    every 2nd frame, linearly interpolate the rest — 12 fps still
    resolves the 0.8–3 Hz bounce band comfortably).
    progress(frac) is called periodically when given.
    """
    import mediapipe as mp  # lazy: keeps the import light for non-pose workers
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pose = mp.solutions.pose.Pose(static_image_mode=False,
                                  model_complexity=model_complexity,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=min_tracking_confidence)
    xs = np.full((n, 33), np.nan)
    ys = np.full((n, 33), np.nan)
    vis = np.zeros((n, 33))
    sampled = np.zeros(n, dtype=bool)
    i = 0
    last_report = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % frame_step == 0 and i < n:
            if max_width and frame.shape[1] > max_width:
                s = max_width / frame.shape[1]
                frame = cv2.resize(frame, (0, 0), fx=s, fy=s,
                                   interpolation=cv2.INTER_AREA)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)
            if res.pose_landmarks:
                for j, lm in enumerate(res.pose_landmarks.landmark):
                    xs[i, j], ys[i, j], vis[i, j] = lm.x, lm.y, lm.visibility
            sampled[i] = True
        i += 1
        if progress is not None and n > 0 and i - last_report >= 30:
            last_report = i
            progress(min(1.0, i / n))
    cap.release()
    pose.close()
    n = min(i, n)
    xs, ys, vis, sampled = xs[:n], ys[:n], vis[:n], sampled[:n]
    if frame_step > 1:
        # Fill skipped frames by linear interpolation between tracked frames
        # (genuine detection gaps stay NaN).
        idx = np.arange(n)
        for arr in (xs, ys, vis):
            for j in range(33):
                v = arr[:, j]
                fill = ~sampled
                if not fill.any():
                    continue
                s_idx = idx[sampled & ~np.isnan(v)]
                if len(s_idx) >= 2:
                    v[fill] = np.interp(idx[fill], s_idx, v[s_idx])
                arr[:, j] = v
    np.savez(npz_path, xs=xs, ys=ys, vis=vis, fps=np.array(fps))
    return n, float(fps)


def interp_nan(v):
    n = len(v)
    x = np.arange(n)
    m = ~np.isnan(v)
    if m.sum() < 3:
        return np.zeros(n)
    return np.interp(x, x[m], v[m])


def band_concentration(sig, fps, lo, hi):
    s = sig - sig.mean()
    if s.std() < 1e-9:
        return 0.0
    spec = np.abs(np.fft.rfft(s)) ** 2
    freqs = np.fft.rfftfreq(len(s), 1.0 / fps)
    band = (freqs >= lo) & (freqs <= hi)
    return float(spec[band].sum() / (spec.sum() + 1e-12))


def xcorr(a, b, max_lag):
    a = a - a.mean(); b = b - b.mean()
    da, db = a.std(), b.std()
    if da < 1e-9 or db < 1e-9:
        return np.zeros(2 * max_lag + 1)
    full = np.correlate(a / da, b / db, mode="full") / len(a)
    c = len(a) - 1
    return full[c - max_lag:c + max_lag + 1]


def movement_peak_hz(pose_npz):
    """Concentration-weighted median of dominant hip-oscillation frequency
    across high-confidence bounce windows (4s windows, zero-padded FFT)."""
    d = np.load(pose_npz)
    xs, ys, vis = d["xs"], d["ys"], d["vis"]
    fps = float(d["fps"])
    n = len(xs)
    shx, shy = (xs[:, SH_L] + xs[:, SH_R]) / 2, (ys[:, SH_L] + ys[:, SH_R]) / 2
    hpx, hpy = (xs[:, HIP_L] + xs[:, HIP_R]) / 2, (ys[:, HIP_L] + ys[:, HIP_R]) / 2
    shv = (vis[:, SH_L] + vis[:, SH_R]) / 2
    hpv = (vis[:, HIP_L] + vis[:, HIP_R]) / 2
    torso = np.sqrt((shx - hpx) ** 2 + (shy - hpy) ** 2)
    ok = (shv > 0.5) & (hpv > 0.5)
    x = np.arange(n); m = ok & ~np.isnan(torso)
    torso_i = np.interp(x, x[m], torso[m]) if m.sum() > 3 else np.ones(n)
    hpy_i = np.interp(x, x[ok], hpy[ok]) if ok.sum() > 3 else np.zeros(n)
    sig = hpy_i / np.maximum(torso_i, 1e-6)

    w = int(4.0 * fps)
    N = 512  # zero-padded FFT -> ~0.047 Hz resolution
    peaks, weights = [], []
    for t in range(0, n - w, w // 2):
        s = sig[t:t + w] - sig[t:t + w].mean()
        if s.std() < 1e-9 or ok[t:t + w].mean() < 0.7:
            continue
        spec = np.abs(np.fft.rfft(s, N)) ** 2
        freqs = np.fft.rfftfreq(N, 1.0 / fps)
        band = (freqs >= 0.6) & (freqs <= 3.4)
        conc = spec[band].sum() / (spec.sum() + 1e-12)
        if conc > 0.5:
            peaks.append(float(freqs[band][np.argmax(spec[band])]))
            weights.append(float(conc))
    if not peaks:
        return 0.0, 0.0
    peaks, weights = np.array(peaks), np.array(weights)
    order = np.argsort(peaks)
    cum = np.cumsum(weights[order]) / weights.sum()
    med = float(peaks[order][np.searchsorted(cum, 0.5)])
    return float(weights.mean()), med


def compute_scores(pose_npz, scores_npz):
    d = np.load(pose_npz)
    xs, ys, vis = d["xs"], d["ys"], d["vis"]
    fps = float(d["fps"])
    n = len(xs)
    shx, shy = (xs[:, SH_L] + xs[:, SH_R]) / 2, (ys[:, SH_L] + ys[:, SH_R]) / 2
    hpx, hpy = (xs[:, HIP_L] + xs[:, HIP_R]) / 2, (ys[:, HIP_L] + ys[:, HIP_R]) / 2
    shv = (vis[:, SH_L] + vis[:, SH_R]) / 2
    hpv = (vis[:, HIP_L] + vis[:, HIP_R]) / 2

    torso = np.sqrt((shx - hpx) ** 2 + (shy - hpy) ** 2)
    torso = interp_nan(np.where((shv > 0.5) & (hpv > 0.5), torso, np.nan))
    torso = np.maximum(torso, 1e-6)
    # Fixed reference: per-frame torso normalization forces the shoulder and
    # hip signals to move in lockstep (corr ~= 1 everywhere), which kills any
    # shoulder-vs-hip discrimination. Normalize by the median torso instead.
    torso_ref = float(np.median(torso[torso > 1e-3])) if (torso > 1e-3).any() else 1.0

    ok = (shv > 0.5) & (hpv > 0.5)
    sh_sig = interp_nan(np.where(ok, shy / torso_ref, np.nan))
    hp_sig = interp_nan(np.where(ok, hpy / torso_ref, np.nan))
    hx_sig = interp_nan(np.where(ok, hpx / torso_ref, np.nan))
    ok_f = ok.astype(float)
    pose_cov = float(ok_f.mean())

    w = int(WIN_S * fps)
    half = w // 2
    bounce = np.zeros(n)
    wave = np.zeros(n)
    rock = np.zeros(n)
    depth_ratio = np.zeros(n)   # shoulder amp / hip amp per window
    depth_corr = np.zeros(n)    # zero-lag shoulder-hip correlation
    deep_frame = np.zeros(n)    # 1 where a chest-accented bounce is happening
    for t in range(n):
        a, b = max(0, t - half), min(n, t + half)
        if ok_f[a:b].mean() < 0.6 or (b - a) < w * 0.7:
            continue
        # Reject windows where the tracked body scale jumps (close-up of
        # another dancer, camera zoom): torso should stay near the median.
        tw = np.median(torso[a:b][ok[a:b]]) if ok[a:b].any() else torso_ref
        if tw < 0.6 * torso_ref or tw > 1.6 * torso_ref:
            continue
        hs, ss = hp_sig[a:b], sh_sig[a:b]
        conc = band_concentration(hs, fps, *BOUNCE_BAND)
        amp = float(np.std(hs - hs.mean()))
        bounce[t] = conc if amp > AMP_MIN else 0.0
        # rock: lateral (side-to-side) hip oscillation, independent of bounce
        rxs = hx_sig[a:b]
        conc_r = band_concentration(rxs, fps, *BOUNCE_BAND)
        amp_r = float(np.std(rxs - rxs.mean()))
        rock[t] = conc_r if amp_r > AMP_MIN else 0.0
        # bounce depth: basic = knees only (chest still), deep = chest pumps too.
        # A genuine chest accent needs the SHOULDERS themselves oscillating
        # rhythmically (not just gesturing while hips bounce): require the
        # shoulder signal to carry bounce-band rhythm too.
        sa, ha = float(np.std(ss - ss.mean())), float(np.std(hs - hs.mean()))
        depth_ratio[t] = sa / ha if ha > 1e-9 else 0.0
        c = np.corrcoef(ss, hs)[0, 1]
        depth_corr[t] = float(c) if not np.isnan(c) else 0.0
        sh_conc = band_concentration(ss, fps, *BOUNCE_BAND)
        deep_frame[t] = 1.0 if (bounce[t] > BOUNCE_TH and sh_conc > 0.35
                                and depth_ratio[t] > DEEP_RATIO_MIN
                                and depth_corr[t] > DEEP_CORR_MIN
                                and ha > 0.05) else 0.0
        ml = int(0.5 * fps)
        xc = xcorr(ss, hs, ml)
        c0 = xc[ml]
        lags = np.arange(-ml, ml + 1)
        valid = (np.abs(lags) >= 2) & (np.abs(lags) <= int(0.42 * fps))
        best = xc.copy(); best[~valid] = -1
        j = int(np.argmax(best))
        if best[j] > WAVE_MIN_CORR and (best[j] - c0) > WAVE_MIN_GAIN:
            amp_s, amp_h = float(np.std(ss - ss.mean())), float(np.std(hs - hs.mean()))
            if amp_s > AMP_MIN and amp_h > AMP_MIN:
                wave[t] = float(best[j])

    k = max(1, int(0.5 * fps))
    ker = np.ones(k) / k
    bounce = np.convolve(bounce, ker, mode="same")
    wave = np.convolve(wave, ker, mode="same")
    rock = np.convolve(rock, ker, mode="same")
    # Smooth the depth signals with the same kernel so the per-frame overlay
    # agrees with the per-segment labels in depth_of().
    depth_ratio = np.convolve(depth_ratio, ker, mode="same")
    depth_corr = np.convolve(depth_corr, ker, mode="same")
    np.savez(scores_npz, bounce=bounce, wave=wave, rock=rock,
             depth_ratio=depth_ratio, depth_corr=depth_corr,
             deep_frame=deep_frame, fps=np.array(fps),
             pose_coverage=np.array(pose_cov))
    return bounce, wave, rock, depth_ratio, depth_corr, deep_frame, fps, pose_cov


def segments(score, fps, thr, min_dur=0.5):
    segs, start = [], None
    for i, s in enumerate(score):
        if s >= thr and start is None:
            start = i
        elif s < thr and start is not None:
            if (i - start) / fps >= min_dur:
                segs.append([round(start / fps, 1), round(i / fps, 1),
                             round(float(score[start:i].max()), 2)])
            start = None
    if start is not None and (len(score) - start) / fps >= min_dur:
        segs.append([round(start / fps, 1), round(len(score) / fps, 1),
                     round(float(score[start:].max()), 2)])
    return segs


def render(src, scores_npz, out_mp4, max_width=0):
    """Draw the score overlay. max_width downscales the output (faster
    encode; the overlay is annotated at full res then shrunk)."""
    d = np.load(scores_npz)
    bounce, wave, fps = d["bounce"], d["wave"], float(d["fps"])
    rock = d["rock"] if "rock" in d else np.zeros_like(bounce)
    dratio = d["depth_ratio"] if "depth_ratio" in d else np.zeros_like(bounce)
    dcorr = d["depth_corr"] if "depth_corr" in d else np.zeros_like(bounce)
    dframe = d["deep_frame"] if "deep_frame" in d else np.zeros_like(bounce)
    cap = cv2.VideoCapture(src)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ow, oh = W, H
    if max_width and W > max_width:
        ow = max_width
        oh = (H * max_width // W) // 2 * 2
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_mp4, fourcc, fps, (ow, oh))
    # Deep-bounce segments, labeled with the same rule as the summary
    # (depth_of), so the on-video tag always matches the reported labels.
    _bsegs = segments(bounce, fps, BOUNCE_TH)
    _bsegs = [[s, e, p] + list(depth_of((s, e), fps, dratio, dcorr, dframe))
              for s, e, p in _bsegs]
    deep_ranges = [(int(s * fps), int(e * fps))
                   for s, e, p, lab, r, c in _bsegs if lab == "deep"]
    # phase ranges: down / up / double bounce, stashed by main()
    phase_ranges = []
    if "seg_phase" in d and "seg_bounds" in d:
        for (s, e), typ in zip(d["seg_bounds"], d["seg_phase"]):
            typ = str(typ)
            if typ in ("down", "up", "double"):
                phase_ranges.append((int(float(s) * fps), int(float(e) * fps), typ))
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        b = bounce[i] if i < len(bounce) else 0
        wv = wave[i] if i < len(wave) else 0
        rk = rock[i] if i < len(rock) else 0
        deep = any(lo <= i < hi for lo, hi in deep_ranges)
        ptyp = next((t for lo, hi, t in phase_ranges if lo <= i < hi), "")
        blabel = (ptyp.upper() + " " if ptyp else "") + "BOUNCE" + (" DEEP" if deep else "")
        y = 40
        items = [(blabel, b, (0, 255, 0)),
                 ("ROCK", rk, (255, 0, 255)),
                 ("BODY WAVE", wv, (255, 200, 0))]
        for label, s, color in items:
            if s > 0.35:
                cv2.putText(frame, f"{label} {s:.2f}", (20, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
                bw = int(200 * min(1.0, s))
                cv2.rectangle(frame, (20, y + 8), (20 + bw, y + 18), color, -1)
                y += 55
        t = i / fps
        cv2.putText(frame, f"{int(t // 60)}:{t % 60:04.1f}", (W - 130, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        if (ow, oh) != (W, H):
            frame = cv2.resize(frame, (ow, oh), interpolation=cv2.INTER_AREA)
        out.write(frame)
        i += 1
    cap.release(); out.release()


def depth_of(seg, fps, depth_ratio, depth_corr, deep_frame):
    # A chest accent is brief: label "deep" when a sustained fraction of the
    # segment shows rhythmically-pumping shoulders moving clearly more than
    # the hips, in phase. (Percentile/median on the ratio alone over-fires on
    # upper-body gesturing; his labeled deep bounce at hiphopindia-choreo
    # 0:16 sustains the full deep_frame condition for ~1s.)
    s, e = int(seg[0] * fps), int(seg[1] * fps)
    frac = float(np.mean(deep_frame[s:e])) if e > s else 0.0
    rmax = float(np.max(depth_ratio[s:e])) if e > s else 0.0
    cmed = float(np.median(depth_corr[s:e])) if e > s else 0.0
    return ("deep" if frac > 0.20 else "basic"), round(rmax, 2), round(cmed, 2)


def main():
    src, prefix = sys.argv[1], sys.argv[2]
    pose_npz = f"{prefix}_pose.npz"
    scores_npz = f"{prefix}_scores.npz"
    out_mp4 = f"{prefix}_annotated.mp4"
    summary_path = f"{prefix}_summary.json"
    if not os.path.exists(pose_npz):
        n, fps = extract_pose(src, pose_npz)
        print(f"pose: {n} frames", flush=True)
    bounce, wave, rock, dratio, dcorr, dframe, fps, cov = compute_scores(pose_npz, scores_npz)
    bsegs = segments(bounce, fps, BOUNCE_TH)
    wsegs = segments(wave, fps, WAVE_TH)
    rsegs = segments(rock, fps, 0.45)
    bsegs = [[s, e, p] + list(depth_of((s, e), fps, dratio, dcorr, dframe)) for s, e, p in bsegs]
    dur = len(bounce) / fps
    bcov = sum(e - s for s, e, *_ in bsegs) / dur if dur else 0
    rcov = sum(e - s for s, e, _ in rsegs) / dur if dur else 0
    lock = None
    key = os.path.basename(src).replace(".mp4", "")
    keymap = {"sample": "footwork-combo"}
    akey = keymap.get(key, key)
    try:
        with open("results/audio_bpm.json") as f:
            ab = json.load(f)
        lock = ab.get(akey, {}).get("lock")
    except Exception:
        pass
    # down / up / double bounce from hip-phase vs the beat grid.
    # (Skipped for the footwork-combo tutorial: the demo isn't danced to its
    # background music, so beat phase there is meaningless.)
    phase_types = [""] * len(bsegs)
    phase_lags = [0.0] * len(bsegs)
    if lock in ("single", "half-time", "double-time") and akey != "footwork-combo":
        try:
            with open("results/audio_beats.json") as f:
                beats = json.load(f).get(akey, [])
            try:
                with open("results/audio_kick.json") as f:
                    kk = json.load(f).get(akey, {})
                kick_par = kk.get("parity") if kk.get("ratio", 1.0) >= 1.5 else None
            except Exception:
                kick_par = None
            from bounce_phase import classify_segment, hip_signal
            pt, pdet, _ = hip_signal(pose_npz)
            for i, (s, e, *_) in enumerate(bsegs):
                typ, lag_ms, conf = classify_segment(pt, pdet, (s, e), beats,
                                                     kick_parity=kick_par)
                phase_types[i] = typ
                phase_lags[i] = lag_ms
        except Exception:
            pass
    bsegs = [seg + [typ, lag] for seg, typ, lag in
             zip(bsegs, phase_types, phase_lags)]
    # stash per-segment phase for the renderer
    _d = dict(np.load(scores_npz))
    _d["seg_bounds"] = np.array([[s, e] for s, e, *_ in bsegs])
    _d["seg_phase"] = np.array(phase_types, dtype="U16")
    np.savez(scores_npz, **_d)
    summary = {
        "video": os.path.basename(src), "duration_s": round(dur, 1),
        "pose_coverage": round(cov, 2),
        "bounce_coverage": round(bcov, 2),
        "bounce_segments": bsegs, "wave_segments": wsegs,
        "rock_coverage": round(rcov, 2), "rock_segments": rsegs,
        "bounce_max": round(float(bounce.max()), 2),
        "rock_max": round(float(rock.max()), 2),
        "wave_max": round(float(wave.max()), 2),
        "bpm_lock": lock,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary), flush=True)
    render(src, scores_npz, out_mp4)
    print("wrote", out_mp4, flush=True)


if __name__ == "__main__":
    main()
