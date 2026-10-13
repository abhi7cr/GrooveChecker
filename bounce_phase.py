"""Down / up / double bounce classification.

Compares the phase of the hip-motion fundamental against the audio beat grid:
  down   = physical crouch lands on the beat
  up     = physical rise lands on the beat (reverse)
  double = crouch lands on the 1 AND the & (two dips per beat)

Method: polarity (down vs up) comes from the mean hip signal at beat frames
with NO lag search -- comparing lag-optimized correlations of the signal vs
its inverse lets the wrong polarity "cheat" by shifting up to half a beat.
Timing lag is then measured with a single-polarity cross-correlation against
a beat pulse train, so the answer is a lag in milliseconds, not just on/off.

Peak-picking was tried first and failed: a chest-accented deep bounce splits
each cycle into two dips in the hip signal, scattering picked phases; a
Hilbert-phase version then failed too, because the chest pump pushes the dip
*peak* ~150 ms after the beat even though the attack starts on time. The
cross-correlation lag below measures that directly instead of forcing a
binary on/off verdict.
"""
import numpy as np

SH_L, SH_R, HIP_L, HIP_R = 11, 12, 23, 24


def hip_signal(pose_npz):
    """Detrended, torso-normalized hip-y series. Larger = hips lower = crouch."""
    d = np.load(pose_npz)
    xs, ys, vis = d["xs"], d["ys"], d["vis"]
    fps = float(d["fps"])
    n = len(xs)
    shy = (ys[:, SH_L] + ys[:, SH_R]) / 2
    hpy = (ys[:, HIP_L] + ys[:, HIP_R]) / 2
    hpx = (xs[:, HIP_L] + xs[:, HIP_R]) / 2
    shx = (xs[:, SH_L] + xs[:, SH_R]) / 2
    ok = ((vis[:, SH_L] + vis[:, SH_R]) / 2 > 0.5) & \
         ((vis[:, HIP_L] + vis[:, HIP_R]) / 2 > 0.5)
    torso = np.sqrt((shx - hpx) ** 2 + (shy - hpy) ** 2)
    x = np.arange(n)
    m = ok & ~np.isnan(torso)
    torso_i = np.interp(x, x[m], torso[m])
    hpy_i = np.interp(x, x[ok], hpy[ok])
    tref = float(np.median(torso_i))
    sig = hpy_i / tref
    w = max(3, int(fps))
    det = sig - np.convolve(sig, np.ones(w) / w, mode="same")
    return np.arange(n) / fps, det, fps


def _pulse_train(d, fps, s, times, beat_period):
    """Smoothed pulse train with unit pulses at `times` (seconds)."""
    p = np.zeros_like(d)
    idx = np.clip(((np.asarray(times) - s) * fps).astype(int), 0, len(d) - 1)
    p[idx] = 1.0
    gs = max(1, int(0.04 * beat_period * fps))
    g = np.exp(-0.5 * (np.arange(-3 * gs, 3 * gs + 1) / gs) ** 2)
    return np.convolve(p, g / g.sum(), mode="same")


def _xcorr_best(dn, pulse, fps, beat_period):
    """Best Pearson r between dn and pulse over lags in ±45% of a beat.
    Returns (r, lag_ms); positive lag = motion peaks AFTER the pulses."""
    maxlag = max(1, int(0.45 * beat_period * fps))
    pn = pulse - pulse.mean()
    pden = float(np.sqrt((pn ** 2).sum()))
    if pden < 1e-12:
        return 0.0, 0.0
    best_r, best_l = -2.0, 0
    for l in range(-maxlag, maxlag + 1):
        a = dn[l:] if l >= 0 else dn[:l]
        b = pn[:len(dn) - l] if l >= 0 else pn[-l:]
        n = min(len(a), len(b))
        a, b = a[:n] - a[:n].mean(), b[:n]
        den = float(np.sqrt((a ** 2).sum())) * pden
        r = float((a * b).sum() / den) if den > 1e-12 else 0.0
        if r > best_r:
            best_r, best_l = r, l
    return best_r, best_l / fps * 1000.0


def classify_segment(t, det, seg, beats, kick_parity=None):
    """Returns (type, lag_ms, conf). Type in down/up/double/unresolved/unknown.
    lag_ms > 0 means the motion accent lands AFTER the beat (laid back).
    kick_parity: which beat-index parity (0/1) carries the kick (beats 1&3);
    needed for half-time, where "dip on 1" vs "dip on 2" is otherwise
    indistinguishable from an up bounce."""
    s, e = seg
    beats = np.asarray(beats, dtype=float)
    seg_beats = beats[(beats >= s) & (beats <= e)]
    if len(seg_beats) < 4:
        return "unknown", 0.0, 0.0
    beat_period = float(np.median(np.diff(seg_beats)))
    if not 0.25 < beat_period < 1.5:  # 40..240 BPM sanity
        return "unknown", 0.0, 0.0
    fps = 1.0 / float(np.median(np.diff(t)))
    i0, i1 = int(s * fps), min(len(det), int(e * fps))
    dn = det[i0:i1] - det[i0:i1].mean()
    sd = float(np.std(dn))
    if len(dn) < int(2 * fps) or sd < 1e-9:
        return "unknown", 0.0, 0.0

    # rhythmic in the bounce band? (else phase is meaningless)
    N = 512
    spec = np.abs(np.fft.rfft(dn, N)) ** 2
    freqs = np.fft.rfftfreq(N, 1.0 / fps)
    band = (freqs >= 0.6) & (freqs <= 3.4)
    if spec[band].sum() / (spec.sum() + 1e-12) < 0.4:
        return "unresolved", 0.0, 0.0
    f0 = float(freqs[band][np.argmax(spec[band])])
    beat_rate = 1.0 / beat_period
    ratio = f0 / beat_rate

    def beat_val(times):
        # interpolate (frame-truncation biases up to a frame early)
        return np.interp(np.asarray(times), t[i0:i1], dn)

    if 1.7 <= ratio <= 2.3:
        # double bounce: dips on the 1 AND the & (motion at 2x beat rate)
        grid = np.sort(np.concatenate(
            [seg_beats, (seg_beats[:-1] + seg_beats[1:]) / 2]))
        m = float(np.mean(beat_val(grid))) / sd
        p2 = _pulse_train(dn, fps, s, grid, beat_period)
        r, lag = _xcorr_best(dn, p2, fps, beat_period)
        if m > 0.35 and r > 0.4:
            return "double", round(lag, 0), round(r, 2)
        return "unresolved", round(lag, 0), round(r, 2)

    # beats per bounce cycle (half-time -> 2)
    k = max(1, min(int(round(beat_rate / f0)) if f0 > 0 else 1, 4))
    if k == 2 and kick_parity is not None:
        # dips on the kick parity (beats 1&3) = down; rises there = up.
        # seg_beats[0] is beats[j0] in the full grid: realign the parity.
        j0 = int(np.searchsorted(beats, s))
        db = seg_beats[(kick_parity - j0) % 2::2]
        m = float(np.mean(beat_val(db))) / sd
        pj = _pulse_train(dn, fps, s, db, beat_period)
        r, lag = _xcorr_best(dn if m >= 0 else -dn, pj, fps, beat_period)
        if abs(m) < 0.35 or r < 0.4:
            return "unresolved", round(lag, 0), round(r, 2)
        return ("down" if m > 0 else "up"), round(lag, 0), round(r, 2)
    m_best, j_best = 0.0, 0
    for j in range(k):
        m = float(np.mean(beat_val(seg_beats[j::k]))) / sd
        if abs(m) > abs(m_best):
            m_best, j_best = m, j
    if k == 2:
        # without the kick reference, dip-on-1 vs dip-on-2 is undetermined
        pj = _pulse_train(dn, fps, s, seg_beats[j_best::k], beat_period)
        r, lag = _xcorr_best(dn if m_best >= 0 else -dn, pj, fps, beat_period)
        return "unresolved", round(lag, 0), round(r, 2)
    pj = _pulse_train(dn, fps, s, seg_beats[j_best::k], beat_period)
    r, lag = _xcorr_best(dn if m_best >= 0 else -dn, pj, fps, beat_period)
    if abs(m_best) < 0.35 or r < 0.4:
        return "unresolved", round(lag, 0), round(r, 2)
    return ("down" if m_best > 0 else "up"), round(lag, 0), round(r, 2)
