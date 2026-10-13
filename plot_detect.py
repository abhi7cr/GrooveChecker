"""Visual explainer: how the bounce and rock detectors work."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

SH_L, SH_R, HIP_L, HIP_R = 11, 12, 23, 24
BOUNCE_BAND = (0.8, 3.0)

d = np.load("pose_tracks.npz")
xs, ys, vis = d["xs"], d["ys"], d["vis"]
fps = float(d["fps"]); n = len(xs)
t = np.arange(n) / fps
shy = (ys[:, SH_L] + ys[:, SH_R]) / 2
hpy = (ys[:, HIP_L] + ys[:, HIP_R]) / 2
hpx = (xs[:, HIP_L] + xs[:, HIP_R]) / 2
shx = (xs[:, SH_L] + xs[:, SH_R]) / 2
ok = ((vis[:, SH_L] + vis[:, SH_R]) / 2 > 0.5) & ((vis[:, HIP_L] + vis[:, HIP_R]) / 2 > 0.5)
torso = np.sqrt((shx - hpx) ** 2 + (shy - hpy) ** 2)
xx = np.arange(n); m = ok & ~np.isnan(torso)
torso_i = np.interp(xx, xx[m], torso[m])
tref = float(np.median(torso_i))
hpy_i = np.interp(xx, xx[ok], hpy[ok]); hpx_i = np.interp(xx, xx[ok], hpx[ok])
vert = hpy_i / tref          # hip height (image y, down positive)
lat = hpx_i / tref           # hip lateral position
vert_d = vert - np.convolve(vert, np.ones(int(fps)) / fps, mode="same")
lat_d = lat - np.convolve(lat, np.ones(int(fps)) / fps, mode="same")

s = np.load("sample_scores.npz")
bounce, rock = s["bounce"], s["rock"]

# --- pick demo windows ---
t0, t1 = 12.0, 26.0                      # trajectory view
w0, w1 = 22.0, 24.0                      # FFT demo window (strong bounce)
wi0, wi1 = int(w0 * fps), int(w1 * fps)
seg = vert_d[wi0:wi1] - vert_d[wi0:wi1].mean()
N = 512
spec = np.abs(np.fft.rfft(seg, N)) ** 2
freqs = np.fft.rfftfreq(N, 1.0 / fps)
band = (freqs >= BOUNCE_BAND[0]) & (freqs <= BOUNCE_BAND[1])
conc = spec[band].sum() / spec.sum()
peak = float(freqs[band][np.argmax(spec[band])])

fig, ax = plt.subplots(4, 1, figsize=(10, 13), sharex=False)
fig.suptitle("How the detectors see your bounce and rock", fontsize=15, y=0.98)

# A: vertical hip trajectory
a = ax[0]
a.plot(t, vert_d, color="#333333", lw=1.2)
a.axvspan(w0, w1, color="gold", alpha=0.25, label="FFT window (panel B)")
a.set_xlim(t0, t1); a.set_ylim(-0.35, 0.35); a.set_ylabel("hip height\n(torso units)")
a.set_title("A  Your hips, up-down: this oscillation IS your bounce", loc="left", fontsize=12)
a.tick_params(labelbottom=False)

# B: spectrum
b = ax[1]
b.plot(freqs, spec / spec.max(), color="#333333", lw=1.4)
b.axvspan(*BOUNCE_BAND, color="green", alpha=0.18)
b.set_xlim(0, 6); b.set_ylim(0, 1.05)
b.set_xlabel("frequency (Hz)"); b.set_ylabel("relative power")
b.set_title(f"B  Fourier transform of the gold window: {conc*100:.0f}% of the energy "
            f"sits in the bounce band, peak at {peak:.2f} Hz", loc="left", fontsize=12)
b.annotate(f"bounce band\n0.8–3 Hz", xy=(1.9, 0.75), fontsize=10, color="darkgreen",
           ha="center", bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="darkgreen"))
b.annotate(f"peak {peak:.2f} Hz", xy=(peak, 1.0), xytext=(peak + 1.2, 0.9),
           arrowprops=dict(arrowstyle="->", color="black"), fontsize=10)

# C: lateral trajectory
c = ax[2]
c.plot(t, lat_d, color="#333333", lw=1.2)
c.set_xlim(t0, t1); c.set_ylim(-0.35, 0.35); c.set_ylabel("hip sideways\n(torso units)")
c.set_title("C  Your hips, side-to-side: rhythmic sway here is the rock", loc="left", fontsize=12)
c.tick_params(labelbottom=False)

# D: scores
e = ax[3]
e.plot(t, bounce, color="green", lw=1.6, label="bounce score")
e.plot(t, rock, color="magenta", lw=1.6, label="rock score")
e.axhline(0.45, color="black", ls="--", lw=1, alpha=0.6)
e.text(t1 - 0.3, 0.47, "threshold 0.45", ha="right", va="bottom", fontsize=9)
e.set_xlim(t0, t1); e.set_ylim(0, 1.05)
e.set_xlabel("time (s)"); e.set_ylabel("score")
e.set_title("D  Score = share of rhythmic energy. Above the line = detected", loc="left", fontsize=12)
e.legend(loc="upper right", fontsize=10)

for a_ in ax:
    a_.grid(alpha=0.25)
    a_.set_xlim(a_.get_xlim())
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("how_detection_works.png", dpi=130)
print("wrote how_detection_works.png")
