"""Campbell diagram: vibration spectrum vs shaft speed, from spin-ups, speed changes and coast-downs.

    python campbell.py                     # newest run in ../data
    python campbell.py ..\\data\\run_X.bin  # a specific run

How it works
  1. Short-time spectra through the whole run (spectrogram).
  2. Each time slice gets a shaft-speed estimate from the vibration itself (a "virtual
     tachometer"): try every candidate speed and score how strongly the spectrum lights up
     at that speed's blade-pass orders (4x, 8x, 12x ...). The best-scoring speed wins.
  3. Slices are re-sorted by shaft speed instead of time and drawn as an image:
     x = shaft speed, y = frequency, colour = level.

Reading it
  - Straight rays from the origin = shaft orders (forcing that scales with speed).
  - Horizontal bright bands = structural resonances (fixed frequency, lit up by
    whichever order or broadband noise passes through them).
  - Where a ray crosses a horizontal band and brightens = order exciting a resonance.

Also saves a plot of the estimated shaft speed vs time, to check the virtual tachometer.
Works best with runs containing spin-ups and a coast-down (more speeds covered).
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

try:
    HERE = Path(__file__).resolve().parent
except NameError:                              # Interactive Window
    HERE = Path.cwd()
sys.path.insert(0, str(HERE))
from rawformat import read_bin
from analyze import load_labels, pick_run, COLORS, INK, MUTED, GRID

# ---- Settings ----
NPERSEG = 1024            # ~0.29 s windows: compromise between time and frequency detail
HOP = 128                 # step between windows (~36 ms)
F0_RANGE = (6.0, 45.0)    # shaft speeds to consider (Hz)
F0_STEP = 0.05            # candidate-speed resolution (Hz)
ORDERS = list(range(4, 49, 4))   # blade-pass family used by the virtual tachometer (4 blades)
MIN_SCORE_DB = 15.0       # slices scoring below this (fan stopped / unclear) are dropped
MAX_JUMP = 0.05           # drop slices whose speed differs >5% from their neighbours (glitches)
SPEED_BIN = 0.25          # Hz, x-axis binning of the diagram
RAYS = [1, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48]
FMAX = None               # e.g. 1200 to zoom; None = Nyquist


def virtual_tach(f, SdB, fs):
    """Shaft speed for each time slice from the comb of blade-pass orders."""
    noise = np.median(SdB, axis=0, keepdims=True)          # per-slice noise floor (dB)
    rel = SdB - noise                                      # level above floor
    cands = np.arange(F0_RANGE[0], F0_RANGE[1], F0_STEP)
    df = f[1] - f[0]
    score = np.full((len(cands), SdB.shape[1]), -np.inf)
    for ci, f0 in enumerate(cands):
        ks = [k for k in ORDERS if k * f0 < fs / 2 - df]
        if len(ks) < 4:
            continue
        idx = np.round(np.array(ks) * f0 / df).astype(int)
        # best of the 3 bins nearest each order (tolerates slight speed error), averaged over orders
        win = np.stack([rel[np.clip(idx + s, 0, len(f) - 1)] for s in (-1, 0, 1)])
        score[ci] = win.max(axis=0).mean(axis=0)
    best = score.argmax(axis=0)
    return cands[best], score[best, np.arange(SdB.shape[1])]


def main():
    run = pick_run()
    d = read_bin(run)
    xyz, fs = d["xyz"], d["fs"]
    ac = xyz - xyz.mean(axis=0)
    segs, _ = load_labels(run, len(xyz))
    fmax = FMAX or fs / 2

    f, t, S = signal.spectrogram(ac, fs=fs, window="hann", nperseg=NPERSEG,
                                 noverlap=NPERSEG - HOP, axis=0, scaling="spectrum")
    SdB = 10 * np.log10(S.sum(axis=1) + 1e-12)              # (freq, time), 3 axes combined
    f0, score = virtual_tach(f, SdB, fs)
    ok = score >= MIN_SCORE_DB
    # Continuity check: a real shaft can't jump; compare each slice with the median of its neighbours
    idx = np.flatnonzero(ok)
    for n_, i in enumerate(idx):
        nb = f0[idx[max(0, n_ - 5): n_ + 6]]
        if abs(f0[i] - np.median(nb)) > MAX_JUMP * np.median(nb):
            ok[i] = False
    print(f"Run {run.name}: {len(t)} slices, {ok.sum()} with a confident shaft speed "
          f"({f0[ok].min():.1f}-{f0[ok].max():.1f} Hz)" if ok.any() else "No confident slices")
    if not ok.any():
        return

    # ---- Bin slices by shaft speed and average (in dB) ----
    edges = np.arange(F0_RANGE[0], F0_RANGE[1] + SPEED_BIN, SPEED_BIN)
    which = np.digitize(f0[ok], edges) - 1
    img = np.full((len(f), len(edges) - 1), np.nan)
    count = np.zeros(len(edges) - 1, int)
    for b in np.unique(which):
        cols = SdB[:, ok][:, which == b]
        img[:, b] = cols.mean(axis=1)
        count[b] = cols.shape[1]
    centers = (edges[:-1] + edges[1:]) / 2
    used = count > 0

    # ---- Figure 1: Campbell diagram ----
    fig, ax = plt.subplots(figsize=(11, 7))
    vals = img[:, used]
    lo, hi = np.nanpercentile(vals, [5, 99.7])
    m = ax.pcolormesh(centers, f, img, shading="nearest", cmap="magma", vmin=lo, vmax=hi)
    xmax = centers[used].max() + 2
    for k in RAYS:
        x_end = min(xmax, fmax / k)
        ax.plot([0, x_end], [0, k * x_end], color="white", lw=0.6, alpha=0.45, ls="--")
        if k * xmax <= fmax:
            ax.text(xmax, k * xmax, f" {k}×", color=MUTED, fontsize=8, va="center")
        else:
            ax.text(fmax / k, fmax, f"{k}×", color=MUTED, fontsize=8, ha="center", va="bottom")
    ax.set_xlim(max(F0_RANGE[0], centers[used].min() - 2), xmax)
    ax.set_ylim(0, fmax)
    ax.set_xlabel("Shaft speed (Hz)   [× 60 = rpm]")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(f"{run.stem}: Campbell diagram (rays = shaft orders; horizontal bands = resonances)",
                 loc="left", pad=18)
    cb = fig.colorbar(m, ax=ax, pad=0.06)
    cb.set_label("Level (dB re 1 (m/s²)²)")
    cb.outline.set_visible(False)
    fig.tight_layout()

    # ---- Figure 2: virtual tachometer vs time ----
    fig2, ax2 = plt.subplots(figsize=(11, 3.6))
    for i, j, l in segs:
        ax2.axvspan(i / fs, j / fs, color=COLORS.get(l, "#9a9993"), alpha=0.10, lw=0)
        ax2.text((i + j) / 2 / fs, 1.01, l, transform=ax2.get_xaxis_transform(),
                 ha="center", va="bottom", color=INK, fontsize=9)
    ax2.plot(t[ok], f0[ok], ".", color=INK, ms=2.5)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Shaft speed (Hz)")
    ax2.set_title(f"{run.stem}: shaft speed estimated from vibration (virtual tachometer)",
                  loc="left", pad=18)
    ax2.grid(True, color=GRID, lw=0.8)
    ax2.set_xlim(0, len(xyz) / fs)
    fig2.tight_layout()

    # ---- Steady-state speeds per condition (median of confident slices) ----
    print(f"\n{'condition':<10} {'shaft Hz':>9} {'rpm':>6}")
    for i, j, l in segs:
        sel = ok & (t >= i / fs + 2) & (t <= j / fs - 1)
        if sel.sum() > 5 and l not in ("start", "off", "coastdown"):
            s = np.median(f0[sel])
            print(f"{l:<10} {s:9.2f} {s * 60:6.0f}")

    for name, fg in (("campbell", fig), ("tach", fig2)):
        fg.savefig(run.with_name(f"{run.stem}_{name}.png"), dpi=150, facecolor="white")
    print(f"\nSaved {run.stem}_campbell.png and _tach.png")
    plt.show()


if __name__ == "__main__":
    main()
