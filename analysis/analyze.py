"""Analyze a raw accelerometer log from logger.py.

    python analyze.py                      # newest run in ../data
    python analyze.py ..\\data\\run_X.bin   # a specific run
Works from the Run button or the Interactive Window too. Saves PNGs next to the data.

Plots:
  1. Vibration RMS vs time, shaded by condition label
  2. Spectrogram (waterfall): frequency vs time, colour = level
  3. Average spectrum per condition (Welch) with the top peaks listed
  4. Coast-down close-up, if a 'coastdown' label exists
"""
from rawformat import read_bin
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

try:
    HERE = Path(__file__).resolve().parent
except NameError:                                  # Interactive Window
    HERE = Path.cwd()
sys.path.insert(0, str(HERE))

DATA = HERE.parent / "data"

# ---- Settings ----
RMS_WINDOW_S = 0.5           # RMS trend resolution
SPEC_NPERSEG = 2048          # spectrogram: ~0.58 s windows, ~1.7 Hz bins
WELCH_NPERSEG = 8192         # condition spectra: ~0.43 Hz bins
COAST_NPERSEG = 512          # coast-down: ~0.14 s windows for time detail
N_PEAKS = 10
FMAX = None                  # e.g. 1200 to zoom plots; None = Nyquist

# Fixed colour per condition (colour follows the condition, never its order)
COLORS = {"off": "#2a78d6", "low": "#eb6834", "medium": "#1baf7a", "high": "#eda100",
          "oscillate": "#e87ba4", "coastdown": "#6250d6", "start": "#9a9993"}
INK, MUTED, GRID = "#2b2b29", "#6b6b66", "#e4e3df"

plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlecolor": INK,
                     "axes.spines.top": False, "axes.spines.right": False})


def pick_run():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if args:
        return Path(args[0])
    runs = sorted(DATA.glob("run_*.bin"))
    if not runs:
        sys.exit(f"No run_*.bin files in {DATA}")
    return runs[-1]


def load_labels(bin_path, n_samples):
    """Return list of (start_idx, end_idx, label) segments and list of (idx, 'mark')."""
    path = bin_path.with_name(bin_path.stem + "_labels.csv")
    rows = []
    if path.exists():
        with open(path, newline="") as f:
            rows = [(int(r["sample_index"]), r["label"])
                    for r in csv.DictReader(f)]
    marks = [(i, l) for i, l in rows if l == "mark"]
    rows = [(i, l) for i, l in rows if l != "mark"] or [(0, "start")]
    segs = []
    for k, (i, l) in enumerate(rows):
        j = rows[k + 1][0] if k + 1 < len(rows) else n_samples
        if j > i:
            segs.append((i, min(j, n_samples), l))
    return segs, marks


def shade(ax, segs, fs, ymax=None, text=True):
    for i, j, l in segs:
        ax.axvspan(i / fs, j / fs, color=COLORS.get(l,
                   "#9a9993"), alpha=0.10, lw=0)
        if text:
            ax.text((i + j) / 2 / fs, 1.01, l, transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", color=INK, fontsize=9)


def top_peaks(f, amp, n):
    idx = signal.find_peaks(amp)[0]
    idx = idx[np.argsort(amp[idx])[::-1][:n]]
    return sorted(zip(f[idx], amp[idx]), key=lambda p: -p[1])


def main():
    run = pick_run()
    d = read_bin(run)
    xyz, fs = d["xyz"], d["fs"]
    n = len(xyz)
    t = np.arange(n) / fs
    segs, marks = load_labels(run, n)
    fmax = FMAX or fs / 2

    print(f"Run: {run.name}")
    print(
        f"  {n:,} samples, {n / fs:.1f} s, Fs = {fs:.2f} Hz (fitted from Feather clock)")
    print(f"  packets {d['n_packets']:,}; overrun packets {d['overrun_packets']}; "
          f"I2C error packets {d['i2c_error_packets']}; missing samples {d['missing_samples']} "
          f"(interpolated); bad bytes {d['bad_bytes']}")

    ac = xyz - xyz.mean(axis=0)

    # ---- 1. RMS vs time ----
    w = int(RMS_WINDOW_S * fs)
    nb = n // w
    blocks = ac[: nb * w].reshape(nb, w, 3)
    rms = np.sqrt(blocks.var(axis=1).sum(axis=1))
    fig1, ax = plt.subplots(figsize=(11, 3.6))
    shade(ax, segs, fs)
    ax.plot((np.arange(nb) + 0.5) * w / fs, rms, color=INK, lw=1.6)
    for i, _ in marks:
        ax.axvline(i / fs, color=MUTED, ls=":", lw=1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Vibration RMS (m/s²)")
    ax.set_title(
        f"{run.stem}: vibration RMS ({RMS_WINDOW_S} s windows, 3 axes combined)", loc="left", pad=18)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_xlim(0, n / fs)
    fig1.tight_layout()

    # ---- 2. Spectrogram (sum of 3 axes) ----
    f, ts, S = signal.spectrogram(ac, fs=fs, window="hann", nperseg=SPEC_NPERSEG,
                                  noverlap=SPEC_NPERSEG * 3 // 4, axis=0, scaling="spectrum")
    S = S.sum(axis=1)                                   # (freq, time)
    SdB = 10 * np.log10(S + 1e-12)
    fig2, ax = plt.subplots(figsize=(11, 5.5))
    lo, hi = np.percentile(SdB, [5, 99.9])
    m = ax.pcolormesh(ts, f, SdB, shading="auto",
                      cmap="magma", vmin=lo, vmax=hi)
    for i, j, l in segs:
        ax.axvline(i / fs, color="white", lw=0.8, alpha=0.6)
        ax.text((i + j) / 2 / fs, 1.01, l, transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", color=INK, fontsize=9)
    ax.set_ylim(0, fmax)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(f"{run.stem}: spectrogram ({SPEC_NPERSEG}-sample windows, "
                 f"{fs / SPEC_NPERSEG:.2f} Hz bins)", loc="left", pad=18)
    cb = fig2.colorbar(m, ax=ax, pad=0.01)
    cb.set_label("Level (dB re 1 (m/s²)²)")
    cb.outline.set_visible(False)
    fig2.tight_layout()

    # ---- 3. Average spectrum per condition ----
    fig3, ax = plt.subplots(figsize=(11, 5.5))
    print(f"\n{'condition':<10} {'dur(s)':>7} {'rms':>7}   top peaks (Hz : rms m/s²)")
    seen = set()
    for i, j, l in segs:
        if j - i < WELCH_NPERSEG or l == "start":
            continue
        fw, P = signal.welch(ac[i:j], fs=fs, window="hann", nperseg=WELCH_NPERSEG, axis=0,
                             scaling="spectrum")
        # rms amplitude per bin, 3 axes
        amp = np.sqrt(P.sum(axis=1))
        lab = l if l not in seen else None
        seen.add(l)
        ax.semilogy(fw, amp, color=COLORS.get(l, MUTED), lw=1.2, label=lab)
        seg_rms = np.sqrt(ac[i:j].var(axis=0).sum())
        pk = top_peaks(fw, amp, N_PEAKS)
        print(f"{l:<10} {(j - i) / fs:7.1f} {seg_rms:7.3f}   "
              + "  ".join(f"{pf:.1f}:{pa:.3f}" for pf, pa in pk))
    ax.set_xlim(0, fmax)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("RMS amplitude per bin (m/s²)")
    ax.set_title(f"{run.stem}: average spectrum by condition ({fs / WELCH_NPERSEG:.2f} Hz bins)",
                 loc="left")
    ax.grid(True, which="both", color=GRID, lw=0.6)
    ax.legend(frameon=False)
    fig3.tight_layout()

    figs = {"rms": fig1, "spectrogram": fig2, "spectra": fig3}

    # ---- 4. Coast-down close-up ----
    coast = [s for s in segs if s[2] == "coastdown"]
    if coast:
        i, j, _ = coast[0]
        i0, j0 = max(0, i - int(1 * fs)), min(n, i + int(8 * fs))
        fc, tc, Sc = signal.spectrogram(ac[i0:j0], fs=fs, window="hann", nperseg=COAST_NPERSEG,
                                        noverlap=COAST_NPERSEG * 7 // 8, axis=0, scaling="spectrum")
        Sc = 10 * np.log10(Sc.sum(axis=1) + 1e-12)
        fig4, ax = plt.subplots(figsize=(11, 5.5))
        lo, hi = np.percentile(Sc, [5, 99.9])
        m = ax.pcolormesh(tc + i0 / fs, fc, Sc, shading="auto",
                          cmap="magma", vmin=lo, vmax=hi)
        ax.axvline(i / fs, color="white", lw=1, ls="--")
        ax.set_ylim(0, fmax)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_title(f"{run.stem}: coast-down close-up ({COAST_NPERSEG}-sample windows, "
                     f"{COAST_NPERSEG / fs * 1000:.0f} ms)", loc="left")
        cb = fig4.colorbar(m, ax=ax, pad=0.01)
        cb.set_label("Level (dB)")
        cb.outline.set_visible(False)
        fig4.tight_layout()
        figs["coastdown"] = fig4

    for name, fig in figs.items():
        out = run.with_name(f"{run.stem}_{name}.png")
        fig.savefig(out, dpi=150, facecolor="white")
    print(f"\nSaved plots to {run.parent}")
    plt.show()


if __name__ == "__main__":
    main()
