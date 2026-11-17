"""Before/after comparison around a mark ('m' key) within one run.

    python compare.py                    # newest run, first mark
    python compare.py ..\\data\\run_X.bin  # specific run
    python compare.py run_X.bin 2        # use the 2nd mark

Takes the steady data before and after the mark (same condition segment),
skipping SKIP_S seconds on each side of the mark to avoid hand-contact transients.
Prints a shaft-order table (before, after, change in dB, overall and per axis)
and plots the two spectra plus the dB change at each order.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

try:
    HERE = Path(__file__).resolve().parent
except NameError:
    HERE = Path.cwd()
sys.path.insert(0, str(HERE))
from rawformat import read_bin
from analyze import load_labels, pick_run, INK, MUTED, GRID

SKIP_S = 3.0               # ignore this much either side of a mark (hand contact, speed settling)
NPERSEG = 8192             # ~0.43 Hz bins
SHAFT_RANGE = (15, 70)     # Hz, where to look for the 1x shaft peak
BEFORE_C, AFTER_C = "#2a78d6", "#eb6834"


def spectrum(x, fs):
    f, P = signal.welch(x - x.mean(axis=0), fs=fs, window="hann", nperseg=NPERSEG, axis=0,
                        scaling="spectrum")
    return f, np.sqrt(P)                     # rms amplitude per bin, per axis (freq, 3)


def shaft_rate(f, tot):
    """Shaft speed from the whole comb of blade-pass orders (4x, 8x, ... up to Nyquist).
    Scoring every candidate speed against all orders at once is robust when a second
    fan or a resonance makes one individual peak misleading. Resolution ~0.01 Hz."""
    rel = 20 * np.log10(tot + 1e-12)
    rel -= np.median(rel)                       # level above the noise floor, dB
    df = f[1] - f[0]
    best, best_score = None, -np.inf
    for f0 in np.arange(SHAFT_RANGE[0], SHAFT_RANGE[1], 0.01):
        ks = np.arange(4, int(f[-1] / f0) + 1, 4)
        if len(ks) < 4:
            continue
        idx = np.round(ks * f0 / df).astype(int)
        near = np.max([rel[np.clip(idx + s, 0, len(f) - 1)] for s in (-1, 0, 1)], axis=0)
        score = near.mean()
        if score > best_score:
            best, best_score = f0, score
    return best


def order_amp(f, amp, f_target, tol=0.004):
    """RMS of the tone near f_target: find the peak bin, sum power over +/-2 bins.
    Hann window spreads a tone's power over its main lobe; dividing by the
    window's noise bandwidth (1.5 bins) recovers the tone's true rms."""
    sel = np.where((f > f_target * (1 - tol)) & (f < f_target * (1 + tol)))[0]
    if len(sel) == 0:
        return None
    p = sel[np.argmax((amp[sel] ** 2).sum(axis=1))]
    lo, hi = max(p - 2, 0), min(p + 3, len(f))
    a = np.sqrt((amp[lo:hi] ** 2).sum(axis=0) / 1.5)
    return np.sqrt((a ** 2).sum()), a


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    which = int(args[1]) if len(args) > 1 else 1
    run = pick_run()
    d = read_bin(run)
    xyz, fs = d["xyz"], d["fs"]
    segs, marks = load_labels(run, len(xyz))
    if len(marks) < which:
        sys.exit("No mark found - press 'm' during logging at the moment of the change.")
    m = marks[which - 1][0]
    seg = next(s for s in segs if s[0] <= m < s[1])
    # Window = from the previous boundary (segment start or earlier mark) to this mark,
    # and from this mark to the next boundary (next mark or segment end).
    marks_idx = sorted(i for i, _ in marks)
    start = max([seg[0]] + [i for i in marks_idx if i < m])
    end = min([seg[1]] + [i for i in marks_idx if i > m])
    skip = int(SKIP_S * fs)
    a, b = xyz[start + skip : m - skip], xyz[m + skip : end - skip]
    if min(len(a), len(b)) < NPERSEG * 2:
        sys.exit("Not enough steady data on one side of the mark (need ~5 s each side).")

    f, A = spectrum(a, fs)
    _, B = spectrum(b, fs)
    tA, tB = np.sqrt((A ** 2).sum(axis=1)), np.sqrt((B ** 2).sum(axis=1))

    f0, f1 = shaft_rate(f, tA), shaft_rate(f, tB)   # each side gets its own shaft rate
    print(f"Run {run.name}, mark {which} at {m / fs:.1f} s in '{seg[2]}' segment")
    print(f"  before: {len(a) / fs:.1f} s   after: {len(b) / fs:.1f} s")
    print(f"  shaft before {f0:.3f} Hz   after {f1:.3f} Hz   change {100 * (f1 / f0 - 1):+.2f}%")
    print(f"  overall rms  before {np.sqrt(a.var(axis=0).sum()):.4f}   after {np.sqrt(b.var(axis=0).sum()):.4f} m/s2\n")

    print(f"{'order':>5} {'freq':>8} {'before':>8} {'after':>8} {'dB':>6}   {'dBx':>6} {'dBy':>6} {'dBz':>6}")
    rows = []
    for k in range(1, int((fs / 2) / f0) + 1):
        ra, rb = order_amp(f, A, k * f0), order_amp(f, B, k * f1)
        if ra is None or rb is None:
            continue
        (ta, axa), (tb, axb) = ra, rb
        if max(ta, tb) < 0.003:              # skip orders lost in the noise
            continue
        db = 20 * np.log10(tb / ta)
        dbx = 20 * np.log10(axb / axa)
        rows.append((k, db))
        print(f"{k:>5} {k * f0:8.1f} {ta:8.4f} {tb:8.4f} {db:+6.1f}   "
              + " ".join(f"{v:+6.1f}" for v in dbx))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7.5), gridspec_kw={"height_ratios": [2, 1]})
    ax1.semilogy(f, tA, color=BEFORE_C, lw=1.1, label="before mark")
    ax1.semilogy(f, tB, color=AFTER_C, lw=1.1, label="after mark", alpha=0.85)
    ax1.set_xlim(0, fs / 2)
    ax1.set_xlabel("Frequency (Hz)")
    ax1.set_ylabel("RMS amplitude per bin (m/s²)")
    ax1.set_title(f"{run.stem}: before vs after mark {which} ('{seg[2]}', shaft {f0:.2f} -> {f1:.2f} Hz)",
                  loc="left")
    ax1.grid(True, which="both", color=GRID, lw=0.6)
    ax1.legend(frameon=False)
    if rows:
        ks, dbs = zip(*rows)
        ax2.bar(ks, dbs, color=[AFTER_C if v > 0 else BEFORE_C for v in dbs], width=0.7)
    ax2.axhline(0, color=INK, lw=0.8)
    for y in (-3, 3):
        ax2.axhline(y, color=MUTED, lw=0.8, ls=":")
    ax2.set_xlabel("Shaft order")
    ax2.set_ylabel("Change (dB)")
    ax2.set_title("Change at each shaft order, each side at its own shaft speed (dotted: ±3 dB)", loc="left")
    ax2.grid(True, axis="y", color=GRID, lw=0.6)
    fig.tight_layout()
    out = run.with_name(f"{run.stem}_compare_mark{which}.png")
    fig.savefig(out, dpi=150, facecolor="white")
    print(f"\nSaved {out.name}")
    plt.show()


if __name__ == "__main__":
    main()
