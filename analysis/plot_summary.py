"""Plot the Feather peak-summary log: RMS vs time and a peak map (frequency vs time)."""
import re
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Works from the command line, the Run button, or the Interactive Window
try:
    HERE = Path(__file__).resolve().parent
except NameError:                       # Interactive Window / Jupyter
    HERE = Path.cwd()
DATA = HERE.parent / "data"
# ignore Jupyter's own args
args = [a for a in sys.argv[1:] if not a.startswith("-")]
path = Path(args[0]) if len(args) > 0 else DATA / "fan_test_2.txt"
out = Path(args[1]) if len(args) > 1 else DATA / "fan_summary.png"

line_re = re.compile(r"t=([\d.]+)s Fs=([\d.]+)Hz rms=([\d.]+) m/s2 \|(.*)")
peak_re = re.compile(r"([\d.]+)Hz:([\d.]+)")

t, rms, pt, pf, pa = [], [], [], [], []
for line in open(path, encoding="utf-8", errors="ignore"):
    m = line_re.search(line)
    if not m:
        continue
    ti = float(m.group(1))
    t.append(ti)
    rms.append(float(m.group(3)))
    for f, a in peak_re.findall(m.group(4)):
        pt.append(ti)
        pf.append(float(f))
        pa.append(float(a))

t, rms = np.array(t), np.array(rms)
pt, pf, pa = np.array(pt), np.array(pf), np.array(pa)

# Conditions (approximate, from the run notes)
conds = [(0, 60, "Off"), (62, 80, "Low"), (82, 100, "High"),
         (102, 120, "Off")]

ink, muted, grid = "#2b2b29", "#6b6b66", "#e4e3df"
blue = "#2a78d6"
cmap = LinearSegmentedColormap.from_list(
    "blue", ["#b7d3f6", "#5598e7", "#1c5cab", "#0d366b"])

plt.rcParams.update({"font.size": 10, "axes.edgecolor": muted, "axes.labelcolor": ink,
                     "xtick.color": muted, "ytick.color": muted, "font.family": "DejaVu Sans"})
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8.5), sharex=True,
                               gridspec_kw={"height_ratios": [1, 2.2], "hspace": 0.12})

for a in (ax1, ax2):
    for i, (x0, x1, lab) in enumerate(conds):
        a.axvspan(x0, x1, color="#f0efec" if i %
                  2 == 0 else "#f7f6f3", zorder=0, lw=0)
    a.grid(True, color=grid, lw=0.8, zorder=1)
    a.spines[["top", "right"]].set_visible(False)

for x0, x1, lab in conds:
    ax1.text((x0 + x1) / 2, 1.13, lab, ha="center",
             va="bottom", color=ink, fontsize=10)

ax1.plot(t, rms, color=blue, lw=2, marker="o", ms=4, zorder=3)
ax1.set_ylabel("Vibration RMS (m/s²)")
ax1.set_ylim(0, 1.25)
ax1.set_xlim(0, 85)
ax1.axhline(0.035, color=muted, ls=":", lw=1.2, zorder=2)
ax1.text(1, 0.06, "Sensor noise floor (datasheet ≈ 0.035)",
         color=muted, fontsize=9)
ax1.set_title("Desk fan vibration — RMS and dominant spectral peaks (Fs ≈ 1780 Hz, Nyquist ≈ 890 Hz)",
              loc="left", color=ink, fontsize=12, pad=26)

# Order lines for low (26.1 Hz) and high (37.0 Hz) shaft rates
for (x0, x1), f0, orders in [((23, 44), 26.1, [16, 20, 28, 32]), ((46, 84), 37.0, [4, 12, 16, 20, 24])]:
    for n in orders:
        ax2.hlines(n * f0, x0, x1, colors=muted,
                   linestyles="--", lw=0.9, zorder=2)
        ax2.text(x1 - 0.3, n * f0 + 6, f"×{n}",
                 ha="right", color=muted, fontsize=8)

# Aliased content annotations (folded from above Nyquist)
for x0, x1, f, lab in [(23, 44, 839.6, "alias of ×36"), (23, 44, 735.2, "alias of ×40"), (46, 84, 743.3, "alias of ×28")]:
    ax2.text(x0 + 0.3, f + 6, lab, color="#b8460e", fontsize=8)

big = pa >= 0.05
ax2.scatter(pt[~big], pf[~big], s=10, color="#c9c8c3", zorder=3, )
sc = ax2.scatter(pt[big], pf[big], s=20 + 220 * pa[big], c=pa[big], cmap=cmap, vmin=0.05, vmax=0.8,
                 edgecolors="white", linewidths=0.8, zorder=4)
ax2.axhline(1780 / 2, color="#b8460e", lw=1.2, zorder=2)
ax2.text(1, 896, "Nyquist (Fs/2)", color="#b8460e", fontsize=9)
ax2.text(64.5, 85, "120 Hz: oscillation motor hum", color=ink, fontsize=8)
ax2.set_ylim(0, 1200)
ax2.set_xlim(0, 120)
ax2.set_ylabel("Peak frequency (Hz)")
ax2.set_xlabel("Time (s)")
cb = fig.colorbar(sc, ax=[ax1, ax2], pad=0.01,
                  fraction=0.02, shrink=0.6, anchor=(0, 0.0))
cb.set_label("Peak amplitude (m/s²)", color=ink)
cb.outline.set_visible(False)
ax2.text(1, 330, "Grey dots: peaks < 0.05 m/s²\n(noise-level)",
         color=muted, fontsize=8)
ax2.text(23.3, 300, "Dashed: shaft orders\n(low 26.1 Hz, high 37.0 Hz)",
         color=muted, fontsize=8)

fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
print("saved", out, len(t), "blocks", len(pf), "peaks")
plt.show()
