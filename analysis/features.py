"""Small, shared live features: fan speed from vibration, and the fan setting.

Used by logger.py (once per second) and readable by any other program.
"""
import numpy as np

SPEED_RANGE = (15.0, 45.0)        # Hz, plausible shaft speeds for this fan
SPEED_STEP = 0.01                 # Hz, candidate-speed resolution
ORDERS = np.arange(4, 49, 4)      # blade-pass family (4 blades): 4x, 8x, ... 48x
MIN_SCORE_DB = 12.0               # comb must stand this far above the noise floor
OFF_RMS = 0.15                    # m/s^2: below this the fan is considered off (noise ~0.05)

# Measured steady speeds for each switch position (Hz). A reading within TOLERANCE of one
# counts as that setting; anything else (spinning up or down) is "CHANGING".
SETTINGS = {"LOW": 25.5, "MEDIUM": 31.7, "HIGH": 36.6}
TOLERANCE = 0.05


def estimate_speed(xyz, fs):
    """Shaft speed (Hz) from a block of 3-axis samples (m/s^2), or None if unclear.

    Scores every candidate speed by how strongly the spectrum lights up at its
    blade-pass orders at once - robust to a second fan or a resonance.
    """
    x = xyz - xyz.mean(axis=0)
    w = np.hanning(len(x))[:, None]
    P = (np.abs(np.fft.rfft(x * w, axis=0)) ** 2).sum(axis=1)
    rel = 10 * np.log10(P + 1e-20)
    rel -= np.median(rel)                                   # dB above noise floor
    df = fs / len(x)

    cands = np.arange(SPEED_RANGE[0], SPEED_RANGE[1], SPEED_STEP)
    F = cands[:, None] * ORDERS[None, :]                    # (candidates, orders)
    valid = F < fs / 2 - 2 * df
    idx = np.clip(np.round(F / df).astype(int), 1, len(rel) - 2)
    near = np.maximum(np.maximum(rel[idx - 1], rel[idx]), rel[idx + 1])
    n_valid = valid.sum(axis=1)
    score = np.where(n_valid >= 4, (near * valid).sum(axis=1) / np.maximum(n_valid, 1), -np.inf)
    best = int(np.argmax(score))
    return (float(cands[best]), float(score[best])) if score[best] >= MIN_SCORE_DB else (None, float(score[best]))


def classify(speed_hz, rms):
    """Fan setting from speed and vibration level."""
    if rms < OFF_RMS:
        return "OFF"
    if speed_hz is None:
        return "UNKNOWN"
    for name, nominal in SETTINGS.items():
        if abs(speed_hz - nominal) <= TOLERANCE * nominal:
            return name
    return "CHANGING"
