"""Shared packet format for the Feather raw-accelerometer stream.

Packet (little-endian):
  0-1   sync 0xA5 0x5A
  2     n samples (1..16)
  3     flags: bit0 FIFO overrun, bit1 I2C error (since previous packet)
  4-7   t_us      uint32 Feather micros() at read time
  8-11  first_idx uint32 running index of first sample in packet
  12..  n x (int16 x, int16 y, int16 z) raw counts
  last  checksum  sum of bytes 2..end-of-samples, mod 256
"""
import struct
import numpy as np

SYNC = b"\xA5\x5A"
HEADER = struct.Struct("<BBII")          # n, flags, t_us, first_idx
MAX_N = 16
G = 9.80665
SCALE = 0.061e-3 * G                     # +/-2 g range: counts -> m/s^2
FLAG_OVERRUN, FLAG_I2C = 0x01, 0x02


class PacketParser:
    """Feed raw bytes in, get complete validated packets out. Resyncs on corruption."""

    def __init__(self):
        self.buf = bytearray()
        self.bad_bytes = 0

    def feed(self, data):
        self.buf.extend(data)
        buf, pos, end = self.buf, 0, len(self.buf)
        packets = []
        while True:
            start = buf.find(SYNC, pos)
            if start < 0:
                keep = 1 if end > pos and buf[end - 1] == SYNC[0] else 0
                self.bad_bytes += end - pos - keep
                pos = end - keep
                break
            self.bad_bytes += start - pos
            pos = start
            if end - pos < 12:
                break
            n, flags, t_us, first_idx = HEADER.unpack_from(buf, pos + 2)
            if not 1 <= n <= MAX_N:
                self.bad_bytes += 1
                pos += 1
                continue
            total = 13 + 6 * n
            if end - pos < total:
                break
            if (sum(buf[pos + 2 : pos + total - 1]) & 0xFF) != buf[pos + total - 1]:
                self.bad_bytes += 1
                pos += 1                          # false sync or corrupt: slide by one byte
                continue
            xyz = np.frombuffer(bytes(buf[pos + 12 : pos + total - 1]), dtype="<i2").reshape(n, 3)
            packets.append((n, flags, t_us, first_idx, xyz))
            pos += total
        del buf[:pos]
        return packets


def read_bin(path):
    """Parse a whole .bin log. Returns dict with samples (m/s^2), timing and quality info."""
    parser = PacketParser()
    with open(path, "rb") as f:
        packets = parser.feed(f.read())
    if not packets:
        raise ValueError(f"No valid packets in {path}")

    first_idx = np.array([p[3] for p in packets], dtype=np.int64)
    n = np.array([p[0] for p in packets], dtype=np.int64)
    t_us = np.array([p[2] for p in packets], dtype=np.int64)
    flags = np.array([p[1] for p in packets], dtype=np.int64)

    # Unwrap the 32-bit counters (micros() wraps every ~71 minutes)
    first_idx = np.concatenate([[first_idx[0]], first_idx[0] + np.cumsum(np.diff(first_idx) % 2**32)])
    t_us = np.concatenate([[t_us[0]], t_us[0] + np.cumsum(np.diff(t_us) % 2**32)])

    base = first_idx[0]
    total = int(first_idx[-1] + n[-1] - base)
    xyz = np.full((total, 3), np.nan)
    for (k, _, _, _, s), i0 in zip(packets, first_idx - base):
        xyz[i0 : i0 + k] = s * SCALE
    missing = np.isnan(xyz[:, 0])
    n_missing = int(missing.sum())
    if n_missing:                                      # bridge host-side gaps by interpolation
        idx = np.arange(total)
        for a in range(3):
            xyz[missing, a] = np.interp(idx[missing], idx[~missing], xyz[~missing, a])

    # Sample rate: straight-line fit of device time vs sample index (packet end)
    end_idx = first_idx + n - base
    slope, _ = np.polyfit(end_idx, t_us - t_us[0], 1)  # microseconds per sample
    fs = 1e6 / slope

    return {
        "xyz": xyz,
        "fs": fs,
        "base_idx": int(base),
        "n_packets": len(packets),
        "overrun_packets": int(((flags & FLAG_OVERRUN) > 0).sum()),
        "i2c_error_packets": int(((flags & FLAG_I2C) > 0).sum()),
        "missing_samples": n_missing,
        "bad_bytes": parser.bad_bytes,
    }
