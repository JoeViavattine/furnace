"""Log the Feather's raw accelerometer stream to disk, with keypress condition labels.

Run from a terminal (not the Interactive Window - keypresses need a real console).
Works on Windows and on Linux / Raspberry Pi (port e.g. /dev/ttyACM0, auto-detected):
    python logger.py                 # auto-detects the Feather's COM port
    python logger.py --port COM3     # or name it

Keys while logging (edit LABEL_KEYS to suit):
    0 off  1 low  2 medium  3 high  4 oscillate  c coastdown  m mark (event only)  q quit
Output (in ../data):
    run_YYYYMMDD_HHMMSS.bin          exact byte stream from the Feather (lossless)
    run_YYYYMMDD_HHMMSS_labels.csv   sample_index, host_time, label
    live.jsonl                       one line per second: time, fan speed, setting, RMS
                                     (read by dashboard.py; shared across runs)
"""
import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import serial
from serial.tools import list_ports

from rawformat import PacketParser, SCALE, FLAG_OVERRUN, FLAG_I2C
from features import estimate_speed, classify

SPEED_BLOCK = 8192           # samples used for each speed estimate (~2.3 s at ~3.56 kHz)

LABEL_KEYS = {"0": "off", "1": "low", "2": "medium", "3": "high", "4": "oscillate",
              "c": "coastdown", "m": "mark"}
FEATHER_VIDS = {0x239A, 0x303A}          # Adafruit, Espressif

try:
    import msvcrt                         # Windows: non-blocking keyboard

    def keys_on():
        pass

    def keys_off():
        pass

    def get_key():
        return msvcrt.getwch().lower() if msvcrt.kbhit() else None
except ImportError:                       # Linux / Raspberry Pi: single-key reads via termios
    import select
    import termios
    import tty
    _saved_tty = None

    def keys_on():
        global _saved_tty
        if sys.stdin.isatty():
            _saved_tty = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())   # keys arrive immediately, no Enter needed

    def keys_off():
        if _saved_tty is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, _saved_tty)

    def get_key():
        if sys.stdin.isatty() and select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1).lower()
        return None

try:
    HERE = Path(__file__).resolve().parent
except NameError:
    HERE = Path.cwd()


def find_port():
    ports = [p for p in list_ports.comports() if p.vid in FEATHER_VIDS]
    if len(ports) == 1:
        return ports[0].device
    names = ", ".join(f"{p.device} ({p.description})" for p in list_ports.comports()) or "none"
    sys.exit(f"Couldn't pick the Feather's port automatically. Ports seen: {names}\n"
             f"Run again with --port COMx")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", help="serial port, e.g. COM3 (default: auto-detect)")
    ap.add_argument("--out", default=str(HERE.parent / "data"), help="output folder")
    ap.add_argument("--name", default="", help="short tag added to the file name, e.g. tape_radius")
    args = ap.parse_args()

    port = args.port or find_port()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = datetime.now().strftime("run_%Y%m%d_%H%M%S") + (f"_{args.name}" if args.name else "")
    bin_path, lab_path = out / f"{stem}.bin", out / f"{stem}_labels.csv"

    ser = serial.Serial(port, 115200, timeout=0.05)   # baud is ignored by native USB
    ser.reset_input_buffer()
    parser = PacketParser()

    print(f"Logging {port} -> {bin_path.name}")
    print("Keys: " + "  ".join(f"{k}={v}" for k, v in LABEL_KEYS.items()) + "  q=quit\n")

    total = 0                  # samples received
    expected = None            # next first_idx we expect from the Feather
    base = None                # first_idx of the first packet (labels are relative to it)
    dropped = overruns = i2c_errs = 0
    fs_ref = None              # (sample_index, unwrapped t_us) of first packet
    t_unwrap = last_t = None
    window = []                # samples since last status line
    recent = []                # rolling raw samples for the speed estimate
    n_recent = 0
    live = open(out / "live.jsonl", "a", buffering=1)   # line-buffered: each line lands at once
    label = "start"
    t_status = time.time()
    settle_until = time.time() + 1.5   # ignore the stale burst the Feather queued before we connected

    with open(bin_path, "wb") as fbin, open(lab_path, "w", newline="") as flab:
        lab = csv.writer(flab)
        lab.writerow(["sample_index", "host_time", "label"])
        lab.writerow([0, f"{time.time():.3f}", label])
        keys_on()
        try:
            while True:
                data = ser.read(ser.in_waiting or 1)
                if data:
                    fbin.write(data)
                    for n, flags, t_us, first_idx, xyz in parser.feed(data):
                        if base is None:
                            base = first_idx
                        if expected is not None and first_idx != expected:
                            dropped += (first_idx - expected) % 2**32
                        expected = (first_idx + n) % 2**32
                        overruns += bool(flags & FLAG_OVERRUN)
                        i2c_errs += bool(flags & FLAG_I2C)
                        t_unwrap = t_us if last_t is None else t_unwrap + ((t_us - last_t) % 2**32)
                        last_t = t_us
                        total += n
                        if time.time() < settle_until:    # start-up: flush counters, re-anchor Fs
                            dropped = overruns = i2c_errs = 0
                            parser.bad_bytes = 0
                            fs_ref = None
                        if fs_ref is None:
                            fs_ref = (total, t_unwrap)
                        window.append(xyz)
                        recent.append(xyz)
                        n_recent += n
                        while n_recent - len(recent[0]) >= SPEED_BLOCK:
                            n_recent -= len(recent.pop(0))

                key = get_key()
                if key == "q":
                    break
                if key in LABEL_KEYS:
                    if LABEL_KEYS[key] != "mark":
                        label = LABEL_KEYS[key]
                    idx = 0 if base is None else (expected - base) % 2**32
                    lab.writerow([idx, f"{time.time():.3f}", LABEL_KEYS[key]])
                    flab.flush()
                    print(f"\n  >> {LABEL_KEYS[key]} at sample {idx:,}")

                if time.time() - t_status >= 1.0:
                    t_status = time.time()
                    fs = 0.0
                    if fs_ref and t_unwrap > fs_ref[1]:
                        fs = (total - fs_ref[0]) / ((t_unwrap - fs_ref[1]) / 1e6)
                    rms = 0.0
                    if window:
                        w = np.vstack(window) * SCALE
                        rms = float(np.sqrt((w.var(axis=0)).sum()))
                        window = []
                    speed, setting = None, "UNKNOWN"
                    if n_recent >= SPEED_BLOCK:
                        blk = np.vstack(recent)[-SPEED_BLOCK:] * SCALE
                        speed, _ = estimate_speed(blk, fs if fs > 3000 else 3558.0)
                        setting = classify(speed, rms)
                        if setting == "OFF":
                            speed = None
                    live.write(json.dumps({
                        "t": round(time.time(), 2), "run": stem, "label": label,
                        "setting": setting,
                        "speed_hz": None if speed is None else round(speed, 2),
                        "rms": round(rms, 4), "fs": round(fs, 1),
                        "ok": not (overruns or dropped or i2c_errs)}) + "\n")
                    warn = []
                    if overruns: warn.append(f"OVERRUN x{overruns}")
                    if dropped: warn.append(f"DROPPED {dropped}")
                    if i2c_errs: warn.append(f"I2C ERR x{i2c_errs}")
                    if parser.bad_bytes: warn.append(f"bad bytes {parser.bad_bytes}")
                    spd = f"{speed:5.2f} Hz" if speed else "  --   "
                    print(f"\r[{label:>9}] samples {total:>9,}  Fs {fs:7.1f} Hz  rms {rms:6.3f} m/s2  "
                          f"fan {spd} {setting:<7} "
                          + ("  ".join(warn) if warn else "ok") + "      ", end="", flush=True)
                    fbin.flush()
        except KeyboardInterrupt:
            pass
        finally:
            keys_off()
            ser.close()
            live.close()

    print(f"\n\nSaved {bin_path}\n      {lab_path}")
    print(f"{total:,} samples, overrun packets {overruns}, dropped {dropped}, "
          f"I2C errors {i2c_errs}, bad bytes {parser.bad_bytes}")


if __name__ == "__main__":
    main()
