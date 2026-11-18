"""Phone alerts via ntfy when the fan setting changes (and if the logger stops).

One-time setup on the Pi (the topic name is your private "channel" - keep it secret,
it is stored OUTSIDE the git repo so it never reaches GitHub):
    python -c "import secrets; print('fan-' + secrets.token_hex(8))" > ~/.furnace_ntfy_topic
    cat ~/.furnace_ntfy_topic          # subscribe to this name in the ntfy phone app

Run (with logger.py running in another terminal):
    python notifier.py
    python notifier.py --test          # send one test notification and exit

Rules that keep it from being annoying:
  - a new setting must hold for DEBOUNCE_S seconds before it counts (ignores spin-up)
  - at most one alert per setting every COOLDOWN_S seconds
  - transitional readings (CHANGING / UNKNOWN) never trigger anything
"""
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

try:
    HERE = Path(__file__).resolve().parent
except NameError:
    HERE = Path.cwd()
LIVE = HERE.parent / "data" / "live.jsonl"
TOPIC_FILE = Path.home() / ".furnace_ntfy_topic"
SERVER = os.environ.get("NTFY_SERVER", "https://ntfy.sh")

DEBOUNCE_S = 5           # setting must be stable this long
COOLDOWN_S = 60          # per-setting minimum gap between alerts (testing value; raise later)
STALE_S = 120            # alert once if no data arrives for this long
STABLE = {"OFF", "LOW", "MEDIUM", "HIGH"}

# What to send for each new setting: (title, priority 1-5, emoji tags)
#   priority 2 = quiet (no sound), 3 = normal, 4 = high (sound + vibration even if 3 is muted)
MESSAGES = {
    "HIGH":   ("Fan switched to HIGH", 4, "wind_face"),
    "MEDIUM": ("Fan switched to MEDIUM", 3, "dash"),
    "LOW":    ("Fan switched to LOW", 3, "leaves"),
    "OFF":    ("Fan turned off", 2, "zzz"),
}


def topic():
    t = os.environ.get("NTFY_TOPIC") or (TOPIC_FILE.read_text().strip() if TOPIC_FILE.exists() else "")
    if not t:
        sys.exit(f"No topic set. Create one with:\n"
                 f"  python -c \"import secrets; print('fan-' + secrets.token_hex(8))\" > {TOPIC_FILE}")
    return t


def send(title, body, priority=3, tags=""):
    req = urllib.request.Request(
        f"{SERVER}/{topic()}", data=body.encode("utf-8"), method="POST",
        headers={"Title": title, "Priority": str(priority), "Tags": tags})
    try:
        urllib.request.urlopen(req, timeout=10).read()
        print(f"{time.strftime('%H:%M:%S')}  sent: {title} - {body}")
    except Exception as e:                                   # never crash on a network hiccup
        print(f"{time.strftime('%H:%M:%S')}  send failed ({e}): {title}")


def follow(path):
    """Yield new lines appended to a file (like `tail -f`), surviving restarts of the logger."""
    while not path.exists():
        time.sleep(1)
    f = open(path, "r", encoding="utf-8")
    f.seek(0, 2)                                             # start at the end: only new data
    while True:
        line = f.readline()
        if line.endswith("\n"):
            yield line
            continue
        time.sleep(0.5)
        yield None                                           # heartbeat for the stale check
        if path.stat().st_size < f.tell():                   # file was replaced/truncated
            f.close()
            f = open(path, "r", encoding="utf-8")


def main():
    if "--test" in sys.argv:
        send("Fan monitor test", "If you can read this, alerts work.", 3, "white_check_mark")
        return

    print(f"Watching {LIVE}  ->  {SERVER}/<your topic>   (Ctrl+C to stop)")
    current = None                     # last confirmed setting
    candidate, since = None, None      # possible new setting and when it started
    last_sent = {}                     # setting -> time of last alert
    last_data = time.time()
    stale_alerted = False

    for line in follow(LIVE):
        now = time.time()
        if line is None:
            if now - last_data > STALE_S and not stale_alerted:
                send("Fan monitor: no data", f"Nothing from the logger for {STALE_S // 60} min.", 3, "warning")
                stale_alerted = True
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        last_data, stale_alerted = now, False
        s, t = row.get("setting"), row.get("t", now)
        if s not in STABLE:
            continue                                         # spinning up/down: wait
        if s != candidate:
            candidate, since = s, t
            continue
        if s == current or t - since < DEBOUNCE_S:
            continue
        previous, current = current, s                       # the change is confirmed
        if previous is None:
            print(f"{time.strftime('%H:%M:%S')}  starting state: {s} (no alert)")
            continue
        if now - last_sent.get(s, 0) < COOLDOWN_S:
            print(f"{time.strftime('%H:%M:%S')}  {previous} -> {s} (cooldown, not sent)")
            continue
        title, prio, tags = MESSAGES[s]
        speed = row.get("speed_hz")
        body = f"{previous} -> {s}" + (f" ({speed:.1f} Hz, {speed * 60:,.0f} rpm)" if speed else "") \
            + f" at {time.strftime('%I:%M %p').lstrip('0')}"
        send(title, body, prio, tags)
        last_sent[s] = now


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
