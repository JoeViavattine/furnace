"""Live fan dashboard for a phone or laptop browser.

On the Pi, with logger.py running in another terminal:
    streamlit run dashboard.py --server.address 0.0.0.0 --server.headless true --client.toolbarMode viewer
Then on any device on the home Wi-Fi:  http://10.0.0.250:8501

Reads ../data/live.jsonl (one line per second, written by logger.py) and refreshes
every 2 seconds. It only reads - it can never disturb data collection.
"""
import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

try:
    HERE = Path(__file__).resolve().parent
except NameError:
    HERE = Path.cwd()
LIVE = HERE.parent / "data" / "live.jsonl"

HISTORY_MIN = 10          # minutes shown in the charts
STALE_S = 5               # warn if the newest line is older than this
REFRESH = "2s"


def read_tail(path, max_bytes=500_000):
    """Last part of the live file as a DataFrame (cheap, however large the file grows)."""
    if not path.exists():
        return pd.DataFrame()
    with open(path, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - max_bytes))
        lines = f.read().decode("utf-8", errors="ignore").splitlines()[1:] if size > max_bytes \
            else f.read().decode("utf-8", errors="ignore").splitlines()
    rows = []
    for ln in lines:
        try:
            rows.append(json.loads(ln))
        except json.JSONDecodeError:
            pass                                   # a line being written right now
    df = pd.DataFrame(rows)
    if not df.empty:
        df["time"] = pd.to_datetime(df["t"].map(datetime.fromtimestamp))
    return df


st.set_page_config(page_title="Fan Monitor", page_icon="🌀", layout="centered")
st.title("Fan monitor")


@st.fragment(run_every=REFRESH)
def live_view():
    df = read_tail(LIVE)
    if df.empty:
        st.info(f"No data yet. Start logger.py on the Pi (looking for {LIVE}).")
        return

    last = df.iloc[-1]
    age = time.time() - last["t"]
    if age > STALE_S:
        st.warning(f"No new data for {age:,.0f} s. Is logger.py running?")

    speed = last.get("speed_hz")
    has_speed = speed is not None and pd.notna(speed)
    c1, c2, c3 = st.columns(3)
    c1.metric("Setting", last["setting"])
    c2.metric("Fan speed", f"{speed:.2f} Hz" if has_speed else "—")
    c3.metric("Vibration", f"{last['rms']:.2f} m/s²")
    st.caption(
        (f"{speed * 60:,.0f} rpm  ·  " if has_speed else "")
        + f"updated {datetime.fromtimestamp(last['t']):%H:%M:%S}  ·  run {last['run']}"
        + ("" if last.get("ok", True) else "  ·  ⚠ data-quality warning (see logger)")
    )

    recent = df[df["t"] > last["t"] - HISTORY_MIN * 60].set_index("time")
    st.subheader(f"Fan speed, last {HISTORY_MIN} min")
    st.line_chart(recent["speed_hz"], height=230, y_label="Hz", x_label="")
    st.subheader("Vibration level")
    st.line_chart(recent["rms"], height=170, y_label="m/s²", x_label="")


live_view()
