"""Lightweight live fan dashboard - plain HTTP polling, no WebSockets, no extra installs.

Works through phone security filters (e.g. Norton Safe Web) that break Streamlit's
live connection. On the Pi, with logger.py running in another terminal:
    python dashboard_lite.py
Then on any device on the home Wi-Fi:  http://10.0.0.250:8502

The page asks the Pi for the latest numbers every 2 seconds and redraws in place.
Reads ../data/live.jsonl (written by logger.py) and never writes anything.
"""
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

try:
    HERE = Path(__file__).resolve().parent
except NameError:
    HERE = Path.cwd()
LIVE = HERE.parent / "data" / "live.jsonl"
PORT = 8502
HISTORY_MIN = 10


def read_tail(path, max_bytes=500_000):
    if not path.exists():
        return []
    with open(path, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - max_bytes))
        lines = f.read().decode("utf-8", errors="ignore").splitlines()
    if size > max_bytes:
        lines = lines[1:]                      # first line may be cut in half
    rows = []
    for ln in lines:
        try:
            rows.append(json.loads(ln))
        except json.JSONDecodeError:
            pass
    return rows


def snapshot():
    rows = read_tail(LIVE)
    if not rows:
        return {"now": time.time(), "last": None, "t": [], "speed": [], "rms": []}
    t_end = rows[-1]["t"]
    rows = [r for r in rows if r["t"] > t_end - HISTORY_MIN * 60]
    return {"now": time.time(), "last": rows[-1],
            "t": [r["t"] for r in rows],
            "speed": [r.get("speed_hz") for r in rows],
            "rms": [r.get("rms") for r in rows]}


PAGE = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Fan Monitor</title>
<style>
:root{--bg:#fbfaf7;--card:#ffffff;--ink:#22221f;--muted:#6b6b66;--grid:#e6e4de;--line:#2a78d6;--line2:#eb6834;--warn-bg:#fff4d6;--warn:#8a5a00;}
@media (prefers-color-scheme: dark){:root{--bg:#171715;--card:#22221f;--ink:#f1efe9;--muted:#a4a29b;--grid:#3a3934;--line:#3987e5;--line2:#d95926;--warn-bg:#3a3020;--warn:#f0c46a;}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.4 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
main{max-width:720px;margin:0 auto;padding:16px}
h1{font-size:22px;margin:4px 0 12px}
.tiles{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}
.tile{background:var(--card);border-radius:12px;padding:12px;box-shadow:0 1px 2px rgba(0,0,0,.06)}
.lbl{font-size:13px;color:var(--muted)}
.val{font-size:28px;font-weight:650;margin-top:2px;font-variant-numeric:tabular-nums}
.sub{font-size:13px;color:var(--muted);font-variant-numeric:tabular-nums}
.meta{font-size:13px;color:var(--muted);margin:10px 2px}
.warn{display:none;background:var(--warn-bg);color:var(--warn);border-radius:10px;padding:10px 12px;margin-bottom:10px;font-size:14px}
.chart{background:var(--card);border-radius:12px;padding:10px 10px 4px;margin-top:12px;box-shadow:0 1px 2px rgba(0,0,0,.06)}
.chart h2{font-size:15px;margin:2px 4px 6px}
svg{width:100%;height:180px;display:block}
@media (max-width:420px){.val{font-size:22px}}
</style></head><body><main>
<h1>Fan monitor</h1>
<div id="warn" class="warn"></div>
<div class="tiles">
 <div class="tile"><div class="lbl">Setting</div><div class="val" id="setting">—</div></div>
 <div class="tile"><div class="lbl">Fan speed</div><div class="val" id="speed">—</div><div class="sub" id="rpm"></div></div>
 <div class="tile"><div class="lbl">Vibration</div><div class="val" id="rms">—</div><div class="sub">m/s²</div></div>
</div>
<div class="meta" id="meta">connecting…</div>
<div class="chart"><h2>Fan speed (Hz), last 10 min</h2><svg id="c1" ></svg></div>
<div class="chart"><h2>Vibration level (m/s²)</h2><svg id="c2" ></svg></div>
</main>
<script>
const $ = id => document.getElementById(id);
function fmtTime(t){return new Date(t*1000).toLocaleTimeString([], {hour:'2-digit',minute:'2-digit',second:'2-digit'});}
function draw(svg, t, y, color){
  const W=Math.max(svg.clientWidth,300),H=180,L=34,R=6,T=8,B=20;
  svg.setAttribute('viewBox',`0 0 ${W} ${H}`);
  const css=getComputedStyle(document.documentElement);
  const grid=css.getPropertyValue('--grid'), muted=css.getPropertyValue('--muted');
  const pts=t.map((x,i)=>[x,y[i]]).filter(p=>p[1]!==null && p[1]!==undefined);
  let s='';
  if(pts.length<2){svg.innerHTML=`<text x="300" y="95" text-anchor="middle" fill="${muted}" font-size="14" x="${W/2}">waiting for data…</text>`;return;}
  const t0=t[0], t1=t[t.length-1]||t0+1, ymax=Math.max(...pts.map(p=>p[1]))*1.1||1;
  const X=x=>L+(x-t0)/Math.max(t1-t0,1)*(W-L-R), Y=v=>H-B-(v/ymax)*(H-B-T);
  for(let k=0;k<=4;k++){const v=ymax*k/4, yy=Y(v);
    s+=`<line x1="${L}" x2="${W-R}" y1="${yy}" y2="${yy}" stroke="${grid}" stroke-width="1"/>`;
    s+=`<text x="${L-4}" y="${yy+4}" text-anchor="end" fill="${muted}" font-size="11">${v<10?v.toFixed(1):v.toFixed(0)}</text>`;}
  s+=`<text x="${L}" y="${H-4}" fill="${muted}" font-size="11">${fmtTime(t0)}</text>`;
  s+=`<text x="${W-R}" y="${H-4}" text-anchor="end" fill="${muted}" font-size="11">${fmtTime(t1)}</text>`;
  let d='', pen=false;                      // break the line where data is missing (fan off)
  t.forEach((x,i)=>{const v=y[i]; if(v===null||v===undefined){pen=false;return;}
    d+=(pen?'L':'M')+X(x).toFixed(1)+','+Y(v).toFixed(1); pen=true;});
  s+=`<path d="${d}" fill="none" stroke="${color}" stroke-width="2" stroke-linejoin="round"/>`;
  svg.innerHTML=s;
}
async function tick(){
  try{
    const r=await fetch('/api/live',{cache:'no-store'}); const d=await r.json();
    const css=getComputedStyle(document.documentElement);
    if(!d.last){$('warn').style.display='block';$('warn').textContent='No data yet. Start logger.py on the Pi.';return;}
    const L=d.last, age=d.now-L.t;
    $('setting').textContent=L.setting;
    $('speed').textContent=(L.speed_hz!=null)?L.speed_hz.toFixed(2)+' Hz':'—';
    $('rpm').textContent=(L.speed_hz!=null)?Math.round(L.speed_hz*60).toLocaleString()+' rpm':'';
    $('rms').textContent=L.rms.toFixed(2);
    $('meta').textContent=`updated ${fmtTime(L.t)} · run ${L.run}`+(L.ok===false?' · ⚠ data-quality warning':'');
    const w=$('warn'); if(age>5){w.style.display='block';w.textContent=`No new data for ${Math.round(age)} s. Is logger.py running?`;} else w.style.display='none';
    draw($('c1'), d.t, d.speed, css.getPropertyValue('--line'));
    draw($('c2'), d.t, d.rms, css.getPropertyValue('--line2'));
  }catch(e){$('warn').style.display='block';$('warn').textContent='Cannot reach the Pi. Retrying…';}
}
tick(); setInterval(tick, 2000);
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/api/live"):
            body, ctype = json.dumps(snapshot()).encode(), "application/json"
        elif self.path in ("/", "/index.html"):
            body, ctype = PAGE.encode(), "text/html; charset=utf-8"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):          # keep the terminal quiet
        pass


if __name__ == "__main__":
    print(f"Fan dashboard on http://0.0.0.0:{PORT}  (phone: http://10.0.0.250:{PORT})  Ctrl+C to stop")
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
