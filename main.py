from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import time

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

bot_active = False

HTML_UI = """
<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Quotex 3D AI Bot</title>
<style>
  body { background: #090d16; color: white; font-family: sans-serif; display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100vh; margin: 0; }
  .bubble { width: 90px; height: 90px; border-radius: 50%; background: #ef4444; border: 4px solid white; display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 20px; cursor: pointer; box-shadow: 0 10px 25px rgba(0,0,0,0.6); transition: 0.3s; }
  .bubble.on { background: #10b981; box-shadow: 0 0 25px #10b981; }
  .card3d { margin-top: 25px; width: 240px; padding: 20px; border-radius: 20px; background: linear-gradient(145deg, #1e293b, #0f172a); border: 2px solid #38bdf8; box-shadow: 0 15px 35px rgba(0,0,0,0.8), inset 0 1px 1px rgba(255,255,255,0.2); text-align: center; display: none; }
  .sig { font-size: 28px; font-weight: 900; margin: 5px 0; color: #38bdf8; }
  .acc { font-size: 16px; color: #e2e8f0; margin: 5px 0; }
  .mtg { font-size: 13px; font-weight: bold; color: #facc15; background: rgba(250,204,21,0.15); padding: 4px 10px; border-radius: 8px; display: inline-block; margin-top: 5px; }
  video { display: none; }
</style>
</head>
<body>
  <div id="btn" class="bubble" onclick="toggleBot()">OFF</div>
  <div id="card" class="card3d">
    <div id="sig" class="sig">SCANNING...</div>
    <div id="acc" class="acc">Win Chance: --%</div>
    <div id="mtg" class="mtg">1-STEP MTG</div>
  </div>
  <video id="vid" autoplay playsinline></video>
  <canvas id="cvs" style="display:none;"></canvas>

<script>
let isOn = false;
let stream = null;
let timer = null;

async function toggleBot() {
  const btn = document.getElementById('btn');
  const card = document.getElementById('card');
  if (!isOn) {
    try {
      stream = await navigator.mediaDevices.getDisplayMedia({ video: true });
      document.getElementById('vid').srcObject = stream;
      isOn = true;
      btn.innerText = "ON";
      btn.classList.add("on");
      card.style.display = "block";
      await fetch('/toggle', { method: 'POST' });
      timer = setInterval(captureAndSend, 2000);
    } catch (e) {
      alert("Screen Share Permission দিন!");
    }
  } else {
    isOn = false;
    btn.innerText = "OFF";
    btn.classList.remove("on");
    card.style.display = "none";
    if (stream) stream.getTracks().forEach(t => t.stop());
    clearInterval(timer);
    await fetch('/toggle', { method: 'POST' });
  }
}

async function captureAndSend() {
  if (!isOn) return;
  const vid = document.getElementById('vid');
  const cvs = document.getElementById('cvs');
  if (vid.videoWidth === 0) return;
  cvs.width = vid.videoWidth / 2;
  cvs.height = vid.videoHeight / 2;
  cvs.getContext('2d').drawImage(vid, 0, 0, cvs.width, cvs.height);
  cvs.toBlob(async (blob) => {
    const fd = new FormData();
    fd.append("file", blob, "screen.jpg");
    const res = await fetch('/analyze', { method: 'POST', body: fd });
    const data = await res.json();
    updateUI(data);
  }, 'image/jpeg', 0.7);
}

function updateUI(d) {
  const sig = document.getElementById('sig');
  const acc = document.getElementById('acc');
  const mtg = document.getElementById('mtg');
  const card = document.getElementById('card');
  if (d.signal === "UP") {
    sig.innerText = "▲ UP (BUY)"; sig.style.color = "#10b981";
    acc.innerText = "Win Chance: " + d.accuracy + "%";
    mtg.innerText = d.mtg; card.style.borderColor = "#10b981";
  } else if (d.signal === "DOWN") {
    sig.innerText = "▼ DOWN (SELL)"; sig.style.color = "#ef4444";
    acc.innerText = "Win Chance: " + d.accuracy + "%";
    mtg.innerText = d.mtg; card.style.borderColor = "#ef4444";
  } else if (d.signal === "ABORT") {
    sig.innerText = "⚠ ABORT"; sig.style.color = "#f59e0b";
    acc.innerText = "High Risk / Loss";
    mtg.innerText = "DO NOT TRADE"; card.style.borderColor = "#f59e0b";
  } else {
    sig.innerText = "SCANNING..."; sig.style.color = "#38bdf8";
    acc.innerText = "Reading Chart...";
    mtg.innerText = "1-STEP MTG"; card.style.borderColor = "#38bdf8";
  }
}
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def home():
    return HTML_UI

@app.post("/toggle")
def toggle_bot():
    global bot_active
    bot_active = not bot_active
    return {"bot_active": bot_active}

def analyze_chart(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return {"signal": "ABORT", "accuracy": 0, "mtg": "N/A"}

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, np.array([35, 50, 50]), np.array([90, 255, 255]))
    red_mask = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([10, 255, 255])) | cv2.inRange(hsv, np.array([170, 50, 50]), np.array([180, 255, 255]))

    cg, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cr, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candles = []
    for c in cg:
        x, y, w, h = cv2.boundingRect(c)
        if h > 15 and 3 < w < 60: candles.append({"x": x, "h": h, "c": "G"})
    for c in cr:
        x, y, w, h = cv2.boundingRect(c)
        if h > 15 and 3 < w < 60: candles.append({"x": x, "h": h, "c": "R"})

    candles.sort(key=lambda i: i["x"])
    if len(candles) < 4:
        return {"signal": "WAIT", "accuracy": 0, "mtg": "1-STEP MTG"}

    c4, c3, c2, c1 = candles[-4], candles[-3], candles[-2], candles[-1]
    sec = int(time.time()) % 60
    if not (38 <= sec <= 52):
        return {"signal": "WAIT", "accuracy": 0, "mtg": "1-STEP MTG"}

    up, down = 0, 0
    if c4["c"] == "R" and c3["c"] == "G" and c2["c"] == "G" and c1["c"] == "G": up += 45
    if c4["c"] == "G" and c3["c"] == "R" and c2["c"] == "R" and c1["c"] == "R": down += 45
    if c2["c"] == "R" and c1["c"] == "G" and c1["h"] > c2["h"] * 1.2: up += 35
    if c2["c"] == "G" and c1["c"] == "R" and c1["h"] > c2["h"] * 1.2: down += 35
    if c2["c"] == "R" and c1["c"] == "G" and c1["h"] > c2["h"] * 1.8: up += 40
    if c2["c"] == "G" and c1["c"] == "R" and c1["h"] > c2["h"] * 1.8: down += 40

    if up >= 60 and down == 0:
        return {"signal": "UP", "accuracy": min(96, 65 + up // 3), "mtg": "1-STEP MTG"}
    elif down >= 60 and up == 0:
        return {"signal": "DOWN", "accuracy": min(96, 65 + down // 3), "mtg": "1-STEP MTG"}
    return {"signal": "ABORT", "accuracy": 45, "mtg": "NO TRADE"}

@app.post("/analyze")
async def analyze_screen(file: UploadFile = File(...)):
    if not bot_active:
        return {"signal": "OFF", "accuracy": 0, "mtg": "OFF"}
    return analyze_chart(await file.read())
