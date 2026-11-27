from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import time

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=list(("*",)), allow_methods=list(("*",)), allow_headers=list(("*",)))

bot_active = False

@app.get("/", response_class=HTMLResponse)
def home():
    return "<h1>Quotex 3D AI Server is LIVE</h1>"

@app.post("/toggle")
def toggle_bot():
    global bot_active
    bot_active = not bot_active
    return dict(bot_active=bot_active)

def analyze_chart(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return dict(signal="ABORT", accuracy=0, mtg="N/A")

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    g1 = np.array(tuple((35, 50, 50)), dtype=np.uint8)
    g2 = np.array(tuple((90, 255, 255)), dtype=np.uint8)
    r1 = np.array(tuple((0, 70, 50)), dtype=np.uint8)
    r2 = np.array(tuple((10, 255, 255)), dtype=np.uint8)
    r3 = np.array(tuple((170, 70, 50)), dtype=np.uint8)
    r4 = np.array(tuple((180, 255, 255)), dtype=np.uint8)

    green_mask = cv2.inRange(hsv, g1, g2)
    red_mask = cv2.inRange(hsv, r1, r2) | cv2.inRange(hsv, r3, r4)

    cg, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cr, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candles = list()
    for c in cg:
        x, y, w, h = cv2.boundingRect(c)
        if h > 15 and 3 < w < 60:
            candles.append(dict(x=x, h=h, c="G"))
    for c in cr:
        x, y, w, h = cv2.boundingRect(c)
        if h > 15 and 3 < w < 60:
            candles.append(dict(x=x, h=h, c="R"))

    candles.sort(key=lambda i: i.get("x"))
    if len(candles) < 4:
        return dict(signal="WAIT", accuracy=0, mtg="1-STEP MTG")

    c4, c3, c2, c1 = candles[-4], candles[-3], candles[-2], candles[-1]
    sec = int(time.time()) % 60
    if not (38 <= sec <= 52):
        return dict(signal="WAIT", accuracy=0, mtg="1-STEP MTG")

    up, down = 0, 0
    if c4.get("c") == "R" and c3.get("c") == "G" and c2.get("c") == "G" and c1.get("c") == "G":
        up += 45
    if c4.get("c") == "G" and c3.get("c") == "R" and c2.get("c") == "R" and c1.get("c") == "R":
        down += 45
    if c2.get("c") == "R" and c1.get("c") == "G" and c1.get("h") > c2.get("h") * 1.2:
        up += 35
    if c2.get("c") == "G" and c1.get("c") == "R" and c1.get("h") > c2.get("h") * 1.2:
        down += 35
    if c2.get("c") == "R" and c1.get("c") == "G" and c1.get("h") > c2.get("h") * 1.8:
        up += 40
    if c2.get("c") == "G" and c1.get("c") == "R" and c1.get("h") > c2.get("h") * 1.8:
        down += 40

    if up >= 60 and down == 0:
        return dict(signal="UP", accuracy=min(96, 65 + up // 3), mtg="1-STEP MTG")
    elif down >= 60 and up == 0:
        return dict(signal="DOWN", accuracy=min(96, 65 + down // 3), mtg="1-STEP MTG")
    return dict(signal="ABORT", accuracy=45, mtg="NO TRADE")

@app.post("/analyze")
async def analyze_screen(file: UploadFile = File(...)):
    if not bot_active:
        return dict(signal="OFF", accuracy=0, mtg="OFF")
    return analyze_chart(await file.read())
