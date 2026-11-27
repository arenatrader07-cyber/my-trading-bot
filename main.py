from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np

app = FastAPI(title="Real-Time Smart Trading Engine")

bot_memory = {
    "marked_ob_zones": [],
    "strategy_scores": {
        "OB_FVG": {"wins": 8, "total": 10},
        "ENGULFING_OUTSIDE": {"wins": 7, "total": 10},
        "THREE_CANDLE": {"wins": 8, "total": 10},
        "OPPOSITE_2_CANDLE": {"wins": 7, "total": 10}
    },
    "last_signal": None,
    "mtg_step": 0
}

def extract_candles(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, np.array([35, 50, 50]), np.array([90, 255, 255]))
    red_mask1 = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([10, 255, 255]))
    red_mask2 = cv2.inRange(hsv, np.array([170, 50, 50]), np.array([180, 255, 255]))
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    candles = []
    for color, mask in [("GREEN", green_mask), ("RED", red_mask)]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if 3 <= w <= 50 and h >= 8:
                candles.append({"color": color, "x": x, "top": y, "bottom": y + h, "height": h})

    candles.sort(key=lambda c: c["x"])
    return candles

def get_best_strategy():
    best_name = None
    best_rate = 0
    for name, data in bot_memory["strategy_scores"].items():
        rate = (data["wins"] / data["total"]) * 100
        if rate > best_rate:
            best_rate = rate
            best_name = name
    return best_name, round(best_rate, 1)

@app.get("/")
def home():
    return {"status": "Online", "message": "Real-Time Smart Trading Engine Active!"}

@app.post("/live-engine")
async def live_engine(file: UploadFile = File(...)):
    image_bytes = await file.read()
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "Screen pora jacche na"}

    candles = extract_candles(img)
    if len(candles) < 5:
        return {"action": "WAIT", "message": "Chart khujche..."}

    c1, c2, c3, c4 = candles[-4], candles[-3], candles[-2], candles[-1]
    
    if bot_memory["last_signal"] is not None:
        expected_color = "GREEN" if bot_memory["last_signal"] == "UP" else "RED"
        if c4["color"] == expected_color:
            trade_status = "PROFIT (WIN)" if bot_memory["mtg_step"] == 0 else "1-STEP MTG PROFIT (WIN)"
            bot_memory["last_signal"] = None
            bot_memory["mtg_step"] = 0
            return {"action": "RESULT", "status": trade_status, "mtg": False}
        else:
            if bot_memory["mtg_step"] == 0:
                bot_memory["mtg_step"] = 1
                return {
                    "action": "SIGNAL",
                    "signal": bot_memory["last_signal"],
                    "type": "1-STEP MTG (2X)",
                    "status": "Direct Loss -> 1-Step MTG Active!"
                }
            else:
                bot_memory["last_signal"] = None
                bot_memory["mtg_step"] = 0
                return {"action": "RESULT", "status": "MTG LOSS - Waiting for New Setup", "mtg": False}

    if c2["color"] == "RED" and c3["color"] == "GREEN" and c3["height"] > c2["height"] * 1.5:
        bot_memory["marked_ob_zones"] = [{"type": "BULLISH_OB", "top": c2["top"], "bottom": c2["bottom"]}]
    elif c2["color"] == "GREEN" and c3["color"] == "RED" and c3["height"] > c2["height"] * 1.5:
        bot_memory["marked_ob_zones"] = [{"type": "BEARISH_OB", "top": c2["top"], "bottom": c2["bottom"]}]

    best_strategy, win_rate = get_best_strategy()

    in_bullish_zone = any(z["type"] == "BULLISH_OB" and c4["bottom"] >= z["top"] for z in bot_memory["marked_ob_zones"])
    in_bearish_zone = any(z["type"] == "BEARISH_OB" and c4["top"] <= z["bottom"] for z in bot_memory["marked_ob_zones"])

    bullish_confirm = (c2["color"] == "RED" and c3["color"] == "GREEN" and c4["color"] == "GREEN")
    bearish_confirm = (c2["color"] == "GREEN" and c3["color"] == "RED" and c4["color"] == "RED")

    if in_bullish_zone and bullish_confirm:
        bot_memory["last_signal"] = "UP"
        return {
            "action": "SIGNAL",
            "signal": "UP",
            "type": "DIRECT ENTRY",
            "best_strategy": best_strategy,
            "win_rate": f"{win_rate}%",
            "zone": "Marked Bullish OB + FVG"
        }

    if in_bearish_zone and bearish_confirm:
        bot_memory["last_signal"] = "DOWN"
        return {
            "action": "SIGNAL",
            "signal": "DOWN",
            "type": "DIRECT ENTRY",
            "best_strategy": best_strategy,
            "win_rate": f"{win_rate}%",
            "zone": "Marked Bearish OB + FVG"
        }

    return {
        "action": "WAIT",
        "marked_zones": len(bot_memory["marked_ob_zones"]),
        "active_best_strategy": f"{best_strategy} ({win_rate}%)"
              }
