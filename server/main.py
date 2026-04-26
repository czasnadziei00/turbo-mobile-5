from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from paddleocr import PaddleOCR
import cv2
import numpy as np
import re

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ocr = PaddleOCR(lang='en', use_angle_cls=True)

# -------------------------------
#  CROP TICKER AREA (XTB TOP BAR)
# -------------------------------
def crop_ticker_area(img):
    h, w, _ = img.shape
    return img[0:int(h*0.15), 0:int(w*0.7)]

# -------------------------------
#  BASIC PREPROCESSING
# -------------------------------
def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    thresh = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY)[1]
    return thresh

# -------------------------------
#  OCR TEXT EXTRACT
# -------------------------------
def extract_text(img):
    result = ocr.ocr(img, cls=True)
    if not result or not result[0]:
        return ""
    return " ".join([line[1][0] for line in result[0]]).upper()

# -------------------------------
#  DETECT INTERVAL (M5/M15/H1)
# -------------------------------
def detect_interval(text):
    intervals = ["M1","M5","M15","M30","H1","H4","D1","W1","MN"]
    for i in intervals:
        if i in text:
            return i
    return "UNKNOWN"

# -------------------------------
#  DETECT TICKER
# -------------------------------
def detect_ticker(text):
    tickers = {
        "GRUPA KETY": "KTY",
        "GRUPA KĘTY": "KTY",
        "KETY": "KTY",
        "KĘTY": "KTY",
        "CD PROJEKT": "CDR",
        "ALLEGRO": "ALE",
        "ORLEN": "PKN",
        "PKN ORLEN": "PKN",
        "ASBIS": "ASB",
        "XTB": "XTB",
        "JSW": "JSW",
        "MBANK": "MBK",
        "KGHM": "KGH"
    }
    for k, t in tickers.items():
        if k in text:
            return t
    return "UNKNOWN"

# -------------------------------
#  NUMBER PARSER
# -------------------------------
def find_number(text, label):
    m = re.search(rf"{label}\s*([0-9.,]+)", text)
    if not m:
        return None
    return float(m.group(1).replace(",", "."))

# -------------------------------
#  MAIN ENDPOINT
# -------------------------------
@app.post("/ocr")
async def ocr_endpoint(file: UploadFile = File(...)):
    content = await file.read()

    img = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
    clean = preprocess(img)
    text = extract_text(clean)

    # ticker detection
    ticker_img = crop_ticker_area(img)
    ticker_text = extract_text(preprocess(ticker_img))
    ticker = detect_ticker(ticker_text)

    if ticker == "UNKNOWN":
        ticker = detect_ticker(text)

    interval = detect_interval(text)

    O    = find_number(text, "O")
    H    = find_number(text, "H")
    L    = find_number(text, "L")
    C    = find_number(text, "C")
    MA20 = find_number(text, "MA 20")
    RSI  = find_number(text, "RSI")
    VOL  = find_number(text, "WOLUMEN")

    return {
        "ticker": ticker,
        "interval": interval,
        "O": O,
        "H": H,
        "L": L,
        "C": C,
        "MA20": MA20,
        "DEMA9": None,
        "RSI": RSI,
        "VOL": VOL
    }
