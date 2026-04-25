from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from paddleocr import PaddleOCR
import cv2
import numpy as np
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ocr = PaddleOCR(lang='en', use_angle_cls=True)

def read_image(file_bytes):
    img_array = np.frombuffer(file_bytes, np.uint8)
    return cv2.imdecode(img_array, cv2.IMREAD_COLOR)

def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    thresh = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY)[1]
    return thresh

def extract_text(img):
    result = ocr.ocr(img, cls=True)
    text = " ".join([line[1][0] for line in result[0]])
    return text.upper()

def find_number(text, label):
    import re
    pattern = rf"{label}\s*([0-9.,]+)"
    m = re.search(pattern, text)
    if not m:
        return None
    return float(m.group(1).replace(",", "."))

def detect_ticker(text):
    tickers = {
        "KETY": "KTY",
        "KĘTY": "KTY",
        "GRUPA KETY": "KTY",
        "CD PROJEKT": "CDR",
        "ALLEGRO": "ALE",
        "ORLEN": "PKN",
        "ASBIS": "ASB",
        "XTB": "XTB",
        "JSW": "JSW",
        "MBANK": "MBK"
    }
    for k, t in tickers.items():
        if k in text:
            return t
    return "UNKNOWN"

@app.post("/ocr")
async def ocr_endpoint(file: UploadFile = File(...)):
    content = await file.read()
    img = read_image(content)

    clean = preprocess(img)
    text = extract_text(clean)

    ticker = detect_ticker(text)

    O = find_number(text, "O")
    H = find_number(text, "H")
    L = find_number(text, "L")
    C = find_number(text, "C")
    MA20 = find_number(text, "MA 20")
    RSI = find_number(text, "RSI")
    VOL = find_number(text, "WOLUMEN")

    return {
        "ticker": ticker,
        "O": O,
        "H": H,
        "L": L,
        "C": C,
        "MA20": MA20,
        "DEMA9": None,
        "RSI": RSI,
        "VOL": VOL
    }
