import io
import re
from typing import Optional
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from PIL import Image

# ---------------------------------------------------------
#  MODEL ODPOWIEDZI
# ---------------------------------------------------------

class OcrResponse(BaseModel):
    ticker: Optional[str]
    interval: Optional[str]
    O: Optional[float]
    H: Optional[float]
    L: Optional[float]
    C: Optional[float]
    MA20: Optional[float]
    DEMA9: Optional[float]
    RSI: Optional[float]
    VOL: Optional[float]


# ---------------------------------------------------------
#  REGIONY (PORTRAIT) — BLOK5 DODANY
# ---------------------------------------------------------

BLOK1 = (72, 291, 72+936, 291+258)
BLOK2 = (1044, 291, 1044+936, 291+258)
BLOK3 = (72, 579, 72+1908, 579+258)
BLOK4 = (72, 867, 72+1908, 867+258)

# BLOK5 — TICKER (Twoje pomiary)
BLOK5 = (237, 71, 237+1061, 71+88)


# ---------------------------------------------------------
#  FUNKCJE POMOCNICZE
# ---------------------------------------------------------

def _to_float(x: Optional[str]) -> Optional[float]:
    if not x:
        return None
    x = x.replace(" ", "").replace(",", ".")
    try:
        return float(x)
    except ValueError:
        return None


# ---------------------------------------------------------
#  OCR — PODSTAW SWÓJ MODEL (HF / Paddle / EasyOCR)
# ---------------------------------------------------------

def ocr_image(img: Image.Image) -> str:
    # TU PODSTAWIASZ SWÓJ MODEL OCR
    # return hf_ocr(img_bytes)
    return ""   # placeholder — Twój OCR tu wchodzi


def ocr_region(full: Image.Image, box):
    crop = full.crop(box)
    return ocr_image(crop)


# ---------------------------------------------------------
#  PARSER XTB — STABILNY, ODPORNY NA OCR
# ---------------------------------------------------------

def parse_xtb_text(text: str) -> OcrResponse:
    t = text.replace("\n", " ")
    t = re.sub(r"\s+", " ", t)

    # O / H / L / C
    o = re.search(r"\bO\s*([0-9][0-9\s.,]+)", t)
    h = re.search(r"\bH\s*([0-9][0-9\s.,]+)", t)
    l = re.search(r"\bL\s*([0-9][0-9\s.,]+)", t)
    c = re.search(r"\bC\s*([0-9][0-9\s.,]+)", t)

    # MA20 / DEMA9
    ma20 = re.search(r"MA\s*20.*?([0-9][0-9\s.,]+)", t, re.IGNORECASE)
    dema = re.search(r"DEMA\s*9.*?([0-9][0-9\s.,]+)", t, re.IGNORECASE)

    # RSI
    rsi = re.search(r"RSI\s*14\s*([0-9][0-9\s.,]+)", t, re.IGNORECASE)

    # Wolumen
    vol = re.search(r"Wolumen\s*([0-9][0-9\s.,]*)", t, re.IGNORECASE)

    # Interwał
    interval = None
    for iv in ["M1", "M5", "M15", "M30", "H1", "H4", "D1"]:
        if re.search(rf"\b{iv}\b", t):
            interval = iv
            break

    # Ticker — odporne na OCR
    ticker = None
    ticker_map = {
        r"\bDINO\b|DINOPL": "DINO",
        r"KĘTY|KETY|GRUPA\s*KĘTY|GRUPA\s*KETY|KETY\.PL|GRUPA\s*K\b": "KETY",
        r"\bKGHM\b": "KGHM",
        r"\bCOPPER\b|MIEDŹ": "COPPER",
        r"\bGOLD\b|ZŁOTO": "GOLD",
        r"\bUS500\b|SP500": "US500",
    }

    for pattern, code in ticker_map.items():
        if re.search(pattern, t, re.IGNORECASE):
            ticker = code
            break

    return OcrResponse(
        ticker=ticker,
        interval=interval,
        O=_to_float(o.group(1)) if o else None,
        H=_to_float(h.group(1)) if h else None,
        L=_to_float(l.group(1)) if l else None,
        C=_to_float(c.group(1)) if c else None,
        MA20=_to_float(ma20.group(1)) if ma20 else None,
        DEMA9=_to_float(dema.group(1)) if dema else None,
        RSI=_to_float(rsi.group(1)) if rsi else None,
        VOL=_to_float(vol.group(1)) if vol else None,
    )


# ---------------------------------------------------------
#  ENDPOINT OCR
# ---------------------------------------------------------

app = FastAPI()

@app.post("/ocr")
async def ocr(file: UploadFile = File(...)) -> OcrResponse:
    content = await file.read()
    img = Image.open(io.BytesIO(content)).convert("RGB")

    # BLOK1–4: dane liczbowe
    text_main = [
        ocr_region(img, BLOK1),
        ocr_region(img, BLOK2),
        ocr_region(img, BLOK3),
        ocr_region(img, BLOK4),
    ]

    # BLOK5: ticker
    text_ticker = ocr_region(img, BLOK5)

    full_text = " ".join(text_main + [text_ticker])

    print("=== OCR TEXT ===")
    print(full_text)

    parsed = parse_xtb_text(full_text)
    return parsed
