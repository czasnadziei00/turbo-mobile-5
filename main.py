import io
import re
from typing import Optional
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from PIL import Image
import numpy as np
from paddleocr import PaddleOCR

# ============================================================
#  OCR (PaddleOCR) — działa na Python 3.12
# ============================================================

ocr = PaddleOCR(
    lang='en',
    use_angle_cls=True
)


def ocr_image(img: Image.Image) -> str:
    np_img = np.array(img)
    result = ocr.ocr(np_img, cls=True)
    out = []
    if result:
        for line in result:
            for box, text in line:
                out.append(text[0])
    return " ".join(out)


def ocr_region(full: Image.Image, box):
    crop = full.crop(box)
    return ocr_image(crop)


# ============================================================
#  MODEL ODPOWIEDZI
# ============================================================

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


# ============================================================
#  REGIONY (PORTRAIT)
# ============================================================

BLOK1 = (3, 291, 3+770, 291+286)
BLOK2 = (695, 116, 695+101, 116+76)
BLOK3 = (8, 1267, 8+297, 1267+76)
BLOK4 = (2, 1778, 2+776, 1778+60)

# BLOK5 — TICKER
BLOK5 = (237, 71, 237+1061, 71+88)


# ============================================================
#  FUNKCJE POMOCNICZE
# ============================================================

def _to_float(x: Optional[str]) -> Optional[float]:
    if not x:
        return None
    x = x.replace(" ", "").replace(",", ".")
    try:
        return float(x)
    except ValueError:
        return None


# ============================================================
#  PARSER XTB
# ============================================================

def parse_xtb_text(text: str) -> OcrResponse:
    t = text.replace("\n", " ")
    t = re.sub(r"\s+", " ", t)

    o = re.search(r"\bO\s*([0-9][0-9\s.,]+)", t)
    h = re.search(r"\bH\s*([0-9][0-9\s.,]+)", t)
    l = re.search(r"\bL\s*([0-9][0-9\s.,]+)", t)
    c = re.search(r"\bC\s*([0-9][0-9\s.,]+)", t)

    ma20 = re.search(r"MA\s*20.*?([0-9][0-9\s.,]+)", t, re.IGNORECASE)
    dema = re.search(r"DEMA\s*9.*?([0-9][0-9\s.,]+)", t, re.IGNORECASE)
    rsi = re.search(r"RSI\s*14\s*([0-9][0-9\s.,]+)", t, re.IGNORECASE)
    vol = re.search(r"Wolumen\s*([0-9][0-9\s.,]*)", t, re.IGNORECASE)

    interval = None
    for iv in ["M1", "M5", "M15", "M30", "H1", "H4", "D1"]:
        if re.search(rf"\b{iv}\b", t):
            interval = iv
            break

    ticker = None
    ticker_map = {
        r"DINO|DINOPL": "DINO",
        r"KĘTY|KETY|GRUPA\s*KĘTY|GRUPA\s*KETY|KETY\.PL|GRUPA\s*K\b": "KETY",
        r"KGHM": "KGHM",
        r"COPPER|MIEDŹ": "COPPER",
        r"GOLD|ZŁOTO": "GOLD",
        r"US500|SP500": "US500",
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


# ============================================================
#  ENDPOINT OCR
# ============================================================

app = FastAPI()

@app.post("/ocr")
async def ocr(file: UploadFile = File(...)) -> OcrResponse:
    content = await file.read()
    print("=== ODEBRANO PLIK ===", file.filename)

    img = Image.open(io.BytesIO(content)).convert("RGB")

    text_main = [
        ocr_region(img, BLOK1),
        ocr_region(img, BLOK2),
        ocr_region(img, BLOK3),
        ocr_region(img, BLOK4),
    ]

    text_ticker = ocr_region(img, BLOK5)

    print("=== BLOK5 RAW ===")
    print(text_ticker)

    full_text = " ".join(text_main + [text_ticker])
    parsed = parse_xtb_text(full_text)
    return parsed
