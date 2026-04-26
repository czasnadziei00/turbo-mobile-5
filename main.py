import os
import re
import io
import base64
from typing import Optional

import requests
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_MODEL_URL = os.getenv(
    "HF_MODEL_URL",
    "https://api-inference.huggingface.co/models/your-ocr-model-id"
)

app = FastAPI(title="TURBO MOBILE AI Vision 2.0 (XTB)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class OcrResponse(BaseModel):
    ticker: Optional[str] = None
    interval: Optional[str] = None
    O: Optional[float] = None
    H: Optional[float] = None
    L: Optional[float] = None
    C: Optional[float] = None
    MA20: Optional[float] = None
    DEMA9: Optional[float] = None
    RSI: Optional[float] = None
    VOL: Optional[float] = None


@app.get("/health")
def health():
    return {"status": "ok"}


def hf_ocr(image_bytes: bytes) -> str:
    headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
    resp = requests.post(
        HF_MODEL_URL,
        headers=headers,
        files={"file": image_bytes},
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()

    print("=== RAW HF RESPONSE ===")
    print(data)

    if isinstance(data, list) and data and "generated_text" in data[0]:
        return data[0]["generated_text"]
    if isinstance(data, dict) and "text" in data:
        return data["text"]

    return str(data)




def _to_float(x: Optional[str]) -> Optional[float]:
    if not x:
        return None
    x = x.replace(" ", "").replace(",", ".")
    try:
        return float(x)
    except ValueError:
        return None


def parse_xtb_text(text: str) -> OcrResponse:
    """
    Parser dopasowany do layoutu XTB.
    Szukamy:
      O33.490 H33.540 L33.490 C33.530
      MA 20 close 33.570
      Podwójna Średnia Krocząca 9–20 Exponential 33.542 33.567
      RSI 14 44.80
      Wolumen 138
      Dino Polska / Grupa Kęty / itp.
      M5 / M15 / H1 / D1
    """
    t = text.replace("\n", " ")
    t = re.sub(r"\s+", " ", t)

    # O/H/L/C
    o = re.search(r"O\s*([0-9]+[.,][0-9]+)", t)
    h = re.search(r"H\s*([0-9]+[.,][0-9]+)", t)
    l = re.search(r"L\s*([0-9]+[.,][0-9]+)", t)
    c = re.search(r"C\s*([0-9]+[.,][0-9]+)", t)

    # MA20
    ma20 = re.search(r"MA\s*20.*?([0-9]+[.,][0-9]+)", t)

    # DEMA9 – bierzemy pierwszą liczbę po "9–20" albo "9-20"
    dema = re.search(r"9[–-]20.*?([0-9]+[.,][0-9]+)", t)

    # RSI
    rsi = re.search(r"RSI\s*14\s*([0-9]+[.,][0-9]+)", t)

    # Wolumen
    vol = re.search(r"Wolumen\s*([0-9]+)", t, re.IGNORECASE)

    # Interval
    interval = None
    for iv in ["M5", "M15", "H1", "D1"]:
        if re.search(rf"\b{iv}\b", t):
            interval = iv
            break

    # Ticker – brutalnie, ale skutecznie: szukamy znanych słów i mapujemy
    ticker = None
    if re.search(r"\bDino\b", t, re.IGNORECASE):
        ticker = "DINO"
    elif re.search(r"Kęty|KETY|Grupa Kęty", t, re.IGNORECASE):
        ticker = "KTY"
    # tu możesz dodać kolejne mapowania GPW/USA

    return OcrResponse(
        ticker=ticker,
        interval=interval,
        O=_to_float(o.group(1) if o else None),
        H=_to_float(h.group(1) if h else None),
        L=_to_float(l.group(1) if l else None),
        C=_to_float(c.group(1) if c else None),
        MA20=_to_float(ma20.group(1) if ma20 else None),
        DEMA9=_to_float(dema.group(1) if dema else None),
        RSI=_to_float(rsi.group(1) if rsi else None),
        VOL=_to_float(vol.group(1) if vol else None),
    )


@app.post("/ocr", response_model=OcrResponse)
async def ocr_xtb(file: UploadFile = File(...)):
    """
    AI Vision 2.0 — XTB‑only.
    Przyjmuje screena z XTB, wysyła do HF OCR, parsuje layout XTB
    i zwraca JSON zgodny z Twoim TURBO MOBILE.
    """
    try:
        content = await file.read()
        text = hf_ocr(content)
        parsed = parse_xtb_text(text)
        return parsed
    except Exception as e:
        # frontend i tak ma fallback na aktywny wiersz
        return OcrResponse()
