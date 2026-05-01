import os
import re
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
  """
  Prosty wrapper na HF OCR:
  - timeout
  - obsługa różnych formatów odpowiedzi
  - czytelne logowanie
  """
  headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
  try:
    resp = requests.post(
      HF_MODEL_URL,
      headers=headers,
      files={"file": image_bytes},
      timeout=60,
    )
    resp.raise_for_status()
  except Exception as e:
    print("=== HF OCR ERROR ===")
    print(repr(e))
    raise

  data = resp.json()

  print("=== RAW HF RESPONSE ===")
  print(data)

  # typowe formaty z HF
  if isinstance(data, list) and data and isinstance(data[0], dict):
    if "generated_text" in data[0]:
      return data[0]["generated_text"]

  if isinstance(data, dict):
    if "text" in data:
      return data["text"]
    if "generated_text" in data:
      return data["generated_text"]

  # fallback – cokolwiek, byle string
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
  o = re.search(r"\bO\s*([0-9]+[.,][0-9]+)", t)
  h = re.search(r"\bH\s*([0-9]+[.,][0-9]+)", t)
  l = re.search(r"\bL\s*([0-9]+[.,][0-9]+)", t)
  c = re.search(r"\bC\s*([0-9]+[.,][0-9]+)", t)

  # MA20
  ma20 = re.search(r"\bMA\s*20\b.*?([0-9]+[.,][0-9]+)", t)

  # DEMA9 – pierwsza liczba po "9–20" albo "9-20"
  dema = re.search(r"9[–-]20.*?([0-9]+[.,][0-9]+)", t)

  # RSI
  rsi = re.search(r"\bRSI\s*14\s*([0-9]+[.,][0-9]+)", t)

  # Wolumen
  vol = re.search(r"\bWolumen\s*([0-9]+)", t, re.IGNORECASE)

  # Interval
  interval = None
  for iv in ["M5", "M15", "H1", "D1"]:
    if re.search(rf"\b{iv}\b", t):
      interval = iv
      break

  # Ticker – mapowanie w jednym miejscu
  ticker = None
  ticker_map = {
    r"\bDino\b": "DINO",
    r"Kęty|KETY|Grupa Kęty": "KTY",
    # tu dopisujesz kolejne:
    # r"Orlen|PKN Orlen": "PKN",
    # r"Allegro": "ALE",
  }
  for pattern, code in ticker_map.items():
    if re.search(pattern, t, re.IGNORECASE):
      ticker = code
      break

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
  i zwraca JSON zgodny z TURBO MOBILE 5 PRO.
  """
  try:
    content = await file.read()
    text = hf_ocr(content)
    parsed = parse_xtb_text(text)
    return parsed
  except Exception as e:
    print("=== OCR_XTB ERROR ===")
    print(repr(e))
    # frontend ma fallback na aktywny wiersz
    return OcrResponse()
