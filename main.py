from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import requests
import os
import re
import time

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
#  CONFIG — HUGGINGFACE DONUT
# ============================================================

HF_TOKEN = os.getenv("HF_TOKEN")

DONUT_MODEL = "naver-clova-ix/donut-base-finetuned-docvqa"
HF_URL = f"https://api-inference.huggingface.co/models/{DONUT_MODEL}"

HF_HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}"
}


# ============================================================
#  HEALTHCHECK
#  (zostawiamy Twój branding tesseract-ultra-pro + feature list)
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "OK",
        "engine": "tesseract-ultra-pro",
        "version": "2.8",
        "features": [
            "AI Ticker Correct",
            "AI Number Correct",
            "AI Interval Correct",
            "Market Detection"
        ]
    }


# ============================================================
#  AI HELPERS — CLEAN TEXT
# ============================================================

def clean_text(t: str) -> str:
    t = t.upper()
    t = t.replace("|", " ")
    t = t.replace(":", " ")
    t = t.replace(";", " ")
    t = re.sub(r"[^A-Z0-9ĄĆĘŁŃÓŚŹŻ .,%/\-]", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


# ============================================================
#  AI HELPERS — NUMBERS (O/H/L/C, MA, EMA, RSI, VOL)
# ============================================================

def extract_number_candidates(text: str):
    # liczby typu 123, 123.45, 1 234,56, 1,234.56
    raw = re.findall(r"[0-9][0-9., ]*[0-9]", text)
    cleaned = []
    for r in raw:
        s = r.replace(" ", "")
        # jeśli są dwie kropki/przecinki, spróbuj uprościć
        if s.count(".") + s.count(",") > 1:
            # weź tylko pierwsze wystąpienie separatora
            s = re.sub(r"[.,]", ".", s, count=1)
            s = re.sub(r"[.,]", "", s)
        s = s.replace(",", ".")
        try:
            v = float(s)
            cleaned.append(v)
        except:
            continue
    return cleaned


def find_labeled_number(text: str, labels):
    """
    Szuka liczby po etykietach typu O, H, L, C, MA20, EMA9 itd.
    """
    for lbl in labels:
        # np. "O 123.45", "O=123.45", "O: 123.45"
        pattern = rf"{lbl}\s*[:=]?\s*([0-9., ]+)"
        m = re.search(pattern, text)
        if m:
            candidates = extract_number_candidates(m.group(1))
            if candidates:
                return candidates[0]
    return None


# ============================================================
#  AI HELPERS — INTERWAŁ
# ============================================================

def normalize_interval(raw: str) -> str:
    raw = raw.upper().replace(" ", "")
    # typowe warianty: M1, 1M, M15, 15M, H1, 1H, D1, 1D, W1, 1W
    mapping = {
        "1M": "M1",
        "M1": "M1",
        "M1S": "M1",   # Twój case: M1S → M1
        "5M": "M5",
        "M5": "M5",
        "15M": "M15",
        "M15": "M15",
        "30M": "M30",
        "M30": "M30",
        "1H": "H1",
        "H1": "H1",
        "4H": "H4",
        "H4": "H4",
        "1D": "D1",
        "D1": "D1",
        "1W": "W1",
        "W1": "W1",
        "1MN": "MN",
        "MN": "MN",
        "1MO": "MN",
    }
    return mapping.get(raw, "UNKNOWN")


def detect_interval_ai(text: str) -> str:
    text = text.upper()
    # wyciągamy wszystkie potencjalne tokeny z M/H/D/W
    tokens = re.findall(r"[MHWD][0-9]{1,2}|[0-9]{1,2}[MHWD]|MN|M1S", text)
    for tok in tokens:
        norm = normalize_interval(tok)
        if norm != "UNKNOWN":
            return norm
    return "UNKNOWN"


# ============================================================
#  AI HELPERS — MARKET DETECTION (GPW / USA / CRYPTO)
# ============================================================

GPW_TICKERS = {
    "KTY", "ALE", "CDR", "PKN", "PKO", "PEO", "PZU", "DNP", "JSW", "MBK",
    "LPP", "CPS", "OPL", "TPE", "KRU", "SPL", "ASB", "BDX", "CMR", "DVL",
    "DOM", "ECH", "ENA", "ENG", "FMF", "ATT", "BHW", "CAR", "LVC", "MAB",
    "MRC", "NEU", "PCO", "STP", "TEN", "WPL", "11B", "EAT", "APT", "ASE",
    "BML", "BRS", "CLN", "CIE", "COG", "CMP", "DBC", "FRO", "FTE", "MRB",
    "MBR", "NWG", "PCE", "PWX", "QRS", "RBW", "SNK", "SNT", "TIM", "TOR",
    "VOT", "VRG", "WLT", "ZEP", "XTB", "KGH", "COPPER"
}

CRYPTO_KEYWORDS = {"BTC", "ETH", "USDT", "USDC", "SOL", "XRP", "BNB", "DOGE"}


def detect_market(ticker: str, text: str) -> str:
    t = text.upper()

    # 1) crypto po słowach kluczowych
    if any(k in t for k in CRYPTO_KEYWORDS):
        return "CRYPTO"

    # 2) GPW po tickerze
    if ticker in GPW_TICKERS:
        return "GPW"

    # 3) USA po typowych sufiksach / kontekstach
    if ".US" in t or "NASDAQ" in t or "NYSE" in t or "SP500" in t:
        return "USA"

    # fallback — jeśli nic nie pasuje
    return "UNKNOWN"


# ============================================================
#  AI HELPERS — TICKER (prosty, AI‑friendly)
# ============================================================

def detect_ticker_ai(text: str) -> str:
    text = clean_text(text)
    words = re.findall(r"[A-ZĄĆĘŁŃÓŚŹŻ]{2,6}", text)
    candidates = []
    for w in words:
        if any(ch.isdigit() for ch in w):
            continue
        candidates.append(w)
    if not candidates:
        return "UNKNOWN"
    candidates.sort(key=len, reverse=True)
    return candidates[0]


# ============================================================
#  HUGGINGFACE DONUT CALL
# ============================================================

def call_donut_api(image_bytes: bytes):
    """
    Wysyła obraz do HuggingFace Donut i zwraca JSON/tekst.
    """
    response = requests.post(
        HF_URL,
        headers=HF_HEADERS,
        data=image_bytes,
        timeout=60
    )

    try:
        return response.json()
    except Exception:
        return {"error": "Invalid response from HuggingFace"}


def extract_text_from_donut_result(result):
    """
    Donut zwykle zwraca listę z obiektami, np.:
    [
      {"generated_text": "{...json...}"}
    ]
    """
    if isinstance(result, list) and len(result) > 0:
        result = result[0]

    if isinstance(result, dict):
        if "generated_text" in result:
            return str(result["generated_text"])
        if "answer" in result:
            return str(result["answer"])

    # fallback — zwróć cokolwiek
    return str(result)


# ============================================================
#  ENDPOINT: /ocr — AI Vision + AI Auto‑Correct
# ============================================================

@app.post("/ocr")
async def ocr_endpoint(file: UploadFile = File(...)):
    t0 = time.time()

    image_bytes = await file.read()
    if not image_bytes:
        return {"error": "EMPTY_FILE"}

    # 1) AI Vision (Donut)
    raw_result = call_donut_api(image_bytes)

    if isinstance(raw_result, dict) and "error" in raw_result:
        return {"error": raw_result["error"]}

    raw_text = extract_text_from_donut_result(raw_result)
    clean = clean_text(raw_text)

    # 2) AI Ticker
    ticker = detect_ticker_ai(clean)

    # 3) AI Interval
    interval = detect_interval_ai(clean)

    # 4) AI Numbers (O/H/L/C, MA, EMA, RSI, VOL, RVOL)
    O = find_labeled_number(clean, ["O", "OPEN"])
    H = find_labeled_number(clean, ["H", "HIGH"])
    L = find_labeled_number(clean, ["L", "LOW"])
    C = find_labeled_number(clean, ["C", "CLOSE"])

    MA20 = find_labeled_number(clean, ["MA20", "MA 20", "SMA20", "SMA 20"])
    EMA9 = find_labeled_number(clean, ["EMA9", "EMA 9"])
    SMA50 = find_labeled_number(clean, ["SMA50", "SMA 50"])
    DEMA9 = find_labeled_number(clean, ["DEMA9", "DEMA 9"])
    RSI = find_labeled_number(clean, ["RSI"])
    VOL = find_labeled_number(clean, ["WOLUMEN", "VOLUME", "VOL", "WOL"])
    RVOL = find_labeled_number(clean, ["RVOL", "R-VOL", "REL VOL"])

    # 5) AI Market Detection
    market = detect_market(ticker, clean)

    return {
        "ticker": ticker,
        "interval": interval,
        "market": market,
        "O": O,
        "H": H,
        "L": L,
        "C": C,
        "MA20": MA20,
        "EMA9": EMA9,
        "DEMA9": DEMA9,
        "SMA50": SMA50,
        "RSI": RSI,
        "VOL": VOL,
        "RVOL": RVOL,
        "raw_text": raw_text,
        "clean_text": clean,
        "ocr_time_ms": round((time.time() - t0) * 1000, 1)
    }
