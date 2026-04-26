from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pytesseract
import cv2
import numpy as np
import re
import time
from rapidfuzz import fuzz, process

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
#  HEALTHCHECK
# ============================================================

@app.get("/health")
def health():
    return {"status": "OK", "engine": "tesseract-pro", "version": "1.1-ai-ticker"}


# ============================================================
#  OCR PRO — TESSERACT (Render Free SAFE)
# ============================================================

def ocr_text(img: np.ndarray) -> str:
    """
    PRO OCR:
    - odszumianie
    - adaptacyjny threshold
    - fallback OCR
    - czyszczenie artefaktów
    """

    # 1) grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2) odszumianie
    denoise = cv2.fastNlMeansDenoising(gray, h=10)

    # 3) threshold adaptacyjny (lepszy niż stały)
    thresh = cv2.adaptiveThreshold(
        denoise, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 5
    )

    # 4) OCR — pierwsza próba
    text = pytesseract.image_to_string(thresh, config="--psm 6")
    text = clean_text(text)

    # 5) fallback — jeśli za mało znaków
    if len(text) < 3:
        fallback = pytesseract.image_to_string(denoise, config="--psm 6")
        fallback = clean_text(fallback)
        if len(fallback) > len(text):
            text = fallback

    return text


# ============================================================
#  LOGI OCR
# ============================================================

def log_ocr(label: str, text: str, t_start: float):
    duration = round((time.time() - t_start) * 1000, 1)
    print(f"[OCR] {label}: {len(text)} chars, {duration} ms, text='{text[:40]}'")


# ============================================================
#  CLEAN TEXT
# ============================================================

def clean_text(t: str) -> str:
    t = t.upper()
    t = t.replace("|", " ")
    t = t.replace(":", " ")
    t = t.replace(";", " ")
    t = re.sub(r"[^A-Z0-9ĄĆĘŁŃÓŚŹŻ .,-]", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


# ============================================================
#  CROP TICKER AREA
# ============================================================

def crop_ticker_area(img: np.ndarray) -> np.ndarray:
    h, w, _ = img.shape
    return img[0:int(h * 0.15), 0:int(w * 0.7)]


# ============================================================
#  PREPROCESS (dla liczb)
# ============================================================

def preprocess(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY)[1]
    return thresh


# ============================================================
#  INTERWAŁ
# ============================================================

def detect_interval(text: str) -> str:
    intervals = ["M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1", "MN"]
    for i in intervals:
        if i in text:
            return i
    return "UNKNOWN"


# ============================================================
#  TICKERY (słownik)
# ============================================================

TICKER_MAP = {
    "GRUPA KETY": "KTY",
    "GRUPA KĘTY": "KTY",
    "KETY": "KTY",
    "KĘTY": "KTY",
    "ALLEGRO": "ALE",
    "CD PROJEKT": "CDR",
    "CDPROJEKT": "CDR",
    "PKN ORLEN": "PKN",
    "ORLEN": "PKN",
    "PKOBP": "PKO",
    "PEKAO": "PEO",
    "PZU": "PZU",
    "DINOPL": "DNP",
    "JSW": "JSW",
    "MBANK": "MBK",
    "LPP": "LPP",
    "CYFROWY POLSAT": "CPS",
    "CYFROWY": "CPS",
    "ORANGE": "OPL",
    "TAURON": "TPE",
    "KRUK": "KRU",
    "SANPL": "SPL",
    "ASBIS": "ASB",
    "BUDIMEX": "BDX",
    "COMARCH": "CMR",
    "DEVELIA": "DVL",
    "DOMDEV": "DOM",
    "ECHO": "ECH",
    "ENEA": "ENA",
    "ENERGA": "ENG",
    "FAMUR": "FMF",
    "GRUPA AZOTY": "ATT",
    "HANDLOWY": "BHW",
    "INTERCARS": "CAR",
    "LIVECHAT": "LVC",
    "MABION": "MAB",
    "MERCATOR": "MRC",
    "NEUCA": "NEU",
    "PEPCO": "PCO",
    "STALPRODUKT": "STP",
    "TEN SQUARE": "TEN",
    "WIRTUALNA POLSKA": "WPL",
    "11BIT": "11B",
    "AMREST": "EAT",
    "APATOR": "APT",
    "ASSECOSEE": "ASE",
    "BIOMED": "BML",
    "BORYSZEW": "BRS",
    "CELON": "CLN",
    "CIECH": "CIE",
    "COGNOR": "COG",
    "COMP": "CMP",
    "DEBICA": "DBC",
    "FERRO": "FRO",
    "FORTE": "FTE",
    "MIRBUD": "MRB",
    "MOBRUK": "MBR",
    "NEWAG": "NWG",
    "POLICE": "PCE",
    "POLWAX": "PWX",
    "QUERCUS": "QRS",
    "RAINBOW": "RBW",
    "SANOK": "SNK",
    "SYNEKTIK": "SNT",
    "TIM": "TIM",
    "TORPOL": "TOR",
    "VOTUM": "VOT",
    "VRG": "VRG",
    "WIELTON": "WLT",
    "ZEPAK": "ZEP",
    "XTB": "XTB",
    "KGHM": "KGH",
    "COPPER": "COPPER",
    "CU": "COPPER",
    "MIEDŹ": "COPPER",
}

BLACKLIST = {
    "MA", "EMA", "SMA", "DEMA", "RSI", "VOL", "WOLUMEN", "VOLUME",
    "GPW", "WIG20", "MWIG40", "SWIG80", "PLN", "BUY", "SELL",
    "STOP", "LOSS", "TAKE", "PROFIT", "TP", "SL", "OPEN", "CLOSE",
    "HIGH", "LOW", "H", "L", "O", "C"
}


# ============================================================
#  AI AUTO-CORRECT TICKERA (RapidFuzz)
# ============================================================

ALL_TICKERS = list(TICKER_MAP.values())

def autocorrect_ticker(raw: str) -> str:
    """
    AI fuzzy matching tickera:
    - poprawia literówki
    - poprawia błędy OCR
    - dopasowuje do najbliższego tickera
    """

    if not raw or len(raw) < 2:
        return "UNKNOWN"

    raw = raw.strip().upper()

    # Jeśli OCR zwrócił pełną nazwę spółki → mapowanie
    for name, ticker in TICKER_MAP.items():
        if name in raw:
            return ticker

    # Fuzzy match do listy tickerów
    match, score, _ = process.extractOne(raw, ALL_TICKERS, scorer=fuzz.ratio)

    # Jeśli dopasowanie słabe → odrzucamy
    if score < 60:
        return "UNKNOWN"

    return match


# ============================================================
#  PARSOWANIE LICZB
# ============================================================

def find_number(text: str, label: str):
    m = re.search(rf"{label}\s*([0-9.,]+)", text)
    if not m:
        return None
    return float(m.group(1).replace(",", "."))


def multi_find(text: str, labels):
    for lbl in labels:
        v = find_number(text, lbl)
        if v is not None:
            return v
    return None


# ============================================================
#  ENDPOINT OCR PRO
# ============================================================

@app.post("/ocr")
async def ocr_endpoint(file: UploadFile = File(...)):
    t0 = time.time()

    content = await file.read()
    img_array = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "IMAGE_DECODE_FAILED"}

    # OCR główny
    clean = preprocess(img)
    text = ocr_text(clean)
    log_ocr("MAIN", text, t0)

    # OCR ticker
    ticker_img = crop_ticker_area(img)
    ticker_text = ocr_text(preprocess(ticker_img))
    log_ocr("TICKER", ticker_text, t0)

    # AI Auto-Correct Tickera
    ticker_raw = detect_ticker(ticker_text)
    ticker = autocorrect_ticker(ticker_raw)

    if ticker == "UNKNOWN":
        ticker = autocorrect_ticker(detect_ticker(text))

    interval = detect_interval(text)

    O = find_number(text, "O")
    H = find_number(text, "H")
    L = find_number(text, "L")
    C = find_number(text, "C")

    MA20 = multi_find(text, ["MA20", "MA 20", "SMA20"])
    EMA9 = multi_find(text, ["EMA9", "EMA 9"])
    SMA50 = multi_find(text, ["SMA50", "SMA 50"])
    DEMA9 = multi_find(text, ["DEMA9", "DEMA 9"])
    RSI = multi_find(text, ["RSI"])
    VOL = multi_find(text, ["WOLUMEN", "VOLUME", "VOL", "WOL"])
    RVOL = multi_find(text, ["RVOL", "R-VOL", "REL VOL"])

    return {
        "ticker": ticker,
        "interval": interval,
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
        "ocr_time_ms": round((time.time() - t0) * 1000, 1)
    }
