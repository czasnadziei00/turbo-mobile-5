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

ocr = PaddleOCR(
    lang="en",
    use_angle_cls=True,
    ocr_version="PP-OCRv4"
)




# ============================================================
#  POMOCNICZE: CZYSZCZENIE TEKSTU
# ============================================================

def clean_text(t: str) -> str:
    t = t.upper()
    t = t.replace("|", " ")
    t = t.replace(":", " ")
    t = t.replace(";", " ")
    t = re.sub(r"\s+", " ", t)
    return t.strip()


# ============================================================
#  CROP TICKER AREA (GÓRNY PASEK XTB)
# ============================================================

def crop_ticker_area(img: np.ndarray) -> np.ndarray:
    h, w, _ = img.shape
    return img[0:int(h * 0.15), 0:int(w * 0.7)]


# ============================================================
#  PREPROCESS (GRAY + BLUR + THRESH)
# ============================================================

def preprocess(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY)[1]
    return thresh


# ============================================================
#  OCR → TEKST
# ============================================================

def extract_text(img: np.ndarray) -> str:
    result = ocr.ocr(img, cls=True)
    if not result or not result[0]:
        return ""
    return clean_text(" ".join([line[1][0] for line in result[0]]))


# ============================================================
#  INTERWAŁ (M1/M5/M15/H1/D1/...)
# ============================================================

def detect_interval(text: str) -> str:
    intervals = ["M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1", "MN"]
    for i in intervals:
        if i in text:
            return i
    return "UNKNOWN"


# ============================================================
#  SŁOWNIK TICKERÓW
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

def detect_ticker_smart(text: str) -> str:
    words = re.findall(r"[A-ZĄĆĘŁŃÓŚŹŻ]{2,6}", text)
    candidates = []
    for w in words:
        if w in BLACKLIST:
            continue
        if any(ch.isdigit() for ch in w):
            continue
        candidates.append(w)
    if not candidates:
        return "UNKNOWN"
    candidates.sort(key=len, reverse=True)
    return candidates[0]


def detect_ticker(text: str) -> str:
    for k, t in TICKER_MAP.items():
        if k in text:
            return t
    return detect_ticker_smart(text)


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
#  GŁÓWNY ENDPOINT OCR — POPRAWIONY
# ============================================================

@app.post("/ocr")
async def ocr_endpoint(file: UploadFile = File(...)):
    content = await file.read()
    print("FILE SIZE:", len(content))

    img_array = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    print("IMG:", img is not None)

    if img is None:
        return {"error": "IMAGE_DECODE_FAILED"}

    clean = preprocess(img)
    text = extract_text(clean)

    ticker_img = crop_ticker_area(img)
    ticker_text = extract_text(preprocess(ticker_img))
    ticker = detect_ticker(ticker_text)

    if ticker == "UNKNOWN":
        ticker = detect_ticker(text)

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
    }
