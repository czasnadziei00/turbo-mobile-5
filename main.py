from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from paddleocr import PaddleOCR
from PIL import Image
import uvicorn
import shutil
import re

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========= OCR =========

ocr = PaddleOCR(
    use_angle_cls=True,
    lang="en"
)

def R(x1, y1, w, h):
    return {"x1": x1, "y1": y1, "x2": x1 + w, "y2": y1 + h}

# REGIONY – skopiowane z Twojego frontu
REGIONY = {
    "portrait": {
        "BLOK1": R(3, 291, 770, 286),    # OHLC + Wolumen
        "BLOK2": R(695, 116, 101, 76),   # interwał / MA info
        "BLOK3": R(8, 1267, 297, 76),    # RSI
        "BLOK4": R(2, 1778, 776, 60),    # godzina
    },
    "landscape": {
        "BLOK1": R(237, 149, 1532, 217),
        "BLOK2": R(86, 702, 121, 90),
        "BLOK3": R(106, 700, 108, 85),
        "BLOK4": R(228, 776, 1124, 50),
    }
}

def detect_orientation(img: Image.Image) -> str:
    w, h = img.size
    return "portrait" if h >= w else "landscape"

def czytaj_region(img: Image.Image, region: dict) -> str:
    x1, y1, x2, y2 = region["x1"], region["y1"], region["x2"], region["y2"]
    crop = img.crop((x1, y1, x2, y2))
    wynik = ocr.ocr(crop, cls=True)
    if not wynik or not wynik[0]:
        return ""
    txt = " ".join([w[1][0] for w in wynik[0]])
    return txt.strip()

def to_float(s: str):
    s = s.replace(" ", "").replace(",", ".")
    m = re.search(r"[-+]?\d+(\.\d+)?", s)
    return float(m.group(0)) if m else None

def parse_blok1(txt: str):
    # O1 101,00 H1 105,00 L1 101,00 C1 105,00 Wolumen 27
    O = H = L = C = VOL = None
    m = re.search(r"O1?\s*([\d\s,\.]+)", txt)
    if m: O = to_float(m.group(1))
    m = re.search(r"H1?\s*([\d\s,\.]+)", txt)
    if m: H = to_float(m.group(1))
    m = re.search(r"L1?\s*([\d\s,\.]+)", txt)
    if m: L = to_float(m.group(1))
    m = re.search(r"C1?\s*([\d\s,\.]+)", txt)
    if m: C = to_float(m.group(1))
    m = re.search(r"Wolumen\s*([\d\s,\.]+)", txt, re.IGNORECASE)
    if m: VOL = int(to_float(m.group(1)) or 0)
    return O, H, L, C, VOL

def parse_blok2(txt: str):
    # MA 20 close 1 106,45  /  DEMA 9 1 105,14
    MA20 = None
    DEMA9 = None
    m = re.search(r"MA\s*20.*?([\d\s,\.]+)", txt, re.IGNORECASE)
    if m: MA20 = to_float(m.group(1))
    m = re.search(r"DEMA\s*9.*?([\d\s,\.]+)", txt, re.IGNORECASE)
    if m: DEMA9 = to_float(m.group(1))
    return MA20, DEMA9

def parse_blok3(txt: str):
    # RSI14 49,82
    RSI = None
    m = re.search(r"RSI\s*14?\s*([\d\s,\.]+)", txt, re.IGNORECASE)
    if m: RSI = to_float(m.group(1))
    return RSI

@app.post("/ocr")
async def ocr_endpoint(file: UploadFile = File(...)):
    try:
        temp_path = "temp.jpg"
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        img = Image.open(temp_path).convert("RGB")
        orient = detect_orientation(img)
        reg = REGIONY[orient]

        t1 = czytaj_region(img, reg["BLOK1"])
        t2 = czytaj_region(img, reg["BLOK2"])
        t3 = czytaj_region(img, reg["BLOK3"])
        # t4 = czytaj_region(img, reg["BLOK4"])  # na razie nie używamy

        O, H, L, C = None, None, None, None
        VOL = None
        MA20 = None
        DEMA9 = None
        RSI = None

        if t1:
            O, H, L, C, VOL = parse_blok1(t1)
        if t2:
            MA20, DEMA9 = parse_blok2(t2)
        if t3:
            RSI = parse_blok3(t3)

        dane = {
            "ticker": file.filename.split(".")[0].upper(),
            "O": O,
            "H": H,
            "L": L,
            "C": C,
            "MA20": MA20,
            "DEMA9": DEMA9,
            "RSI": RSI,
            "VOL": VOL,
            "interval": "M15"
        }

        return dane

    except Exception as e:
        return {"error": str(e)}

@app.get("/")
def root():
    return {"status": "OK", "message": "Turbo Mobile OCR backend działa"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
