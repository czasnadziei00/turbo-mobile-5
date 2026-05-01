from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import shutil
import os
from paddlex import create_model
from paddleocr import PaddleOCR
from PIL import Image

# ============================================================
# FASTAPI + CORS
# ============================================================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # pozwalamy na połączenia z localhost i Render
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# MODELE OCR — ŁADOWANIE
# ============================================================

ocr = PaddleOCR(
    use_angle_cls=True,
    lang="en",
    show_log=False
)

# ============================================================
# FUNKCJA OCR — CZYTANIE BLOKÓW
# ============================================================

def czytaj_region(img, region):
    x1, y1, x2, y2 = region["x1"], region["y1"], region["x2"], region["y2"]
    crop = img.crop((x1, y1, x2, y2))
    wynik = ocr.ocr(crop, cls=True)
    tekst = " ".join([w[1][0] for w in wynik[0]]) if wynik and wynik[0] else ""
    return tekst.strip()

# ============================================================
# ENDPOINT OCR
# ============================================================

@app.post("/ocr")
async def ocr_endpoint(file: UploadFile = File(...)):
    try:
        # zapis pliku tymczasowego
        temp_path = "temp.jpg"
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        img = Image.open(temp_path).convert("RGB")

        # przykładowe dane — backend zwraca strukturę zgodną z frontendem
        # (frontend sam liczy widełki, sygnały, trailing itd.)
        dane = {
            "ticker": "COPPER",
            "O": 4.123,
            "H": 4.200,
            "L": 4.090,
            "C": 4.150,
            "MA20": 4.130,
            "DEMA9": 4.145,
            "RSI": 58,
            "VOL": 2200,
            "interval": "M15"
        }

        return dane

    except Exception as e:
        return {"error": str(e)}

# ============================================================
# ROOT — opcjonalnie
# ============================================================

@app.get("/")
def root():
    return {"status": "OK", "message": "Turbo Mobile OCR backend działa"}

# ============================================================
# START UVICORN (Render używa CMD z Dockerfile)
# ============================================================

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
