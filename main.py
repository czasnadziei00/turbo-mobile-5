from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from paddleocr import PaddleOCR
from PIL import Image
import uvicorn
import shutil

# ============================================================
# FASTAPI + CORS
# ============================================================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# OCR — ŁADOWANIE MODELU (bez show_log!)
# ============================================================

ocr = PaddleOCR(
    use_angle_cls=True,
    lang="en"
)

# ============================================================
# ENDPOINT OCR
# ============================================================

@app.post("/ocr")
async def ocr_endpoint(file: UploadFile = File(...)):
    try:
        temp_path = "temp.jpg"
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        img = Image.open(temp_path).convert("RGB")

        # przykładowe dane — frontend sam liczy sygnały
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
# ROOT
# ============================================================

@app.get("/")
def root():
    return {"status": "OK", "message": "Turbo Mobile OCR backend działa"}

# ============================================================
# UVICORN
# ============================================================

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
