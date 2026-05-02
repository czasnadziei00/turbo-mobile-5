from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/ocr")
async def ocr_endpoint(file: UploadFile = File(...)):
    try:
        content = await file.read()
        img = Image.open(io.BytesIO(content))
        img.verify()

        return {
            "status": "OK",
            "msg": "Backend działa. OCR wykonuje Tesseract.js w przeglądarce."
        }

    except Exception as e:
        return {"status": "ERROR", "msg": str(e)}

@app.get("/")
def root():
    return {"status": "OK"}
