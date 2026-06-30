from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io

from utils.predictor import predict

app = FastAPI(
    title="OsteoScan AI API",
    version="1.0"
)


@app.get("/")
def home():
    return {
        "message": "OsteoScan AI API Running Successfully"
    }


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    try:
        # Read uploaded image
        image_bytes = await file.read()

        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

        # Prediction
        label, confidence = predict(image)

        # Convert NumPy types to Python types
        label = str(label)
        confidence = float(confidence)

        return JSONResponse(
            content={
                "prediction": label,
                "confidence": round(confidence, 2)
            }
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )