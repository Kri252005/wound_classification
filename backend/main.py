import time
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from backend.config import ALLOWED_ORIGINS
from backend.services.model_service import ModelUnavailableError, model_service
from backend.services.preprocessing import decode_upload

@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        model_service.load()
        print(f"WoundScope ML Backend\nModel: DenseNet169\nInput: 224x224x3\nClasses: {len(model_service.class_names)}\nModel status: READY")
    except Exception as exc:
        print(f"WoundScope ML Backend\nModel status: UNAVAILABLE\nError: {exc}")
    yield

app = FastAPI(title="WoundScope ML API", version="1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_credentials=False, allow_methods=["GET", "POST"], allow_headers=["*"])

@app.get("/health")
def health():
    return {"status": "healthy" if model_service.model else "unavailable", "model_loaded": model_service.model is not None, "model": "DenseNet169", "classes": len(model_service.class_names)}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model_service.model is None:
        raise HTTPException(503, "Classification model is currently unavailable.")
    image, source_width, source_height = await decode_upload(file)
    try:
        values, processing_time = model_service.predict(image)
    except ModelUnavailableError as exc:
        raise HTTPException(503, "Classification model is currently unavailable.") from exc
    winner = int(values.argmax())
    return {"success": True, "analysis_id": f"WS-{uuid.uuid4().hex[:8].upper()}", "prediction": {"class": model_service.class_names[winner], "class_index": winner, "confidence": float(values[winner])}, "probabilities": [{"class": name, "class_index": index, "probability": float(values[index])} for index, name in enumerate(model_service.class_names)], "model": {"name": "DenseNet169", "version": "1.0"}, "image": {"width": 224, "height": 224, "source_width": source_width, "source_height": source_height}, "processing_time_ms": round(processing_time, 2), "timestamp": int(time.time())}
