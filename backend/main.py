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

        print(
            f"WoundScope ML Backend\n"
            f"Wound detector: model\n"
            f"Wound classifier: m_model\n"
            f"Input: 224x224x3\n"
            f"Classes: {len(model_service.class_names)}\n"
            f"Model status: READY"
        )

    except Exception as exc:
        print(
            f"WoundScope ML Backend\n"
            f"Model status: UNAVAILABLE\n"
            f"Error: {exc}"
        )

    yield


app = FastAPI(
    title="WoundScope ML API",
    version="1.0",
    lifespan=lifespan
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"]
)


@app.get("/health")
def health():

    wound_loaded = model_service.wound_model is not None

    classification_loaded = (
        model_service.classification_model is not None
    )

    return {
        "status": (
            "healthy"
            if wound_loaded and classification_loaded
            else "unavailable"
        ),
        "wound_model_loaded": wound_loaded,
        "classification_model_loaded": classification_loaded,
        "wound_detector": "model",
        "wound_classifier": "m_model",
        "classes": len(model_service.class_names)
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    if (
        model_service.wound_model is None
        or model_service.classification_model is None
    ):
        raise HTTPException(
            503,
            "Classification models are currently unavailable."
        )

    image, source_width, source_height = await decode_upload(file)

    try:
        prediction, processing_time = model_service.predict(image)

    except ModelUnavailableError as exc:
        raise HTTPException(503, str(exc)) from exc

    return {
        "success": True,

        "analysis_id": f"WS-{uuid.uuid4().hex[:8].upper()}",

        "prediction": {
            "class": prediction["class"],
            "class_index": prediction["class_index"],
            "confidence": prediction["confidence"],
            "is_wound": prediction["is_wound"]
        },

        "probabilities": prediction["probabilities"],

        "model": {
            "name": "WoundScope Two-Stage Model",
            "version": "1.0",
            "wound_detector": "model",
            "wound_classifier": "m_model"
        },

        "image": {
            "width": 224,
            "height": 224,
            "source_width": source_width,
            "source_height": source_height
        },

        "processing_time_ms": round(processing_time, 2),

        "timestamp": int(time.time())
    }