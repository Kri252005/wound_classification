from pathlib import Path
import os

BACKEND_DIR = Path(__file__).resolve().parent
MODEL_DIR = BACKEND_DIR / "model"
WEIGHTS_PATH = MODEL_DIR / "model.weights.h5"
ARCHITECTURE_PATH = MODEL_DIR / "config.json"
METADATA_PATH = MODEL_DIR / "metadata.json"
ALLOWED_ORIGINS = [origin.strip() for origin in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",") if origin.strip()]
ALLOWED_TYPES = {"image/jpeg", "image/png"}
CLASS_NAMES = ("Abrasions", "Bruises", "Burns", "Cut", "Laceration", "Stab_wound")
