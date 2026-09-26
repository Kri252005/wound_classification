import json
import time
from pathlib import Path
from typing import Any
import numpy as np
import tensorflow as tf
from tensorflow import keras
from backend.config import ARCHITECTURE_PATH, CLASS_NAMES, METADATA_PATH, WEIGHTS_PATH

class ModelUnavailableError(RuntimeError):
    pass

class WoundModelService:
    """A singleton inference service. The model accepts raw [0, 255] RGB pixels;
    DenseNet preprocessing is embedded in the reconstructed Functional model."""
    def __init__(self) -> None:
        self.model: keras.Model | None = None
        self.metadata: dict[str, Any] = {}
        self.class_names: tuple[str, ...] = CLASS_NAMES

    def load(self) -> None:
        for path in (WEIGHTS_PATH, ARCHITECTURE_PATH, METADATA_PATH):
            if not path.is_file():
                raise ModelUnavailableError(f"Required model artifact is missing: {path.name}")
        self.metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        config = json.loads(ARCHITECTURE_PATH.read_text(encoding="utf-8"))
        self._validate_class_mapping(config)
        model = self._model_from_config(config)
        try:
            model.load_weights(WEIGHTS_PATH)
        except Exception as exc:
            raise ModelUnavailableError(f"Unable to load model.weights.h5: {exc}") from exc
        if model.input_shape != (None, 224, 224, 3):
            raise ModelUnavailableError(f"Unexpected model input shape: {model.input_shape}")
        if model.output_shape != (None, len(self.class_names)):
            raise ModelUnavailableError(f"Unexpected model output shape: {model.output_shape}")
        self.model = model

    def _validate_class_mapping(self, config: dict[str, Any]) -> None:
        # A config/metadata class order, if supplied, must agree with the training order.
        supplied = self.metadata.get("class_names") or self.metadata.get("classes") or config.get("class_names")
        if isinstance(supplied, dict):
            try:
                supplied = [supplied[str(index)] if str(index) in supplied else supplied[index] for index in range(len(supplied))]
            except KeyError as exc:
                raise ModelUnavailableError("Metadata class indices must be contiguous from 0.") from exc
        if supplied:
            if not isinstance(supplied, list) or not all(isinstance(name, str) for name in supplied):
                raise ModelUnavailableError("Metadata class_names must be an ordered list of strings.")
            self.class_names = tuple(supplied)

    def _model_from_config(self, config: dict[str, Any]) -> keras.Model:
        # Prefer a complete Keras configuration when the provided config contains one.
        serialized = config.get("model_config") or config.get("keras_model")
        if isinstance(serialized, str):
            return keras.models.model_from_json(serialized)
        if isinstance(serialized, dict) and "class_name" in serialized:
            return keras.models.model_from_json(json.dumps(serialized))
        if "class_name" in config and "config" in config:
            return keras.models.model_from_json(json.dumps(config))
        # config.json contains no complete serialised graph: reconstruct exactly from
        # the supplied original training architecture, with weights=None because the
        # provided H5 file is the source of all trained weights.
        augmentation = keras.Sequential([
            keras.layers.RandomFlip("horizontal"), keras.layers.RandomRotation(0.1),
            keras.layers.RandomZoom(0.1), keras.layers.RandomContrast(0.1),
        ], name="data_augmentation")
        inputs = keras.Input(shape=(224, 224, 3))
        x = augmentation(inputs)
        x = keras.applications.densenet.preprocess_input(x)
        base = keras.applications.DenseNet169(include_top=False, weights=None)
        x = base(x, training=False)
        x = keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.Dense(128, activation="relu")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.3)(x)
        outputs = keras.layers.Dense(6, activation="softmax")(x)
        return keras.Model(inputs, outputs, name="woundscope_densenet169")

    def predict(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        if self.model is None:
            raise ModelUnavailableError("Classification model is currently unavailable.")
        started = time.perf_counter()
        result = self.model.predict(image, verbose=0)
        elapsed = (time.perf_counter() - started) * 1000
        values = np.asarray(result[0], dtype=np.float64)
        if values.shape != (len(self.class_names),) or not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1) or not np.isclose(values.sum(), 1.0, atol=1e-3):
            raise ModelUnavailableError("Model output failed probability validation.")
        return values, elapsed

model_service = WoundModelService()
