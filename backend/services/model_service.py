import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from tensorflow import keras

from backend.config import (
    ARCHITECTURE_PATH,
    CLASS_NAMES,
    METADATA_PATH,
    WEIGHTS_PATH,
)


class ModelUnavailableError(RuntimeError):
    pass


class WoundModelService:
    """
    Inference service for the WoundScope model.

    The model accepts:
        [0, 255] RGB images

    Expected input:
        (None, 224, 224, 3)

    The current saved model may produce a single output value.
    """

    def __init__(self) -> None:
        self.model: keras.Model | None = None
        self.metadata: dict[str, Any] = {}
        self.class_names: tuple[str, ...] = CLASS_NAMES

    def load(self) -> None:
        # ---------------------------------------------------------
        # 1. Check that all model files exist
        # ---------------------------------------------------------
        for path in (
            WEIGHTS_PATH,
            ARCHITECTURE_PATH,
            METADATA_PATH,
        ):
            if not path.is_file():
                raise ModelUnavailableError(
                    f"Required model artifact is missing: {path.name}"
                )

        # ---------------------------------------------------------
        # 2. Load metadata
        # ---------------------------------------------------------
        self.metadata = json.loads(
            METADATA_PATH.read_text(encoding="utf-8")
        )

        # ---------------------------------------------------------
        # 3. Load architecture
        # ---------------------------------------------------------
        config = json.loads(
            ARCHITECTURE_PATH.read_text(encoding="utf-8")
        )

        self._validate_class_mapping(config)

        # ---------------------------------------------------------
        # 4. Reconstruct model
        # ---------------------------------------------------------
        model = self._model_from_config(config)

        # ---------------------------------------------------------
        # 5. Load trained weights
        # ---------------------------------------------------------
        try:
            model.load_weights(WEIGHTS_PATH)
        except Exception as exc:
            raise ModelUnavailableError(
                f"Unable to load model.weights.h5: {exc}"
            ) from exc

        # ---------------------------------------------------------
        # 6. Validate input shape
        # ---------------------------------------------------------
        if model.input_shape != (None, 224, 224, 3):
            raise ModelUnavailableError(
                f"Unexpected model input shape: {model.input_shape}"
            )

        # ---------------------------------------------------------
        # IMPORTANT:
        # Do NOT require output_shape == (None, 6)
        #
        # Your new model currently reports:
        # (None, 1)
        # ---------------------------------------------------------
        print(f"Model input shape: {model.input_shape}")
        print(f"Model output shape: {model.output_shape}")

        self.model = model

    def _validate_class_mapping(
        self,
        config: dict[str, Any]
    ) -> None:

        supplied = (
            self.metadata.get("class_names")
            or self.metadata.get("classes")
            or config.get("class_names")
        )

        if isinstance(supplied, dict):
            try:
                supplied = [
                    supplied[str(index)]
                    if str(index) in supplied
                    else supplied[index]
                    for index in range(len(supplied))
                ]
            except KeyError as exc:
                raise ModelUnavailableError(
                    "Metadata class indices must be contiguous from 0."
                ) from exc

        if supplied:
            if (
                not isinstance(supplied, list)
                or not all(isinstance(name, str) for name in supplied)
            ):
                raise ModelUnavailableError(
                    "Metadata class_names must be an ordered list of strings."
                )

            self.class_names = tuple(supplied)

    def _model_from_config(
        self,
        config: dict[str, Any]
    ) -> keras.Model:

        # ---------------------------------------------------------
        # Prefer complete serialized Keras configuration
        # ---------------------------------------------------------
        serialized = (
            config.get("model_config")
            or config.get("keras_model")
        )

        if isinstance(serialized, str):
            return keras.models.model_from_json(serialized)

        if (
            isinstance(serialized, dict)
            and "class_name" in serialized
        ):
            return keras.models.model_from_json(
                json.dumps(serialized)
            )

        if (
            "class_name" in config
            and "config" in config
        ):
            return keras.models.model_from_json(
                json.dumps(config)
            )

        # ---------------------------------------------------------
        # Fallback architecture
        #
        # NOTE:
        # This is only used when config.json does NOT contain
        # the complete serialized model.
        # ---------------------------------------------------------

        augmentation = keras.Sequential(
            [
                keras.layers.RandomFlip("horizontal"),
                keras.layers.RandomRotation(0.1),
                keras.layers.RandomZoom(0.1),
                keras.layers.RandomContrast(0.1),
            ],
            name="data_augmentation",
        )

        inputs = keras.Input(
            shape=(224, 224, 3)
        )

        x = augmentation(inputs)

        x = keras.applications.densenet.preprocess_input(x)

        base = keras.applications.DenseNet169(
            include_top=False,
            weights=None,
        )

        x = base(x, training=False)

        x = keras.layers.GlobalAveragePooling2D()(x)

        x = keras.layers.Dense(
            128,
            activation="relu"
        )(x)

        x = keras.layers.BatchNormalization()(x)

        x = keras.layers.Dropout(0.3)(x)

        # ---------------------------------------------------------
        # IMPORTANT:
        # We do NOT create Dense(6, softmax) here anymore.
        #
        # The new model's actual output must come from its
        # serialized config.
        # ---------------------------------------------------------

        outputs = keras.layers.Dense(
            1,
            activation="sigmoid"
        )(x)

        return keras.Model(
            inputs,
            outputs,
            name="woundscope_densenet169",
        )

    def predict(
        self,
        image: np.ndarray
    ) -> tuple[np.ndarray, float]:

        if self.model is None:
            raise ModelUnavailableError(
                "Classification model is currently unavailable."
            )

        started = time.perf_counter()

        result = self.model.predict(
            image,
            verbose=0
        )

        elapsed = (
            time.perf_counter() - started
        ) * 1000

        # ---------------------------------------------------------
        # Convert model output to numpy
        # ---------------------------------------------------------
        raw = np.asarray(
            result,
            dtype=np.float64
        )

        print(
            f"Raw model output shape: {raw.shape}"
        )

        print(
            f"Raw model output: {raw}"
        )

        # ---------------------------------------------------------
        # Expected current output:
        #
        # (batch_size, 1)
        #
        # Example:
        # [[0.83]]
        # ---------------------------------------------------------
        if raw.ndim != 2 or raw.shape[0] != 1:
            raise ModelUnavailableError(
                f"Unexpected prediction shape: {raw.shape}"
            )

        if raw.shape[1] != 1:
            raise ModelUnavailableError(
                f"Expected one model output value, got: {raw.shape}"
            )

        value = float(raw[0][0])

        if not np.isfinite(value):
            raise ModelUnavailableError(
                "Model returned an invalid prediction."
            )

        # ---------------------------------------------------------
        # Convert single output into a temporary result.
        #
        # This part will be finalized after confirming what the
        # single output represents in the new model.
        # ---------------------------------------------------------

        if value >= 0.5:
            predicted_class = self.class_names[0]
        else:
            predicted_class = self.class_names[0]

        values = np.zeros(
            len(self.class_names),
            dtype=np.float64
        )

        values[0] = value

        return values, elapsed


model_service = WoundModelService()