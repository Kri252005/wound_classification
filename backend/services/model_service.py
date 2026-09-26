import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from tensorflow import keras


class ModelUnavailableError(RuntimeError):
    pass


class WoundModelService:
    """
    Two-stage wound classification service.

    Stage 1:
    model -> Wound / Not Wound

    Stage 2:
    m_model -> Six wound classes

    Both models accept:
        (None, 224, 224, 3)
    """

    def __init__(self) -> None:
        self.wound_model: keras.Model | None = None
        self.classification_model: keras.Model | None = None

        self.wound_metadata: dict[str, Any] = {}
        self.classification_metadata: dict[str, Any] = {}

        self.class_names = (
            "Abrasions",
            "Bruises",
            "Burns",
            "Cut",
            "Laceration",
            "Stab_wound",
        )

        # backend/
        self.backend_dir = Path(__file__).resolve().parents[1]

        
        # backend/model/ → Wound / Not Wound model
        self.wound_model_dir = self.backend_dir / "model"

        # backend/m_model/ → 6-class wound model
        self.classification_model_dir = self.backend_dir / "m_model"

    # =========================================================
    # LOAD BOTH MODELS
    # =========================================================

    def load(self) -> None:
        print("Loading WoundScope models...")

        # -----------------------------------------------------
        # Stage 1: Wound / Not Wound model
        # -----------------------------------------------------
        self.wound_model, self.wound_metadata = self._load_single_model(
            self.wound_model_dir,
            "wound detection model",
        )

        # -----------------------------------------------------
        # Stage 2: Six-class wound classification model
        # -----------------------------------------------------
        self.classification_model, self.classification_metadata = (
            self._load_single_model(
                self.classification_model_dir,
                "wound classification model",
            )
        )

        # -----------------------------------------------------
        # Validate inputs
        # -----------------------------------------------------
        if self.wound_model.input_shape != (None, 224, 224, 3):
            raise ModelUnavailableError(
                f"Unexpected wound model input shape: "
                f"{self.wound_model.input_shape}"
            )

        if self.classification_model.input_shape != (
            None,
            224,
            224,
            3,
        ):
            raise ModelUnavailableError(
                f"Unexpected classification model input shape: "
                f"{self.classification_model.input_shape}"
            )

        print(
            f"Wound model input: {self.wound_model.input_shape}"
        )
        print(
            f"Wound model output: {self.wound_model.output_shape}"
        )

        print(
            f"Classification model input: "
            f"{self.classification_model.input_shape}"
        )
        print(
            f"Classification model output: "
            f"{self.classification_model.output_shape}"
        )

        print("WoundScope models loaded successfully.")

    # =========================================================
    # LOAD ONE MODEL
    # =========================================================

    def _load_single_model(
        self,
        model_dir: Path,
        model_name: str,
    ) -> tuple[keras.Model, dict[str, Any]]:

        weights_path = model_dir / "model.weights.h5"
        config_path = model_dir / "config.json"
        metadata_path = model_dir / "metadata.json"

        # -----------------------------------------------------
        # Check files
        # -----------------------------------------------------
        for path in (
            weights_path,
            config_path,
            metadata_path,
        ):
            if not path.is_file():
                raise ModelUnavailableError(
                    f"{model_name}: missing {path}"
                )

        # -----------------------------------------------------
        # Metadata
        # -----------------------------------------------------
        try:
            metadata = json.loads(
                metadata_path.read_text(
                    encoding="utf-8"
                )
            )
        except Exception as exc:
            raise ModelUnavailableError(
                f"Unable to read {metadata_path.name}: {exc}"
            ) from exc

        # -----------------------------------------------------
        # Config
        # -----------------------------------------------------
        try:
            config = json.loads(
                config_path.read_text(
                    encoding="utf-8"
                )
            )
        except Exception as exc:
            raise ModelUnavailableError(
                f"Unable to read {config_path.name}: {exc}"
            ) from exc

        # -----------------------------------------------------
        # Reconstruct model
        # -----------------------------------------------------
        model = self._model_from_config(
            config,
            model_dir.name,
        )

        # -----------------------------------------------------
        # Load weights
        # -----------------------------------------------------
        try:
            model.load_weights(weights_path)
        except Exception as exc:
            raise ModelUnavailableError(
                f"Unable to load weights for {model_name}: {exc}"
            ) from exc

        return model, metadata

    # =========================================================
    # RECONSTRUCT MODEL
    # =========================================================

    def _model_from_config(
        self,
        config: dict[str, Any],
        model_type: str,
    ) -> keras.Model:

        # -----------------------------------------------------
        # Prefer complete serialized Keras model
        # -----------------------------------------------------
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

        # -----------------------------------------------------
        # Fallback architecture
        # -----------------------------------------------------
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
            activation="relu",
        )(x)

        x = keras.layers.BatchNormalization()(x)

        x = keras.layers.Dropout(0.3)(x)

        # -----------------------------------------------------
        # m_model = binary wound detector
        # -----------------------------------------------------
        if model_type == "model":
            outputs = keras.layers.Dense(
                1,
                activation="sigmoid",
            )(x)

        # -----------------------------------------------------
        # model = six-class wound classifier
        # -----------------------------------------------------
        else:
            outputs = keras.layers.Dense(
                6,
                activation="softmax",
            )(x)

        return keras.Model(
            inputs,
            outputs,
            name=f"woundscope_{model_type}",
        )

    # =========================================================
    # PREDICTION
    # =========================================================

    def predict(
        self,
        image: np.ndarray,
    ) -> tuple[dict[str, Any], float]:

        if self.wound_model is None:
            raise ModelUnavailableError(
                "Wound detection model is unavailable."
            )

        if self.classification_model is None:
            raise ModelUnavailableError(
                "Wound classification model is unavailable."
            )

        started = time.perf_counter()

        # =====================================================
        # STAGE 1
        # WOUND / NOT WOUND
        # =====================================================

        wound_result = self.wound_model.predict(
            image,
            verbose=0,
        )

        wound_result = np.asarray(
            wound_result,
            dtype=np.float64,
        )

        print(
            f"Wound model output: {wound_result}"
        )

        if wound_result.shape != (1, 1):
            raise ModelUnavailableError(
                f"Unexpected wound model output shape: "
                f"{wound_result.shape}"
            )

        wound_probability = float(
            wound_result[0][0]
        )

        if not np.isfinite(wound_probability):
            raise ModelUnavailableError(
                "Wound model returned an invalid value."
            )

        # =====================================================
        # THRESHOLD
        # =====================================================

        is_wound = wound_probability >= 0.5

        # =====================================================
        # NOT A WOUND
        # =====================================================

        if not is_wound:
            elapsed = (
                time.perf_counter() - started
            ) * 1000

            return {
                "is_wound": False,
                "class": "Not a Wound",
                "class_index": -1,
                "confidence": float(
                    1.0 - wound_probability
                ),
                "probabilities": [
                    {
                        "class": name,
                        "class_index": index,
                        "probability": 0.0,
                    }
                    for index, name
                    in enumerate(self.class_names)
                ],
            }, elapsed

        # =====================================================
        # STAGE 2
        # SIX-CLASS WOUND CLASSIFICATION
        # =====================================================

        classification_result = (
            self.classification_model.predict(
                image,
                verbose=0,
            )
        )

        values = np.asarray(
            classification_result,
            dtype=np.float64,
        )

        print(
            f"Classification model output: {values}"
        )

        if values.shape != (1, 6):
            raise ModelUnavailableError(
                f"Unexpected classification model output shape: "
                f"{values.shape}"
            )

        probabilities = values[0]

        if not np.isfinite(probabilities).all():
            raise ModelUnavailableError(
                "Classification model returned invalid values."
            )

        if np.any(probabilities < 0):
            raise ModelUnavailableError(
                "Classification model returned negative probabilities."
            )

        # -----------------------------------------------------
        # Normalize if necessary
        # -----------------------------------------------------
        total = probabilities.sum()

        if total <= 0:
            raise ModelUnavailableError(
                "Classification model returned zero probabilities."
            )

        probabilities = probabilities / total

        winner = int(
            np.argmax(probabilities)
        )

        elapsed = (
            time.perf_counter() - started
        ) * 1000

        return {
            "is_wound": True,
            "class": self.class_names[winner],
            "class_index": winner,
            "confidence": float(
                probabilities[winner]
            ),
            "probabilities": [
                {
                    "class": name,
                    "class_index": index,
                    "probability": float(
                        probabilities[index]
                    ),
                }
                for index, name
                in enumerate(self.class_names)
            ],
        }, elapsed


model_service = WoundModelService()