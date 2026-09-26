"""Train a reliable WoundScope DenseNet169 classifier.

Expected dataset layout (the class names are read from folder names, in sorted order):

dataset_root/
    train/Abrasions/*.jpg      val/Abrasions/*.jpg      test/Abrasions/*.jpg
    train/Bruises/*.jpg        val/Bruises/*.jpg        test/Bruises/*.jpg
    ...
    train/Other/*.jpg          val/Other/*.jpg          test/Other/*.jpg

The `Other` folder is optional for six-class experiments. Add it to every split
to produce a seven-class model that can learn non-wound examples.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


IMAGE_SIZE = (224, 224)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DenseNet169 wound classifier")
    parser.add_argument("--data-root", type=Path, required=True, help="Folder containing train/, val/, and test/")
    parser.add_argument("--output-dir", type=Path, default=Path("training_output"), help="Where to save the trained artifacts")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--head-epochs", type=int, default=30, help="Maximum frozen-backbone epochs; early stopping may end earlier")
    parser.add_argument("--fine-tune-epochs", type=int, default=20, help="Maximum partial fine-tuning epochs")
    parser.add_argument("--unfreeze-layers", type=int, default=40, help="Number of final DenseNet layers to fine-tune")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_reproducibility(seed: int) -> None:
    """Make repeated runs more comparable. GPU operations can still vary by platform."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def image_count(class_dir: Path) -> int:
    return sum(item.suffix.lower() in IMAGE_EXTENSIONS for item in class_dir.iterdir() if item.is_file())


def inspect_splits(data_root: Path) -> list[str]:
    """Reject silent class-order or missing-folder mistakes before training."""
    split_names = ("train", "val", "test")
    folders: dict[str, list[str]] = {}
    for split in split_names:
        split_path = data_root / split
        if not split_path.is_dir():
            raise FileNotFoundError(f"Missing split folder: {split_path}")
        folders[split] = sorted(item.name for item in split_path.iterdir() if item.is_dir())
        if not folders[split]:
            raise ValueError(f"No class folders in {split_path}")
    if folders["train"] != folders["val"] or folders["train"] != folders["test"]:
        raise ValueError(f"Class folders must match across splits. Found: {folders}")
    for split in split_names:
        counts = {name: image_count(data_root / split / name) for name in folders[split]}
        print(f"{split:5} images: {counts}")
        empty = [name for name, count in counts.items() if count == 0]
        if empty:
            raise ValueError(f"Empty classes in {split}: {empty}")
    return folders["train"]


def load_dataset(directory: Path, batch_size: int, shuffle: bool, seed: int) -> tf.data.Dataset:
    """This uses TensorFlow's bilinear resize, matching evaluation and backend expectations."""
    return keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="int",
        image_size=IMAGE_SIZE,
        interpolation="bilinear",
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed if shuffle else None,
    ).prefetch(tf.data.AUTOTUNE)


def make_model(class_count: int) -> tuple[keras.Model, keras.Model]:
    """Create the original DenseNet169 architecture with a class-count-aware head.

    Random augmentation is in the saved model so Keras automatically disables it
    during `model.predict`; an inference service must pass raw RGB pixels once.
    """
    augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ], name="data_augmentation")
    base_model = keras.applications.DenseNet169(
        include_top=False,
        weights="imagenet",
        input_shape=(*IMAGE_SIZE, 3),
    )
    base_model.trainable = False

    inputs = keras.Input(shape=(*IMAGE_SIZE, 3), name="image")
    x = augmentation(inputs)
    # DenseNet's ImageNet normalization is embedded exactly once in the model.
    x = keras.applications.densenet.preprocess_input(x)
    # Keep BatchNorm in inference mode while fine-tuning; this is safer with small batches.
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(128, activation="relu", name="dense_128")(x)
    x = layers.BatchNormalization(name="batch_norm")(x)
    x = layers.Dropout(0.3, name="dropout")(x)
    outputs = layers.Dense(class_count, activation="softmax", name="classification")(x)
    return keras.Model(inputs, outputs, name="woundscope_densenet169"), base_model


def callbacks(output_dir: Path, stage: str) -> list[keras.callbacks.Callback]:
    """Save/restore the best validation model instead of retaining the final epoch."""
    return [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True, verbose=1),
        keras.callbacks.ModelCheckpoint(output_dir / f"best_{stage}.keras", monitor="val_loss", save_best_only=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3, min_lr=1e-7, verbose=1),
    ]


def class_weights(train_dir: Path, class_names: list[str]) -> dict[int, float]:
    """Reduce domination by large classes; inspect per-class metrics before relying on this."""
    counts = np.array([image_count(train_dir / name) for name in class_names], dtype=np.float32)
    weights = counts.sum() / (len(counts) * counts)
    return {index: float(weight) for index, weight in enumerate(weights)}


def evaluate(model: keras.Model, dataset: tf.data.Dataset, class_names: list[str]) -> dict[str, object]:
    """Return JSON-friendly evaluation data and print a per-class report."""
    y_true: list[int] = []
    probabilities: list[np.ndarray] = []
    for images, labels in dataset:
        probabilities.append(model.predict(images, verbose=0))
        y_true.extend(labels.numpy().tolist())
    y_prob = np.concatenate(probabilities)
    y_pred = y_prob.argmax(axis=1)
    matrix = tf.math.confusion_matrix(y_true, y_pred, num_classes=len(class_names)).numpy().tolist()
    accuracy = float(np.mean(np.asarray(y_true) == y_pred))
    per_class = {}
    for index, name in enumerate(class_names):
        mask = np.asarray(y_true) == index
        per_class[name] = {"support": int(mask.sum()), "recall": float(np.mean(y_pred[mask] == index)) if mask.any() else None}
    print("Test accuracy:", f"{accuracy:.4f}")
    print("Confusion matrix:", matrix)
    return {"accuracy": accuracy, "confusion_matrix": matrix, "per_class": per_class}


def main() -> None:
    args = parse_args()
    set_reproducibility(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    class_names = inspect_splits(args.data_root)
    print("Class order:", class_names)

    train_data = load_dataset(args.data_root / "train", args.batch_size, True, args.seed)
    val_data = load_dataset(args.data_root / "val", args.batch_size, False, args.seed)
    test_data = load_dataset(args.data_root / "test", args.batch_size, False, args.seed)
    if train_data.class_names != class_names:
        raise RuntimeError("Unexpected TensorFlow class ordering")

    model, base_model = make_model(len(class_names))
    sample_weights = class_weights(args.data_root / "train", class_names)
    print("Class weights:", sample_weights)

    # Stage 1 learns the classifier head without altering pretrained DenseNet features.
    model.compile(optimizer=keras.optimizers.Adam(1e-4), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_data, validation_data=val_data, epochs=args.head_epochs, callbacks=callbacks(args.output_dir, "head"), class_weight=sample_weights)

    # Stage 2 carefully adapts only the final backbone layers at a lower learning rate.
    base_model.trainable = True
    for layer in base_model.layers[:-args.unfreeze_layers]:
        layer.trainable = False
    model.compile(optimizer=keras.optimizers.Adam(1e-5), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_data, validation_data=val_data, epochs=args.fine_tune_epochs, callbacks=callbacks(args.output_dir, "fine_tune"), class_weight=sample_weights)

    metrics = evaluate(model, test_data, class_names)
    # These three files are the private server-side artifacts used by the FastAPI backend.
    model.save_weights(args.output_dir / "model.weights.h5")
    (args.output_dir / "config.json").write_text(model.to_json(), encoding="utf-8")
    metadata = {"model": "DenseNet169", "class_names": class_names, "input_shape": [224, 224, 3], "classes": len(class_names), "date_saved": datetime.now(timezone.utc).isoformat(), "test_metrics": metrics}
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved model artifacts to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
