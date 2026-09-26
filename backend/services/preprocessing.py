from io import BytesIO
import numpy as np
from PIL import Image, UnidentifiedImageError
from fastapi import HTTPException, UploadFile

ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png"}

async def decode_upload(upload: UploadFile) -> tuple[np.ndarray, int, int]:
    filename = upload.filename or ""
    if upload.content_type not in {"image/jpeg", "image/png"} and not filename.lower().endswith(tuple(ALLOWED_SUFFIXES)):
        raise HTTPException(415, "Invalid image. Please upload a JPG, JPEG, or PNG file.")
    payload = await upload.read()
    if not payload:
        raise HTTPException(400, "Invalid image. Please upload a JPG, JPEG, or PNG file.")
    try:
        with Image.open(BytesIO(payload)) as image:
            image.load()
            if image.format not in {"JPEG", "PNG"}:
                raise ValueError("unsupported image format")
            rgb = image.convert("RGB")
            original_width, original_height = rgb.size
            # image_dataset_from_directory uses bilinear interpolation by default;
            # matching it keeps API inputs aligned with notebook evaluation.
            resized = rgb.resize((224, 224), Image.Resampling.BILINEAR)
            array = np.asarray(resized, dtype=np.float32)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(400, "Invalid image. Please upload a JPG, JPEG, or PNG file.") from exc
    return np.expand_dims(array, axis=0), original_width, original_height
