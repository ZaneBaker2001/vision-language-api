from io import BytesIO

from fastapi import HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError


def read_upload_image(upload: UploadFile) -> Image.Image:
    try:
        raw = upload.file.read()
        image = Image.open(BytesIO(raw)).convert("RGB")
        return image
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Invalid image file.") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to process uploaded image.") from exc
    finally:
        upload.file.close()