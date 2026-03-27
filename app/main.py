from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from src.config import get_settings
from src.logging_config import configure_logging, get_logger
from src.schemas import AnalyzeResponse, CaptionResponse, HealthResponse, VQAResponse
from src.services.model_service import VisionLanguageService
from src.utils import read_upload_image

configure_logging()
logger = get_logger(__name__)
settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    description="Image captioning and visual question answering API using BLIP.",
    version=settings.app_version,
)

vlm_service = VisionLanguageService()


@app.on_event("startup")
def startup_event() -> None:
    logger.info("Starting application")
    try:
        vlm_service.load_models()
        logger.info("Models loaded successfully")
    except Exception as exc:
        logger.exception("Model loading failed: %s", exc)


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return HealthResponse(
        status="ok",
        app_name=settings.app_name,
        version=settings.app_version,
        device=vlm_service.device,
        caption_model_loaded=vlm_service.caption_model is not None,
        vqa_model_loaded=vlm_service.vqa_model is not None,
    )


@app.post("/caption", response_model=CaptionResponse)
async def generate_caption(image: UploadFile = File(...)) -> CaptionResponse:
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    pil_image = read_upload_image(image)
    caption = vlm_service.generate_caption(pil_image)

    return CaptionResponse(filename=image.filename or "unknown", caption=caption)


@app.post("/vqa", response_model=VQAResponse)
async def answer_visual_question(
    image: UploadFile = File(...),
    question: str = Form(...),
) -> VQAResponse:
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")
    if not question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    pil_image = read_upload_image(image)
    answer = vlm_service.answer_question(pil_image, question.strip())

    return VQAResponse(
        filename=image.filename or "unknown",
        question=question.strip(),
        answer=answer,
    )


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_image(
    image: UploadFile = File(...),
    question: str = Form("What is happening in this image?"),
) -> AnalyzeResponse:
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    pil_image = read_upload_image(image)
    caption = vlm_service.generate_caption(pil_image)
    answer = vlm_service.answer_question(pil_image, question.strip())

    return AnalyzeResponse(
        filename=image.filename or "unknown",
        caption=caption,
        question=question.strip(),
        answer=answer,
    )