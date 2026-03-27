from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    app_name: str
    version: str
    device: str
    caption_model_loaded: bool
    vqa_model_loaded: bool


class CaptionResponse(BaseModel):
    filename: str
    caption: str


class VQAResponse(BaseModel):
    filename: str
    question: str
    answer: str


class AnalyzeResponse(BaseModel):
    filename: str
    caption: str
    question: str
    answer: str