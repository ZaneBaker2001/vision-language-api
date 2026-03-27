from __future__ import annotations

from typing import Optional

import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipForQuestionAnswering, BlipProcessor

from src.config import get_settings
from src.exceptions import ModelNotLoadedError
from src.logging_config import get_logger

logger = get_logger(__name__)


class VisionLanguageService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.caption_processor: Optional[BlipProcessor] = None
        self.caption_model: Optional[BlipForConditionalGeneration] = None

        self.vqa_processor: Optional[BlipProcessor] = None
        self.vqa_model: Optional[BlipForQuestionAnswering] = None

    def load_models(self) -> None:
        if self.caption_model is None:
            logger.info("Loading caption model: %s", self.settings.caption_model_name)
            self.caption_processor = BlipProcessor.from_pretrained(
                self.settings.caption_model_name
            )
            self.caption_model = BlipForConditionalGeneration.from_pretrained(
                self.settings.caption_model_name
            ).to(self.device)

        if self.vqa_model is None:
            logger.info("Loading VQA model: %s", self.settings.vqa_model_name)
            self.vqa_processor = BlipProcessor.from_pretrained(
                self.settings.vqa_model_name
            )
            self.vqa_model = BlipForQuestionAnswering.from_pretrained(
                self.settings.vqa_model_name
            ).to(self.device)

    def _ensure_loaded(self) -> None:
        if self.caption_model is None or self.vqa_model is None:
            self.load_models()
        if (
            self.caption_model is None
            or self.caption_processor is None
            or self.vqa_model is None
            or self.vqa_processor is None
        ):
            raise ModelNotLoadedError("Models are not available.")

    @torch.inference_mode()
    def generate_caption(self, image: Image.Image) -> str:
        self._ensure_loaded()
        assert self.caption_processor is not None
        assert self.caption_model is not None

        inputs = self.caption_processor(images=image, return_tensors="pt").to(self.device)
        output = self.caption_model.generate(
            **inputs,
            max_new_tokens=self.settings.max_caption_tokens,
            num_beams=self.settings.caption_num_beams,
        )
        caption = self.caption_processor.decode(output[0], skip_special_tokens=True)
        return caption.strip()

    @torch.inference_mode()
    def answer_question(self, image: Image.Image, question: str) -> str:
        self._ensure_loaded()
        assert self.vqa_processor is not None
        assert self.vqa_model is not None

        inputs = self.vqa_processor(
            images=image,
            text=question,
            return_tensors="pt",
        ).to(self.device)

        output = self.vqa_model.generate(
            **inputs,
            max_new_tokens=self.settings.max_answer_tokens,
        )
        answer = self.vqa_processor.decode(output[0], skip_special_tokens=True)
        return answer.strip()