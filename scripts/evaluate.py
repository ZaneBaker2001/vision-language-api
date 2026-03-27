import argparse
import json
import re
from pathlib import Path
from typing import Any

from PIL import Image

from services.model_service import VisionLanguageService


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def load_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Evaluation file must contain a list of examples.")
    return data


def evaluate(dataset_path: Path) -> dict[str, Any]:
    service = VisionLanguageService()
    service.load_models()

    rows = load_json(dataset_path)

    caption_total = 0
    caption_match = 0

    vqa_total = 0
    vqa_match = 0

    detailed_results = []

    for row in rows:
        image_path = Path(row["image_path"])
        task = row["task"]

        if not image_path.exists():
            raise FileNotFoundError(f"Missing image: {image_path}")

        image = Image.open(image_path).convert("RGB")

        if task == "caption":
            reference = row["reference"]
            prediction = service.generate_caption(image)

            is_match = normalize_text(reference) == normalize_text(prediction)
            caption_total += 1
            caption_match += int(is_match)

            detailed_results.append(
                {
                    "task": "caption",
                    "image_path": str(image_path),
                    "reference": reference,
                    "prediction": prediction,
                    "match": is_match,
                }
            )

        elif task == "vqa":
            question = row["question"]
            reference = row["reference"]
            prediction = service.answer_question(image, question)

            is_match = normalize_text(reference) == normalize_text(prediction)
            vqa_total += 1
            vqa_match += int(is_match)

            detailed_results.append(
                {
                    "task": "vqa",
                    "image_path": str(image_path),
                    "question": question,
                    "reference": reference,
                    "prediction": prediction,
                    "match": is_match,
                }
            )

        else:
            raise ValueError(f"Unsupported task: {task}")

    results = {
        "caption_examples": caption_total,
        "caption_exact_match": (caption_match / caption_total) if caption_total else None,
        "vqa_examples": vqa_total,
        "vqa_accuracy": (vqa_match / vqa_total) if vqa_total else None,
        "details": detailed_results,
    }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the Vision Language API models.")
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to evaluation JSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save evaluation results JSON.",
    )
    args = parser.parse_args()

    results = evaluate(args.dataset)

    print(json.dumps(results, indent=2))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()