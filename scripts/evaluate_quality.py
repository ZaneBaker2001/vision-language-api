import argparse
import json
import math
import re
import statistics
import string
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
from PIL import Image

from src.services.model_service import VisionLanguageService


ARTICLES = {"a", "an", "the"}
PUNCT_TRANSLATION = str.maketrans("", "", string.punctuation)


def load_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Dataset must be a list of examples.")
    return data


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_image(image_path: Path) -> Image.Image:
    if not image_path.exists():
        raise FileNotFoundError(f"Missing image: {image_path}")
    return Image.open(image_path).convert("RGB")


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = text.translate(PUNCT_TRANSLATION)
    text = re.sub(r"\s+", " ", text)
    return text


def normalized_tokens(text: str, *, drop_articles: bool = False) -> list[str]:
    tokens = [token for token in normalize_text(text).split() if token]
    if drop_articles:
        tokens = [token for token in tokens if token not in ARTICLES]
    return tokens


def ensure_list(value: str | Iterable[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [item for item in value if isinstance(item, str) and item.strip()]


def strict_exact_match(prediction: str, references: list[str]) -> float:
    return float(any(prediction.strip() == reference.strip() for reference in references))


def top1_accuracy(prediction: str, references: list[str]) -> float:
    normalized_prediction = normalize_text(prediction)
    return float(any(normalized_prediction == normalize_text(reference) for reference in references))


def _ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    if len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def bleu_4(prediction: str, references: list[str]) -> float:
    pred_tokens = normalized_tokens(prediction)
    ref_tokens_list = [normalized_tokens(reference) for reference in references if reference.strip()]

    if not pred_tokens or not ref_tokens_list:
        return 0.0

    precisions: list[float] = []
    for n in range(1, 5):
        pred_ngrams = Counter(_ngrams(pred_tokens, n))
        if not pred_ngrams:
            precisions.append(0.0)
            continue

        max_ref_counts: Counter[tuple[str, ...]] = Counter()
        for ref_tokens in ref_tokens_list:
            ref_counts = Counter(_ngrams(ref_tokens, n))
            for gram, count in ref_counts.items():
                if count > max_ref_counts[gram]:
                    max_ref_counts[gram] = count

        clipped = sum(min(count, max_ref_counts[gram]) for gram, count in pred_ngrams.items())
        total = sum(pred_ngrams.values())
        precisions.append((clipped + 1.0) / (total + 1.0))

    pred_len = len(pred_tokens)
    ref_lens = [len(tokens) for tokens in ref_tokens_list]
    closest_ref_len = min(ref_lens, key=lambda ref_len: (abs(ref_len - pred_len), ref_len))

    if pred_len == 0:
        brevity_penalty = 0.0
    elif pred_len > closest_ref_len:
        brevity_penalty = 1.0
    else:
        brevity_penalty = math.exp(1 - (closest_ref_len / pred_len))

    score = brevity_penalty * math.exp(sum(0.25 * math.log(max(p, 1e-12)) for p in precisions))
    return max(0.0, min(1.0, score))


def meteor(prediction: str, references: list[str], alpha: float = 0.9, beta: float = 3.0, gamma: float = 0.5) -> float:
    pred_tokens = normalized_tokens(prediction)
    if not pred_tokens or not references:
        return 0.0

    best_score = 0.0
    for reference in references:
        ref_tokens = normalized_tokens(reference)
        if not ref_tokens:
            continue

        ref_positions: defaultdict[str, list[int]] = defaultdict(list)
        for idx, token in enumerate(ref_tokens):
            ref_positions[token].append(idx)

        matches: list[tuple[int, int]] = []
        used_ref_indices: set[int] = set()
        last_ref_idx = -1

        for pred_idx, token in enumerate(pred_tokens):
            candidates = [i for i in ref_positions.get(token, []) if i not in used_ref_indices and i > last_ref_idx]
            if not candidates:
                candidates = [i for i in ref_positions.get(token, []) if i not in used_ref_indices]
            if not candidates:
                continue
            ref_idx = candidates[0]
            used_ref_indices.add(ref_idx)
            last_ref_idx = ref_idx
            matches.append((pred_idx, ref_idx))

        m = len(matches)
        if m == 0:
            continue

        precision = m / len(pred_tokens)
        recall = m / len(ref_tokens)
        f_mean = (precision * recall) / (alpha * precision + (1 - alpha) * recall)

        chunks = 1
        for i in range(1, len(matches)):
            prev_pred, prev_ref = matches[i - 1]
            curr_pred, curr_ref = matches[i]
            if curr_pred != prev_pred + 1 or curr_ref != prev_ref + 1:
                chunks += 1

        penalty = gamma * ((chunks / m) ** beta)
        score = f_mean * (1 - penalty)
        best_score = max(best_score, score)

    return max(0.0, min(1.0, best_score))


def cider(prediction: str, references: list[str], max_n: int = 4, sigma: float = 6.0) -> float:
    pred_tokens = normalized_tokens(prediction)
    ref_tokens_list = [normalized_tokens(reference) for reference in references if reference.strip()]

    if not pred_tokens or not ref_tokens_list:
        return 0.0

    num_refs = len(ref_tokens_list)
    document_frequency: dict[tuple[int, tuple[str, ...]], int] = defaultdict(int)
    for ref_tokens in ref_tokens_list:
        seen: set[tuple[int, tuple[str, ...]]] = set()
        for n in range(1, max_n + 1):
            for gram in set(_ngrams(ref_tokens, n)):
                key = (n, gram)
                if key not in seen:
                    document_frequency[key] += 1
                    seen.add(key)

    def tfidf_vector(tokens: list[str], n: int) -> dict[tuple[str, ...], float]:
        grams = _ngrams(tokens, n)
        counts = Counter(grams)
        total = sum(counts.values())
        if total == 0:
            return {}
        vector: dict[tuple[str, ...], float] = {}
        for gram, count in counts.items():
            df = document_frequency.get((n, gram), 0)
            idf = math.log((num_refs + 1.0) / (df + 1.0))
            vector[gram] = (count / total) * idf
        return vector

    def cosine_similarity(vec_a: dict[tuple[str, ...], float], vec_b: dict[tuple[str, ...], float]) -> float:
        if not vec_a or not vec_b:
            return 0.0
        dot = sum(value * vec_b.get(key, 0.0) for key, value in vec_a.items())
        norm_a = math.sqrt(sum(value * value for value in vec_a.values()))
        norm_b = math.sqrt(sum(value * value for value in vec_b.values()))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)

    pred_len = len(pred_tokens)
    scores: list[float] = []
    for ref_tokens in ref_tokens_list:
        ref_len = len(ref_tokens)
        length_penalty = math.exp(-((pred_len - ref_len) ** 2) / (2 * sigma * sigma))
        n_scores = []
        for n in range(1, max_n + 1):
            pred_vec = tfidf_vector(pred_tokens, n)
            ref_vec = tfidf_vector(ref_tokens, n)
            n_scores.append(cosine_similarity(pred_vec, ref_vec) * length_penalty)
        scores.append(sum(n_scores) / max_n)

    return max(0.0, 10.0 * sum(scores) / len(scores))


def summarize_metric(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
        }

    return {
        "count": len(values),
        "mean": round(statistics.mean(values), 4),
        "median": round(statistics.median(values), 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
    }


def plot_summary_metrics(results: dict[str, Any], output_dir: Path) -> None:
    caption_metrics = results["summary"]["caption"]
    vqa_metrics = results["summary"]["vqa"]

    caption_names = []
    caption_values = []
    for name, stats in caption_metrics.items():
        if stats["mean"] is not None:
            caption_names.append(name)
            caption_values.append(stats["mean"])

    if caption_names:
        plt.figure(figsize=(8, 5))
        plt.bar(caption_names, caption_values)
        plt.title("Image Captioning Metrics (Mean)")
        plt.xlabel("Metric")
        plt.ylabel("Score")
        plt.xticks(rotation=15)
        plt.tight_layout()
        plt.savefig(output_dir / "caption_metrics_summary.png", dpi=200)
        plt.close()

    vqa_names = []
    vqa_values = []
    for name, stats in vqa_metrics.items():
        if stats["mean"] is not None:
            vqa_names.append(name)
            vqa_values.append(stats["mean"])

    if vqa_names:
        plt.figure(figsize=(8, 5))
        plt.bar(vqa_names, vqa_values)
        plt.title("VQA Metrics (Mean)")
        plt.xlabel("Metric")
        plt.ylabel("Score")
        plt.xticks(rotation=15)
        plt.tight_layout()
        plt.savefig(output_dir / "vqa_metrics_summary.png", dpi=200)
        plt.close()


def plot_per_example_metrics(details: list[dict[str, Any]], output_dir: Path) -> None:
    caption_example_ids: list[str] = []
    caption_bleu: list[float] = []
    caption_meteor_scores: list[float] = []
    caption_cider_scores: list[float] = []

    vqa_example_ids: list[str] = []
    vqa_top1: list[float] = []
    vqa_exact: list[float] = []

    for detail in details:
        example_id = str(detail.get("example_id", ""))
        if "caption_metrics" in detail:
            caption_example_ids.append(example_id)
            caption_bleu.append(detail["caption_metrics"]["bleu_4"])
            caption_meteor_scores.append(detail["caption_metrics"]["meteor"])
            caption_cider_scores.append(detail["caption_metrics"]["cider"])

        if "vqa_metrics" in detail:
            vqa_example_ids.append(example_id)
            vqa_top1.append(detail["vqa_metrics"]["top_1_accuracy"])
            vqa_exact.append(detail["vqa_metrics"]["exact_match"])

    if caption_example_ids:
        x = list(range(len(caption_example_ids)))
        width = 0.25
        plt.figure(figsize=(10, 5))
        plt.bar([i - width for i in x], caption_bleu, width=width, label="BLEU-4")
        plt.bar(x, caption_meteor_scores, width=width, label="METEOR")
        plt.bar([i + width for i in x], caption_cider_scores, width=width, label="CIDEr")
        plt.title("Per-Example Captioning Metrics")
        plt.xlabel("Example ID")
        plt.ylabel("Score")
        plt.xticks(x, caption_example_ids)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "caption_metrics_per_example.png", dpi=200)
        plt.close()

    if vqa_example_ids:
        x = list(range(len(vqa_example_ids)))
        width = 0.35
        plt.figure(figsize=(10, 5))
        plt.bar([i - width / 2 for i in x], vqa_top1, width=width, label="Top-1 Accuracy")
        plt.bar([i + width / 2 for i in x], vqa_exact, width=width, label="Exact Match")
        plt.title("Per-Example VQA Metrics")
        plt.xlabel("Example ID")
        plt.ylabel("Score")
        plt.xticks(x, vqa_example_ids)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "vqa_metrics_per_example.png", dpi=200)
        plt.close()


def evaluate_quality(dataset_path: Path) -> dict[str, Any]:
    rows = load_json(dataset_path)

    service = VisionLanguageService()
    service.load_models()

    caption_bleu_scores: list[float] = []
    caption_meteor_scores: list[float] = []
    caption_cider_scores: list[float] = []
    vqa_top1_scores: list[float] = []
    vqa_exact_scores: list[float] = []
    details: list[dict[str, Any]] = []

    for idx, row in enumerate(rows, start=1):
        image_path = Path(row["image_path"])
        image = load_image(image_path)

        caption_prediction = service.generate_caption(image)
        example_result: dict[str, Any] = {
            "example_id": row.get("id", idx),
            "image_path": str(image_path),
            "caption_prediction": caption_prediction,
        }

        caption_references = ensure_list(row.get("caption_references"))
        if caption_references:
            caption_bleu = bleu_4(caption_prediction, caption_references)
            caption_meteor_score = meteor(caption_prediction, caption_references)
            caption_cider_score = cider(caption_prediction, caption_references)

            caption_bleu_scores.append(caption_bleu)
            caption_meteor_scores.append(caption_meteor_score)
            caption_cider_scores.append(caption_cider_score)

            example_result["caption_references"] = caption_references
            example_result["caption_metrics"] = {
                "bleu_4": round(caption_bleu, 4),
                "meteor": round(caption_meteor_score, 4),
                "cider": round(caption_cider_score, 4),
            }

        question = row.get("question")
        answer_references = ensure_list(row.get("answer_references") or row.get("answer_reference"))
        if question and answer_references:
            answer_prediction = service.answer_question(image, question)
            answer_top1 = top1_accuracy(answer_prediction, answer_references)
            answer_exact = strict_exact_match(answer_prediction, answer_references)

            vqa_top1_scores.append(answer_top1)
            vqa_exact_scores.append(answer_exact)

            example_result["question"] = question
            example_result["answer_references"] = answer_references
            example_result["answer_prediction"] = answer_prediction
            example_result["vqa_metrics"] = {
                "top_1_accuracy": round(answer_top1, 4),
                "exact_match": round(answer_exact, 4),
            }

        details.append(example_result)

    return {
        "device": service.device,
        "caption_model": service.settings.caption_model_name,
        "vqa_model": service.settings.vqa_model_name,
        "summary": {
            "caption": {
                "bleu_4": summarize_metric(caption_bleu_scores),
                "meteor": summarize_metric(caption_meteor_scores),
                "cider": summarize_metric(caption_cider_scores),
            },
            "vqa": {
                "top_1_accuracy": summarize_metric(vqa_top1_scores),
                "exact_match": summarize_metric(vqa_exact_scores),
            },
        },
        "details": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate caption and VQA output quality for the configured VLMs."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to quality evaluation JSON file.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("data/quality_results.json"),
        help="Where to save the evaluation results JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/quality_plots"),
        help="Where to save quality evaluation plots.",
    )
    args = parser.parse_args()

    results = evaluate_quality(args.dataset)
    ensure_parent_dir(args.output_json)
    ensure_output_dir(args.output_dir)

    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    plot_summary_metrics(results, args.output_dir)
    plot_per_example_metrics(results["details"], args.output_dir)

    print(json.dumps(results["summary"], indent=2))
    print(f"Saved JSON results to {args.output_json}")
    print(f"Saved plots to {args.output_dir}")


if __name__ == "__main__":
    main()