import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from PIL import Image

from src.services.model_service import VisionLanguageService


def load_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Dataset must be a list of examples.")
    return data


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    values_sorted = sorted(values)
    k = (len(values_sorted) - 1) * pct
    f = int(k)
    c = min(f + 1, len(values_sorted) - 1)
    if f == c:
        return values_sorted[f]
    d0 = values_sorted[f] * (c - k)
    d1 = values_sorted[c] * (k - f)
    return d0 + d1


def summarize(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean_ms": None,
            "median_ms": None,
            "min_ms": None,
            "max_ms": None,
            "p95_ms": None,
        }
    return {
        "count": len(values),
        "mean_ms": round(statistics.mean(values), 2),
        "median_ms": round(statistics.median(values), 2),
        "min_ms": round(min(values), 2),
        "max_ms": round(max(values), 2),
        "p95_ms": round(percentile(values, 0.95) or 0.0, 2),
    }


def load_image(image_path: Path) -> Image.Image:
    if not image_path.exists():
        raise FileNotFoundError(f"Missing image: {image_path}")
    return Image.open(image_path).convert("RGB")


def time_caption(service: VisionLanguageService, image: Image.Image) -> tuple[float, str]:
    start = time.perf_counter()
    prediction = service.generate_caption(image)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return elapsed_ms, prediction


def time_vqa(service: VisionLanguageService, image: Image.Image, question: str) -> tuple[float, str]:
    start = time.perf_counter()
    prediction = service.answer_question(image, question)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return elapsed_ms, prediction


def time_analyze(
    service: VisionLanguageService,
    image: Image.Image,
    question: str,
) -> tuple[float, dict[str, str]]:
    start = time.perf_counter()
    caption = service.generate_caption(image)
    answer = service.answer_question(image, question)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return elapsed_ms, {"caption": caption, "answer": answer}


def warmup_model(service: VisionLanguageService, rows: list[dict[str, Any]], warmup_runs: int) -> None:
    if not rows or warmup_runs <= 0:
        return

    first_image_path = Path(rows[0]["image_path"])
    question = rows[0].get("question", "What is happening in this image?")
    image = load_image(first_image_path)

    for _ in range(warmup_runs):
        service.generate_caption(image)
        service.answer_question(image, question)


def plot_average_latency(summary: dict[str, dict[str, float | int | None]], output_dir: Path) -> None:
    tasks = []
    means = []

    for task, stats in summary.items():
        if stats["mean_ms"] is not None:
            tasks.append(task)
            means.append(stats["mean_ms"])

    plt.figure(figsize=(8, 5))
    plt.bar(tasks, means)
    plt.title("Average VLM Inference Latency by Task")
    plt.xlabel("Task")
    plt.ylabel("Mean latency (ms)")
    plt.tight_layout()
    plt.savefig(output_dir / "latency_mean_by_task.png", dpi=200)
    plt.close()


def plot_latency_histograms(latency_data: dict[str, list[float]], output_dir: Path) -> None:
    for task, values in latency_data.items():
        if not values:
            continue
        plt.figure(figsize=(8, 5))
        plt.hist(values, bins=min(20, max(5, len(values))))
        plt.title(f"Latency Distribution: {task}")
        plt.xlabel("Latency (ms)")
        plt.ylabel("Frequency")
        plt.tight_layout()
        safe_name = task.replace("/", "_")
        plt.savefig(output_dir / f"latency_hist_{safe_name}.png", dpi=200)
        plt.close()


def evaluate_latency(
    dataset_path: Path,
    repeats: int,
    warmup_runs: int,
) -> dict[str, Any]:
    rows = load_json(dataset_path)

    service = VisionLanguageService()
    service.load_models()

    warmup_model(service, rows, warmup_runs)

    latency_data: dict[str, list[float]] = {
        "caption": [],
        "vqa": [],
        "analyze": [],
    }
    detailed_results: list[dict[str, Any]] = []

    for row in rows:
        image_path = Path(row["image_path"])
        question = row.get("question", "What is happening in this image?")
        image = load_image(image_path)

        for run_idx in range(repeats):
            elapsed_ms, caption_prediction = time_caption(service, image)
            latency_data["caption"].append(elapsed_ms)
            detailed_results.append(
                {
                    "task": "caption",
                    "run": run_idx + 1,
                    "image_path": str(image_path),
                    "latency_ms": round(elapsed_ms, 2),
                    "prediction": caption_prediction,
                }
            )

            elapsed_ms, vqa_prediction = time_vqa(service, image, question)
            latency_data["vqa"].append(elapsed_ms)
            detailed_results.append(
                {
                    "task": "vqa",
                    "run": run_idx + 1,
                    "image_path": str(image_path),
                    "question": question,
                    "latency_ms": round(elapsed_ms, 2),
                    "prediction": vqa_prediction,
                }
            )

            elapsed_ms, analyze_prediction = time_analyze(service, image, question)
            latency_data["analyze"].append(elapsed_ms)
            detailed_results.append(
                {
                    "task": "analyze",
                    "run": run_idx + 1,
                    "image_path": str(image_path),
                    "question": question,
                    "latency_ms": round(elapsed_ms, 2),
                    "prediction": analyze_prediction,
                }
            )

    summary = {
        task: summarize(values)
        for task, values in latency_data.items()
    }

    return {
        "device": service.device,
        "caption_model": service.settings.caption_model_name,
        "vqa_model": service.settings.vqa_model_name,
        "repeats": repeats,
        "warmup_runs": warmup_runs,
        "summary": summary,
        "details": detailed_results,
    }, latency_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate direct VLM inference latency.")
    parser.add_argument("--dataset", type=Path, required=True, help="Path to latency evaluation JSON file.")
    parser.add_argument("--repeats", type=int, default=5, help="Number of timed runs per example.")
    parser.add_argument("--warmup-runs", type=int, default=2, help="Number of warmup runs before timing.")
    parser.add_argument("--output-json", type=Path, default=Path("data/latency_results.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/latency_plots"))
    args = parser.parse_args()

    ensure_output_dir(args.output_dir)
    results, latency_data = evaluate_latency(
        dataset_path=args.dataset,
        repeats=args.repeats,
        warmup_runs=args.warmup_runs,
    )

    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    plot_average_latency(results["summary"], args.output_dir)
    plot_latency_histograms(latency_data, args.output_dir)

    print(json.dumps(results["summary"], indent=2))
    print(f"Saved JSON results to {args.output_json}")
    print(f"Saved plots to {args.output_dir}")


if __name__ == "__main__":
    main()