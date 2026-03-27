import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import requests


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


def send_caption_request(base_url: str, image_path: Path) -> tuple[float, int, dict[str, Any] | None]:
    with image_path.open("rb") as f:
        files = {"image": (image_path.name, f, "application/octet-stream")}
        start = time.perf_counter()
        response = requests.post(f"{base_url}/caption", files=files, timeout=120)
        elapsed_ms = (time.perf_counter() - start) * 1000
    payload = response.json() if response.headers.get("content-type", "").startswith("application/json") else None
    return elapsed_ms, response.status_code, payload


def send_vqa_request(base_url: str, image_path: Path, question: str) -> tuple[float, int, dict[str, Any] | None]:
    with image_path.open("rb") as f:
        files = {"image": (image_path.name, f, "application/octet-stream")}
        data = {"question": question}
        start = time.perf_counter()
        response = requests.post(f"{base_url}/vqa", files=files, data=data, timeout=120)
        elapsed_ms = (time.perf_counter() - start) * 1000
    payload = response.json() if response.headers.get("content-type", "").startswith("application/json") else None
    return elapsed_ms, response.status_code, payload


def send_analyze_request(base_url: str, image_path: Path, question: str) -> tuple[float, int, dict[str, Any] | None]:
    with image_path.open("rb") as f:
        files = {"image": (image_path.name, f, "application/octet-stream")}
        data = {"question": question}
        start = time.perf_counter()
        response = requests.post(f"{base_url}/analyze", files=files, data=data, timeout=120)
        elapsed_ms = (time.perf_counter() - start) * 1000
    payload = response.json() if response.headers.get("content-type", "").startswith("application/json") else None
    return elapsed_ms, response.status_code, payload


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


def plot_average_latency(summary: dict[str, dict[str, float | int | None]], output_dir: Path) -> None:
    endpoints = []
    means = []

    for endpoint, stats in summary.items():
        if stats["mean_ms"] is not None:
            endpoints.append(endpoint)
            means.append(stats["mean_ms"])

    plt.figure(figsize=(8, 5))
    plt.bar(endpoints, means)
    plt.title("Average Latency by Endpoint")
    plt.xlabel("Endpoint")
    plt.ylabel("Mean latency (ms)")
    plt.tight_layout()
    plt.savefig(output_dir / "latency_mean_by_endpoint.png", dpi=200)
    plt.close()


def plot_latency_histograms(latency_data: dict[str, list[float]], output_dir: Path) -> None:
    for endpoint, values in latency_data.items():
        if not values:
            continue
        plt.figure(figsize=(8, 5))
        plt.hist(values, bins=min(20, max(5, len(values))))
        plt.title(f"Latency Distribution: {endpoint}")
        plt.xlabel("Latency (ms)")
        plt.ylabel("Frequency")
        plt.tight_layout()
        safe_name = endpoint.strip("/").replace("/", "_") or "root"
        plt.savefig(output_dir / f"latency_hist_{safe_name}.png", dpi=200)
        plt.close()


def evaluate_latency(
    dataset_path: Path,
    base_url: str,
    repeats: int,
) -> dict[str, Any]:
    rows = load_json(dataset_path)

    latency_data: dict[str, list[float]] = {
        "/caption": [],
        "/vqa": [],
        "/analyze": [],
    }
    detailed_results: list[dict[str, Any]] = []

    for row in rows:
        image_path = Path(row["image_path"])
        if not image_path.exists():
            raise FileNotFoundError(f"Missing image: {image_path}")

        question = row.get("question", "What is happening in this image?")

        for run_idx in range(repeats):
            elapsed_ms, status_code, payload = send_caption_request(base_url, image_path)
            latency_data["/caption"].append(elapsed_ms)
            detailed_results.append(
                {
                    "endpoint": "/caption",
                    "run": run_idx + 1,
                    "image_path": str(image_path),
                    "latency_ms": round(elapsed_ms, 2),
                    "status_code": status_code,
                    "response": payload,
                }
            )

            elapsed_ms, status_code, payload = send_vqa_request(base_url, image_path, question)
            latency_data["/vqa"].append(elapsed_ms)
            detailed_results.append(
                {
                    "endpoint": "/vqa",
                    "run": run_idx + 1,
                    "image_path": str(image_path),
                    "question": question,
                    "latency_ms": round(elapsed_ms, 2),
                    "status_code": status_code,
                    "response": payload,
                }
            )

            elapsed_ms, status_code, payload = send_analyze_request(base_url, image_path, question)
            latency_data["/analyze"].append(elapsed_ms)
            detailed_results.append(
                {
                    "endpoint": "/analyze",
                    "run": run_idx + 1,
                    "image_path": str(image_path),
                    "question": question,
                    "latency_ms": round(elapsed_ms, 2),
                    "status_code": status_code,
                    "response": payload,
                }
            )

    summary = {
        endpoint: summarize(values)
        for endpoint, values in latency_data.items()
    }

    return {
        "base_url": base_url,
        "repeats": repeats,
        "summary": summary,
        "details": detailed_results,
    }, latency_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate endpoint latency for the Vision Language API.")
    parser.add_argument("--dataset", type=Path, required=True, help="Path to latency evaluation JSON file.")
    parser.add_argument("--base-url", type=str, default="http://127.0.0.1:8000", help="API base URL.")
    parser.add_argument("--repeats", type=int, default=3, help="Number of runs per example.")
    parser.add_argument("--output-json", type=Path, default=Path("data/latency_results.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/latency_plots"))
    args = parser.parse_args()

    ensure_output_dir(args.output_dir)
    results, latency_data = evaluate_latency(
        dataset_path=args.dataset,
        base_url=args.base_url.rstrip("/"),
        repeats=args.repeats,
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