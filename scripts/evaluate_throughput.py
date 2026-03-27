import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path
from typing import Any

import httpx
import matplotlib.pyplot as plt


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


def summarize_latencies(latencies_ms: list[float]) -> dict[str, float | int | None]:
    if not latencies_ms:
        return {
            "count": 0,
            "mean_ms": None,
            "median_ms": None,
            "min_ms": None,
            "max_ms": None,
            "p95_ms": None,
            "p99_ms": None,
        }

    return {
        "count": len(latencies_ms),
        "mean_ms": round(statistics.mean(latencies_ms), 2),
        "median_ms": round(statistics.median(latencies_ms), 2),
        "min_ms": round(min(latencies_ms), 2),
        "max_ms": round(max(latencies_ms), 2),
        "p95_ms": round(percentile(latencies_ms, 0.95) or 0.0, 2),
        "p99_ms": round(percentile(latencies_ms, 0.99) or 0.0, 2),
    }


def build_request_spec(base_url: str, row: dict[str, Any]) -> dict[str, Any]:
    endpoint = row.get("endpoint", "analyze").strip("/").lower()
    image_path = Path(row["image_path"])
    if not image_path.exists():
        raise FileNotFoundError(f"Missing image: {image_path}")

    image_bytes = image_path.read_bytes()
    filename = image_path.name
    files = {
        "image": (filename, image_bytes, "image/jpeg"),
    }

    if endpoint == "caption":
        data: dict[str, str] = {}
    elif endpoint in {"vqa", "analyze"}:
        data = {"question": row.get("question", "What is happening in this image?")}
    else:
        raise ValueError(f"Unsupported endpoint: {endpoint}")

    return {
        "endpoint": endpoint,
        "url": f"{base_url.rstrip('/')}/{endpoint}",
        "files": files,
        "data": data,
        "image_path": str(image_path),
        "question": data.get("question"),
    }


async def send_one_request(
    client: httpx.AsyncClient,
    request_spec: dict[str, Any],
    request_index: int,
) -> dict[str, Any]:
    start = time.perf_counter()
    status_code: int | None = None
    error_message: str | None = None

    try:
        response = await client.post(
            request_spec["url"],
            files=request_spec["files"],
            data=request_spec["data"],
        )
        status_code = response.status_code
        response.raise_for_status()
    except Exception as exc:
        error_message = str(exc)

    elapsed_ms = (time.perf_counter() - start) * 1000.0
    ok = status_code == 200 and error_message is None

    return {
        "request_index": request_index,
        "endpoint": request_spec["endpoint"],
        "image_path": request_spec["image_path"],
        "question": request_spec["question"],
        "status_code": status_code,
        "ok": ok,
        "latency_ms": round(elapsed_ms, 2),
        "error": error_message,
    }


async def run_load_stage(
    client: httpx.AsyncClient,
    request_specs: list[dict[str, Any]],
    concurrency: int,
    total_requests: int,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    queue: asyncio.Queue[int] = asyncio.Queue()

    for idx in range(total_requests):
        queue.put_nowait(idx)

    async def worker(worker_id: int) -> None:
        while True:
            try:
                request_index = queue.get_nowait()
            except asyncio.QueueEmpty:
                break

            request_spec = request_specs[request_index % len(request_specs)]
            result = await send_one_request(client, request_spec, request_index + 1)
            result["worker_id"] = worker_id
            results.append(result)
            queue.task_done()

    start = time.perf_counter()
    workers = [asyncio.create_task(worker(i + 1)) for i in range(concurrency)]
    await asyncio.gather(*workers)
    duration_s = time.perf_counter() - start

    total_completed = len(results)
    success_count = sum(1 for item in results if item["ok"])
    server_error_count = sum(
        1 for item in results if item["status_code"] is not None and item["status_code"] >= 500
    )
    non_200_count = sum(1 for item in results if item["status_code"] != 200)
    latencies_ms = [item["latency_ms"] for item in results]

    by_endpoint: dict[str, dict[str, Any]] = {}
    for endpoint in sorted({item["endpoint"] for item in results}):
        endpoint_results = [item for item in results if item["endpoint"] == endpoint]
        endpoint_latencies = [item["latency_ms"] for item in endpoint_results]
        endpoint_success = sum(1 for item in endpoint_results if item["ok"])
        endpoint_server_errors = sum(
            1 for item in endpoint_results if item["status_code"] is not None and item["status_code"] >= 500
        )
        by_endpoint[endpoint] = {
            "requests": len(endpoint_results),
            "success_rate": round(endpoint_success / len(endpoint_results), 4) if endpoint_results else None,
            "error_rate": round(1.0 - (endpoint_success / len(endpoint_results)), 4) if endpoint_results else None,
            "server_error_rate": round(endpoint_server_errors / len(endpoint_results), 4) if endpoint_results else None,
            "latency": summarize_latencies(endpoint_latencies),
        }

    return {
        "concurrency": concurrency,
        "total_requests": total_requests,
        "completed_requests": total_completed,
        "duration_s": round(duration_s, 3),
        "rps": round(total_completed / duration_s, 2) if duration_s > 0 else None,
        "success_rate": round(success_count / total_completed, 4) if total_completed else None,
        "error_rate": round(1.0 - (success_count / total_completed), 4) if total_completed else None,
        "server_error_rate": round(server_error_count / total_completed, 4) if total_completed else None,
        "non_200_rate": round(non_200_count / total_completed, 4) if total_completed else None,
        "latency": summarize_latencies(latencies_ms),
        "by_endpoint": by_endpoint,
        "details": sorted(results, key=lambda item: item["request_index"]),
    }


def choose_max_sustainable_stage(stages: list[dict[str, Any]], latency_spike_factor: float) -> dict[str, Any] | None:
    if not stages:
        return None

    baseline_p95 = None
    sustainable: dict[str, Any] | None = None

    for stage in stages:
        p95 = stage["latency"]["p95_ms"]
        error_rate = stage["error_rate"]

        if baseline_p95 is None and p95 is not None and error_rate == 0.0:
            baseline_p95 = p95

        latency_ok = True
        if baseline_p95 is not None and p95 is not None:
            latency_ok = p95 <= baseline_p95 * latency_spike_factor

        error_ok = error_rate == 0.0
        if latency_ok and error_ok:
            sustainable = {
                "concurrency": stage["concurrency"],
                "rps": stage["rps"],
                "p95_ms": p95,
                "p99_ms": stage["latency"]["p99_ms"],
                "error_rate": error_rate,
                "selection_rule": (
                    "highest concurrency with zero errors and p95 latency below "
                    f"{latency_spike_factor}x the baseline stage"
                ),
            }

    return sustainable


def plot_rps_vs_concurrency(stages: list[dict[str, Any]], output_dir: Path) -> None:
    if not stages:
        return
    x = [stage["concurrency"] for stage in stages]
    y = [stage["rps"] for stage in stages]
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, marker="o")
    plt.title("Requests Per Second by Concurrency")
    plt.xlabel("Concurrency")
    plt.ylabel("RPS")
    plt.tight_layout()
    plt.savefig(output_dir / "throughput_rps_by_concurrency.png", dpi=200)
    plt.close()


def plot_tail_latency(stages: list[dict[str, Any]], output_dir: Path) -> None:
    if not stages:
        return
    x = [stage["concurrency"] for stage in stages]
    p95 = [stage["latency"]["p95_ms"] for stage in stages]
    p99 = [stage["latency"]["p99_ms"] for stage in stages]
    plt.figure(figsize=(8, 5))
    plt.plot(x, p95, marker="o", label="P95")
    plt.plot(x, p99, marker="o", label="P99")
    plt.title("Tail Latency by Concurrency")
    plt.xlabel("Concurrency")
    plt.ylabel("Latency (ms)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "throughput_tail_latency_by_concurrency.png", dpi=200)
    plt.close()


def plot_error_rate(stages: list[dict[str, Any]], output_dir: Path) -> None:
    if not stages:
        return
    x = [str(stage["concurrency"]) for stage in stages]
    error_rates = [100.0 * stage["error_rate"] for stage in stages]
    server_error_rates = [100.0 * stage["server_error_rate"] for stage in stages]
    indices = list(range(len(x)))
    width = 0.35
    plt.figure(figsize=(8, 5))
    plt.bar([i - width / 2 for i in indices], error_rates, width=width, label="Total error rate %")
    plt.bar([i + width / 2 for i in indices], server_error_rates, width=width, label="5xx error rate %")
    plt.title("Error Rate by Concurrency")
    plt.xlabel("Concurrency")
    plt.ylabel("Error rate (%)")
    plt.xticks(indices, x)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "throughput_error_rate_by_concurrency.png", dpi=200)
    plt.close()


async def evaluate_throughput_async(
    base_url: str,
    dataset_path: Path,
    concurrency_levels: list[int],
    requests_per_level: int,
    timeout_s: float,
    warmup_requests: int,
    latency_spike_factor: float,
) -> dict[str, Any]:
    rows = load_json(dataset_path)
    request_specs = [build_request_spec(base_url, row) for row in rows]

    timeout = httpx.Timeout(timeout_s)
    limits = httpx.Limits(
        max_keepalive_connections=max(concurrency_levels),
        max_connections=max(concurrency_levels),
    )

    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        health_response = await client.get(f"{base_url.rstrip('/')}/health")
        health_response.raise_for_status()
        health_payload = health_response.json()

        warmup_details: list[dict[str, Any]] = []
        for idx in range(max(0, warmup_requests)):
            spec = request_specs[idx % len(request_specs)]
            warmup_details.append(await send_one_request(client, spec, idx + 1))

        stages: list[dict[str, Any]] = []
        for concurrency in concurrency_levels:
            stage = await run_load_stage(
                client=client,
                request_specs=request_specs,
                concurrency=concurrency,
                total_requests=requests_per_level,
            )
            stages.append(stage)

    return {
        "base_url": base_url,
        "dataset_path": str(dataset_path),
        "requests_per_level": requests_per_level,
        "warmup_requests": warmup_requests,
        "timeout_s": timeout_s,
        "health": health_payload,
        "max_sustainable_stage": choose_max_sustainable_stage(stages, latency_spike_factor),
        "stages": stages,
        "warmup": {
            "requests": len(warmup_details),
            "success_rate": round(sum(1 for item in warmup_details if item["ok"]) / len(warmup_details), 4)
            if warmup_details
            else None,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stress-test the FastAPI server and measure throughput, tail latency, and error rate."
    )
    parser.add_argument("--base-url", type=str, default="http://127.0.0.1:8000", help="Base URL of the FastAPI app.")
    parser.add_argument("--dataset", type=Path, required=True, help="Path to throughput evaluation JSON file.")
    parser.add_argument(
        "--concurrency-levels",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16],
        help="Concurrency levels to test in sequence.",
    )
    parser.add_argument(
        "--requests-per-level",
        type=int,
        default=40,
        help="How many total requests to send at each concurrency level.",
    )
    parser.add_argument("--timeout-s", type=float, default=120.0, help="Per-request timeout in seconds.")
    parser.add_argument("--warmup-requests", type=int, default=2, help="Warmup requests before load testing.")
    parser.add_argument(
        "--latency-spike-factor",
        type=float,
        default=2.0,
        help="Defines a latency spike relative to the baseline p95 when selecting max sustainable RPS.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("data/throughput_results.json"),
        help="Where to save the throughput evaluation results JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/throughput_plots"),
        help="Where to save throughput evaluation plots.",
    )
    args = parser.parse_args()

    ensure_parent_dir(args.output_json)
    ensure_output_dir(args.output_dir)

    results = asyncio.run(
        evaluate_throughput_async(
            base_url=args.base_url,
            dataset_path=args.dataset,
            concurrency_levels=args.concurrency_levels,
            requests_per_level=args.requests_per_level,
            timeout_s=args.timeout_s,
            warmup_requests=args.warmup_requests,
            latency_spike_factor=args.latency_spike_factor,
        )
    )

    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    plot_rps_vs_concurrency(results["stages"], args.output_dir)
    plot_tail_latency(results["stages"], args.output_dir)
    plot_error_rate(results["stages"], args.output_dir)

    summary = {
        "max_sustainable_stage": results["max_sustainable_stage"],
        "stages": [
            {
                "concurrency": stage["concurrency"],
                "rps": stage["rps"],
                "p95_ms": stage["latency"]["p95_ms"],
                "p99_ms": stage["latency"]["p99_ms"],
                "error_rate": stage["error_rate"],
                "server_error_rate": stage["server_error_rate"],
            }
            for stage in results["stages"]
        ],
    }
    print(json.dumps(summary, indent=2))
    print(f"Saved JSON results to {args.output_json}")
    print(f"Saved plots to {args.output_dir}")


if __name__ == "__main__":
    main()