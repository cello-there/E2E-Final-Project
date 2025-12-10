#!/usr/bin/env python
"""
throughput_sweep.py

Batch-size sweep for throughput + peak GPU memory, reusing the same
model wrappers and dataset loading logic as run_benchmarks.py.

This script:
  - Builds a single (model, benchmark) pair
  - Samples up to N images (default: 1000) from the benchmark
  - For a list of batch sizes, runs:
      * encode_all_images(...)
      * encode_all_texts(...)
    and records:
      * throughput (imgs/s or caps/s)
      * peak GPU memory (MB)
  - Saves everything into a JSON file.

Intended usage in your proposal:
  - Run on COCO (e.g., "coco_5k") and Flickr30k (e.g., "flickr30k_1k")
    with num_samples=1000.
  - CIRR is omitted for this scaling test.

Assumes:
  - run_benchmarks.py is in the same directory and defines:
      BENCHMARKS, MODEL_REGISTRY, load_benchmark,
      ModelConfig, encode_all_images, encode_all_texts
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

# Import utilities from your existing benchmark script
from run_benchmarks import (
    BENCHMARKS,
    MODEL_REGISTRY,
    load_benchmark,
    ModelConfig,
    encode_all_images,
    encode_all_texts,
)
from vl_benchmark.preprocessing import PreprocessConfig


def sample_subset(
    images: List[Any],
    all_captions: List[str],
    img_to_cap_indices: List[List[int]],
    num_samples: int,
    seed: int,
):
    """
    Take up to `num_samples` images (and their captions) from the benchmark.

    Returns:
      sub_images, sub_captions
    (We don't need img_to_cap_indices for throughput-only scaling.)
    """
    n_images = len(images)
    num = min(num_samples, n_images)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_images)[:num]

    sub_images: List[Any] = []
    sub_captions: List[str] = []

    for idx in perm:
        sub_images.append(images[idx])
        cap_indices = img_to_cap_indices[idx]
        for ci in cap_indices:
            sub_captions.append(all_captions[ci])

    return sub_images, sub_captions


def run_throughput_sweep(
    model_key: str,
    benchmark_key: str,
    precision: str,
    mode: str,
    batch_sizes: list[int],
    num_samples: int,
    warmup_batches: int,
    device: torch.device,
    seed: int,
) -> Dict[str, Any]:
    """
    Run throughput + peak memory sweep over a list of batch sizes
    for a single (model, benchmark) pair.
    """
    if model_key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model key: {model_key}")
    if benchmark_key not in BENCHMARKS:
        raise ValueError(f"Unknown benchmark key: {benchmark_key}")

    model_builder = MODEL_REGISTRY[model_key]
    spec = BENCHMARKS[benchmark_key]

    print(f"\n=== Throughput sweep: model={model_key}, benchmark={benchmark_key} ===")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Num samples (max): {num_samples}")

    # Build model wrapper (same as run_benchmarks)
    cfg = ModelConfig(
        name=model_key,
        device=device,
        precision=precision,
        mode=mode,
        shared_preprocess=PreprocessConfig(),
    )
    model = model_builder(cfg)

    # Load full benchmark then subsample
    images_full, caps_full, img_to_cap_indices_full = load_benchmark(spec)
    n_images_full = len(images_full)
    m_caps_full = len(caps_full)
    print(f"Loaded full benchmark: {n_images_full} images, {m_caps_full} captions")

    sub_images, sub_captions = sample_subset(
        images_full,
        caps_full,
        img_to_cap_indices_full,
        num_samples=num_samples,
        seed=seed,
    )
    print(
        f"Using subset for sweep: {len(sub_images)} images, {len(sub_captions)} captions"
    )

    # Ensure deterministic-ish behavior for this run
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Ensure cuDNN autotune is on for realistic throughput
    torch.backends.cudnn.benchmark = True

    results: Dict[str, Any] = {
        "model": model_key,
        "benchmark": benchmark_key,
        "precision": precision,
        "mode": mode,
        "device": str(device),
        "num_samples_images": len(sub_images),
        "num_samples_captions": len(sub_captions),
        "warmup_batches": warmup_batches,
        "batch_sizes": {},
    }

    for bs in batch_sizes:
        print(f"\n--- Batch size = {bs} ---")

        # Clear allocator between batch sizes so peak memory is comparable
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # Image encoding sweep
        img_embeds, img_stats = encode_all_images(
            model=model,
            images=sub_images,
            device=device,
            batch_size=bs,
            warmup_batches=warmup_batches,
        )

        # Text encoding sweep
        txt_embeds, txt_stats = encode_all_texts(
            model=model,
            captions=sub_captions,
            device=device,
            batch_size=bs,
            warmup_batches=warmup_batches,
        )

        # We don't keep img_embeds / txt_embeds here; this script is only for throughput.
        del img_embeds, txt_embeds

        results["batch_sizes"][str(bs)] = {
            "image_stats": img_stats,
            "text_stats": txt_stats,
        }

        print(
            f"[images]  time={img_stats['total_time_s']:.2f}s, "
            f"throughput={img_stats['throughput_imgs_per_s']:.1f} img/s, "
            f"peak_mem={img_stats['peak_mem_mb']:.1f} MB"
        )
        print(
            f"[texts]   time={txt_stats['total_time_s']:.2f}s, "
            f"throughput={txt_stats['throughput_caps_per_s']:.1f} cap/s, "
            f"peak_mem={txt_stats['peak_mem_mb']:.1f} MB"
        )

    return results


def parse_batch_sizes(arg: str) -> list[int]:
    """
    Parse a comma-separated list of integers, e.g. "1,2,4,8,16".
    """
    parts = [p.strip() for p in arg.split(",") if p.strip()]
    return [int(p) for p in parts]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(MODEL_REGISTRY.keys()),
        help=f"Model key to run. Options: {list(MODEL_REGISTRY.keys())}",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        required=True,
        choices=list(BENCHMARKS.keys()),
        help=f"Benchmark key to run. Options: {list(BENCHMARKS.keys())}",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp16",
        choices=["fp32", "fp16", "bf16"],
        help="Autocast precision for CUDA models (default: fp16).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="shared",
        choices=["native", "shared"],
        help="Preprocessing mode (must match how you benchmark models).",
    )
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="1,2,4,8,16,32,64,128",
        help='Comma-separated list of batch sizes, e.g. "1,2,4,8,16".',
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Max number of images to sample from the benchmark (default: 1000).",
    )
    parser.add_argument(
        "--warmup-batches",
        type=int,
        default=1,
        help="Warmup batches for each sweep (excluded from timing).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device string, e.g. 'cuda', 'cuda:0', or 'cpu'.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for sampling subset.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to JSON file where sweep results will be stored.",
    )

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    batch_sizes = parse_batch_sizes(args.batch_sizes)

    results = run_throughput_sweep(
        model_key=args.model,
        benchmark_key=args.benchmark,
        precision=args.precision,
        mode=args.mode,
        batch_sizes=batch_sizes,
        num_samples=args.num_samples,
        warmup_batches=args.warmup_batches,
        device=device,
        seed=args.seed,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved throughput sweep results to {out_path}")


if __name__ == "__main__":
    main()
