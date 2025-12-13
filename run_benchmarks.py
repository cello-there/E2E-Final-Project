#!/usr/bin/env python
"""
run_benchmarks.py

Single (model, benchmark) runner for your VLM retrieval benchmarks.

Design:
  - Each Python process runs exactly ONE (model, benchmark) pair.
  - This avoids CUDA allocator / autotune / cache interference between models.
  - Warmup batches are run but EXCLUDED from timing.
  - We measure:
      * Global latency + throughput for image encoding
      * Global latency + throughput for text encoding
      * Peak VRAM (per phase)
      * Retrieval metrics: Recall@{1,5,10} for T2I and I2T

Assumes the following project structure (adjust imports if needed):

  repo_root/
    run_benchmarks.py
    vl_benchmark/
      __init__.py
      models.py
      preprocessing.py
      ...

Datasets:
  We use simple JSON index files per benchmark, e.g.:

  [
    {
      "image": "relative/path/to/image1.jpg",
      "captions": [
        "caption 1",
        "caption 2",
        ...
      ]
    },
    ...
  ]

This matches the MS-COCO-5k, Flickr30k-1k, and CIRR usage described in the proposal.
"""

from __future__ import annotations
import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple
from torch.utils.data import Dataset, DataLoader

import numpy as np
import torch
from PIL import Image

# Adjust this if your package layout differs
from vl_benchmark.models import (
    ModelConfig,
    build_openclip_wrapper,
    build_siglip2_wrapper,
    build_vlm2vecv2_wrapper,
)
from vl_benchmark.preprocessing import PreprocessConfig


# ================================================================
#  Dataset spec + loading (JSON index format)
# ================================================================

@dataclass
class BenchmarkSpec:
    name: str
    image_root: Path
    index_json: Path


# Update DATA_ROOT / image_root paths to your actual dataset locations
DATA_ROOT = Path("data")

# BENCHMARKS: Dict[str, BenchmarkSpec] = {
#     "coco_5k": BenchmarkSpec(
#         name="coco_5k",
#         image_root=DATA_ROOT / "coco" / "images" / "val2014",
#         index_json=DATA_ROOT / "coco" / "coco_5k_index.json",
#     ),
#     "flickr30k_1k": BenchmarkSpec(
#         name="flickr30k_1k",
#         image_root=DATA_ROOT / "flickr30k" / "images",
#         index_json=DATA_ROOT / "flickr30k" / "flickr30k_1k_index.json",
#     ),
#     "cirr": BenchmarkSpec(
#         name="cirr",
#         image_root=DATA_ROOT / "cirr" / "images",
#         index_json=DATA_ROOT / "cirr" / "cirr_index.json",
#     ),
# }

DATA_ROOT = Path("DATA")         # where the *index* JSONs live
DATASETS_ROOT = Path("DATA") # where the raw datasets live (your coco2014 folder)

# BENCHMARKS: Dict[str, BenchmarkSpec] = {
#     # Mini COCO benchmark for pipeline sanity testing
#     "coco_mini": BenchmarkSpec(
#         name="coco_mini",
#         image_root=DATASETS_ROOT / "coco2014",
#         index_json=DATA_ROOT / "coco" / "coco_mini_index.json",
#     ),

#     # Mini Flickr30k benchmark
#     "flickr30k_mini": BenchmarkSpec(
#         name="flickr30k_mini",
#         # Images live under datasets/fickr_30k/images/
#         image_root=DATASETS_ROOT / "flickr_30k",
#         index_json=DATASETS_ROOT / "flickr" / "flickr30k_mini_index.json",
#     ),
# }

BENCHMARKS: Dict[str, BenchmarkSpec] = {
    # Mini COCO benchmark for pipeline sanity testing
    "coco_mini": BenchmarkSpec(
        name="coco_mini",
        image_root=DATASETS_ROOT / "coco2014",
        index_json=DATA_ROOT / "coco" / "coco_mini_index.json",
    ),

    # Mini Flickr30k benchmark
    "flickr30k_mini": BenchmarkSpec(
        name="flickr30k_mini",
        # Images live under datasets/fickr_30k/images/
        image_root=DATASETS_ROOT / "flickr_30k",
        index_json=DATASETS_ROOT / "flickr" / "flickr30k_mini_index.json",
    ),
    # Mini COCO benchmark for pipeline sanity testing
    "coco": BenchmarkSpec(
        name="coco",
        image_root=DATASETS_ROOT / "coco2014",
        index_json=DATA_ROOT / "coco" / "coco_val2014_index.json",
    ),

    # Mini Flickr30k benchmark
    "flickr30k": BenchmarkSpec(
        name="flickr30k",
        # Images live under datasets/fickr_30k/images/
        image_root=DATASETS_ROOT / "flickr_30k",
        index_json=DATASETS_ROOT / "flickr" / "flickr30k_index.json",
    ),
}



def load_benchmark(spec: BenchmarkSpec) -> Tuple[List[Image.Image], List[str], List[List[int]]]:
    """
    Load images + captions using the JSON index.

    JSON format:

      [
        {
          "image": "relative/path/to/image.jpg",
          "captions": ["cap1", "cap2", ...]
        },
        ...
      ]

    Returns:
      images: list of PIL.Image (len = N images)
      all_captions: flat list[str] of all captions (len = M captions)
      img_to_cap_indices: list[list[int]] mapping image_idx -> list of indices into all_captions
    """
    with spec.index_json.open("r", encoding="utf-8") as f:
        items = json.load(f)

    image_paths: List[Path] = []
    all_captions: List[str] = []
    img_to_cap_indices: List[List[int]] = []

    for item in items:
        img_path = spec.image_root / item["image"]
        image_paths.append(img_path)

        caps = item["captions"]
        cap_indices = []
        for cap in caps:
            cap_indices.append(len(all_captions))
            all_captions.append(cap)
        img_to_cap_indices.append(cap_indices)

    return image_paths, all_captions, img_to_cap_indices

class ImagePathDataset(Dataset):
    def __init__(self, paths: List[Path]):
        self.paths=paths
    
    def __len__(self) -> ints:
        return len(self.paths)
    
    def __getitem__(self, idx: int):
        p = self.paths[idx]

        with Image.open(p) as im:
            return im.convert("RGB")

def pil_collate(batch):
    return batch

# ================================================================
#  Retrieval metrics (Recall@K)
# ================================================================

def compute_text_to_image_recall(
    text_embeds: torch.Tensor,
    image_embeds: torch.Tensor,
    caption_to_image: np.ndarray,
    ks: Tuple[int, ...] = (1, 5, 10),
) -> Dict[str, float]:
    """
    Text-to-image retrieval (caption → image).

    text_embeds: [M, D] (L2-normalized)
    image_embeds: [N, D] (L2-normalized)
    caption_to_image: [M] ground-truth image index per caption
    """
    text_embeds = text_embeds.float()
    image_embeds = image_embeds.float()

    sim = text_embeds @ image_embeds.t()  # [M, N]

    max_k = max(ks)
    _, topk_idx = sim.topk(k=max_k, dim=1)  # [M, max_k]

    gt = torch.from_numpy(caption_to_image).to(text_embeds.device)
    recalls: Dict[str, float] = {}

    for k in ks:
        hits = (topk_idx[:, :k] == gt.unsqueeze(1)).any(dim=1).float().mean().item()
        recalls[f"t2i_R@{k}"] = hits

    return recalls


def compute_image_to_text_recall(
    image_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    img_to_cap_indices: List[List[int]],
    ks: Tuple[int, ...] = (1, 5, 10),
) -> Dict[str, float]:
    """
    Image-to-text retrieval (image → caption).

    image_embeds: [N, D] (L2-normalized)
    text_embeds: [M, D] (L2-normalized)
    img_to_cap_indices: ground-truth caption indices per image
    """
    image_embeds = image_embeds.float()
    text_embeds = text_embeds.float()

    sim = image_embeds @ text_embeds.t()  # [N, M]

    max_k = max(ks)
    _, topk_idx = sim.topk(k=max_k, dim=1)  # [N, max_k]

    recalls: Dict[str, float] = {}
    N = image_embeds.size(0)

    for k in ks:
        hits = 0
        for i in range(N):
            gt_set = set(img_to_cap_indices[i])
            preds = topk_idx[i, :k].tolist()
            if any(p in gt_set for p in preds):
                hits += 1
        recalls[f"i2t_R@{k}"] = hits / N

    return recalls


# ================================================================
#  Encoding with warmup + global timing
# ================================================================

def _load_rgb_batch(paths: List[Path]) -> List[Image.Image]:
    imgs:List[Image.Image] = []
    for p in paths:
        with Image.open(p) as im:
            imgs.append(im.convert("RGB"))
    return imgs

def encode_all_images(
    model,
    image_paths: List[Path],
    device: torch.device,
    batch_size: int,
    warmup_batches: int = 1,
    num_workers: int = 8,
    prefetch_factor: int = 4 
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Encode all images with warmup.

    Returns:
      embs: [N, D] on CPU
      stats: dict with total_time_s, throughput_imgs_per_s, peak_mem_mb, etc.
    """
    model.eval()

    ds = ImagePathDataset(image_paths)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory = (device.type == "cuda"),
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=(num_workers > 0),
        collate_fn = pil_collate,
        drop_last = False,
    )

    # --- Warmup (excluded from timing) ---
    with torch.no_grad():
        for i, batch_imgs in enumerate(loader):
            _embs, _ = model.encode_images(
                batch_imgs,
                normalize=True,
                return_stats=False,
            )
            if i + 1 >= warmup_batches:
                break

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    start = time.perf_counter()
    all_embs: List[torch.Tensor] = []

    with torch.no_grad():
        for batch_imgs in loader:
            embs, _ = model.encode_images(
                batch_imgs,
                normalize=True,
                return_stats=False,
            )
            all_embs.append(embs.cpu())

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    else:
        peak_mem_mb = float("nan")
    
    n = len(ds)

    total_time_s = time.perf_counter() - start
    throughput = n / total_time_s if total_time_s > 0 else float("nan")

    stats = {
        "total_time_s": total_time_s,
        "throughput_imgs_per_s": throughput,
        "peak_mem_mb": peak_mem_mb,
        "num_images": n,
        "batch_size": batch_size,
        "warmup_batches": warmup_batches,
    }

    embs_tensor = torch.cat(all_embs, dim=0)  # [N, D]
    return embs_tensor, stats


def encode_all_texts(
    model,
    captions: List[str],
    device: torch.device,
    batch_size: int,
    warmup_batches: int = 1,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Encode all captions with warmup.

    Returns:
      embs: [M, D] on CPU
      stats: dict with total_time_s, throughput_caps_per_s, peak_mem_mb, etc.
    """
    model.eval()
    m = len(captions)
    num_batches = math.ceil(m / batch_size)

    # --- Warmup (excluded from timing) ---
    with torch.no_grad():
        for b in range(min(warmup_batches, num_batches)):
            batch_caps = captions[b * batch_size : (b + 1) * batch_size]
            _embs, _ = model.encode_texts(
                batch_caps,
                normalize=True,
                return_stats=False,
            )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    start = time.perf_counter()
    all_embs: List[torch.Tensor] = []

    with torch.no_grad():
        for b in range(num_batches):
            batch_caps = captions[b * batch_size : (b + 1) * batch_size]
            embs, _ = model.encode_texts(
                batch_caps,
                normalize=True,
                return_stats=False,
            )
            all_embs.append(embs.cpu())

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    else:
        peak_mem_mb = float("nan")

    total_time_s = time.perf_counter() - start
    throughput = m / total_time_s if total_time_s > 0 else float("nan")

    stats = {
        "total_time_s": total_time_s,
        "throughput_caps_per_s": throughput,
        "peak_mem_mb": peak_mem_mb,
        "num_captions": m,
        "batch_size": batch_size,
        "warmup_batches": warmup_batches,
    }

    embs_tensor = torch.cat(all_embs, dim=0)  # [M, D]
    return embs_tensor, stats


# ================================================================
#  Model registry (what we benchmark)
# ================================================================

def build_openclip(cfg: ModelConfig):
    # Adjust to your chosen OpenCLIP variant
    return build_openclip_wrapper(
        cfg,
        model_name="ViT-B-16",
        pretrained="laion2b_s34b_b88k",
    )


def build_siglip2(cfg: ModelConfig):
    # Adjust to your chosen SigLIP-2 checkpoint
    #hf_name = "google/siglip-base-patch16-224"
    hf_name ="google/siglip-so400m-patch14-384"
    return build_siglip2_wrapper(cfg, hf_name=hf_name)


def build_vlm2vecv2(cfg: ModelConfig):
    # Default HF name is encoded in the wrapper; override if needed
    return build_vlm2vecv2_wrapper(cfg)


MODEL_REGISTRY = {
    "openclip_vitb16": build_openclip,
    "siglip2": build_siglip2,
    "vlm2vecv2": build_vlm2vecv2,
}


# ================================================================
#  Core runner for a single (model, benchmark)
# ================================================================

def run_single_benchmark(
    model_key: str,
    benchmark_key: str,
    precision: str,
    mode: str,
    image_batch_size: int,
    text_batch_size: int,
    warmup_batches: int,
    device: torch.device,
    save_embeddings: bool = False,
    embeddings_dir: Path | None = None,
) -> Dict[str, Any]:
    """
    Run a single (model, benchmark) pair and return a result dict.
    """
    if model_key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model key: {model_key}")
    if benchmark_key not in BENCHMARKS:
        raise ValueError(f"Unknown benchmark key: {benchmark_key}")

    model_builder = MODEL_REGISTRY[model_key]
    spec = BENCHMARKS[benchmark_key]

    print(f"\n=== Running model={model_key} on benchmark={benchmark_key} ===")

    # Optional: cuDNN autotune for best throughput
    torch.backends.cudnn.benchmark = True

    # Fresh CUDA allocator state for this run
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ----------------- Build model wrapper -----------------
    cfg = ModelConfig(
        name=model_key,
        device=device,
        precision=precision,          # "fp32", "fp16", or "bf16"
        mode=mode,                    # "native" or "shared"
        shared_preprocess=PreprocessConfig(),  # identical shared pipeline across models
    )
    model = model_builder(cfg)

    # ----------------- Load data -----------------
    image_paths, all_captions, img_to_cap_indices = load_benchmark(spec)
    n_images = len(image_paths)
    m_caps = len(all_captions)
    print(f"Loaded {n_images} images and {m_caps} captions from {spec.index_json}")

    # caption_index -> image_index mapping (for T2I GT)
    caption_to_image = np.empty(m_caps, dtype=np.int64)
    for img_idx, cap_indices in enumerate(img_to_cap_indices):
        for ci in cap_indices:
            caption_to_image[ci] = img_idx

    # ----------------- Encode images -----------------
    img_embeds, img_stats = encode_all_images(
        model=model,
        image_paths=image_paths,
        device=device,
        batch_size=image_batch_size,
        warmup_batches=warmup_batches,
    )
    print(
        f"Image encoding: {img_stats['total_time_s']:.2f}s, "
        f"{img_stats['throughput_imgs_per_s']:.1f} img/s, "
        f"peak {img_stats['peak_mem_mb']:.1f} MB"
    )

    # ----------------- Encode texts -----------------
    txt_embeds, txt_stats = encode_all_texts(
        model=model,
        captions=all_captions,
        device=device,
        batch_size=text_batch_size,
        warmup_batches=warmup_batches,
    )
    print(
        f"Text encoding: {txt_stats['total_time_s']:.2f}s, "
        f"{txt_stats['throughput_caps_per_s']:.1f} cap/s, "
        f"peak {txt_stats['peak_mem_mb']:.1f} MB"
    )

    # ----------------- Retrieval metrics -----------------
    t2i_recalls = compute_text_to_image_recall(
        text_embeds=txt_embeds,
        image_embeds=img_embeds,
        caption_to_image=caption_to_image,
        ks=(1, 5, 10),
    )
    i2t_recalls = compute_image_to_text_recall(
        image_embeds=img_embeds,
        text_embeds=txt_embeds,
        img_to_cap_indices=img_to_cap_indices,
        ks=(1, 5, 10),
    )

    print("Text→Image:", {k: round(v * 100, 2) for k, v in t2i_recalls.items()})
    print("Image→Text:", {k: round(v * 100, 2) for k, v in i2t_recalls.items()})

    # ----------------- Optionally save embeddings -----------------
    embedding_paths: Dict[str, str] = {}
    if save_embeddings:
        if embeddings_dir is None:
            embeddings_dir = Path("embeddings")

        embeddings_dir.mkdir(parents=True, exist_ok=True)

        # Make filename unique per (model, benchmark, precision, mode)
        stem = f"{model_key}__{benchmark_key}__{precision}__{mode}"

        img_path = embeddings_dir / f"{stem}__img_embeds.pt"
        txt_path = embeddings_dir / f"{stem}__txt_embeds.pt"

        # img_embeds / txt_embeds are already on CPU
        torch.save(img_embeds, img_path)
        torch.save(txt_embeds, txt_path)

        embedding_paths = {
            "image_embeddings_path": str(img_path),
            "text_embeddings_path": str(txt_path),
        }


    result: Dict[str, Any] = {
        "model": model_key,
        "benchmark": benchmark_key,
        "precision": precision,
        "mode": mode,
        "image_batch_size": image_batch_size,
        "text_batch_size": text_batch_size,
        "warmup_batches": warmup_batches,
        "image_stats": img_stats,
        "text_stats": txt_stats,
        "t2i_recalls": t2i_recalls,
        "i2t_recalls": i2t_recalls,
        **embedding_paths,
    }
    return result


# ================================================================
#  CLI entrypoint (single combo per process)
# ================================================================

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
        help="Preprocessing mode. 'shared' enforces common preprocessing across models.",
    )
    parser.add_argument(
        "--image-batch-size",
        type=int,
        default=256,
        help="Batch size for image encoding.",
    )
    parser.add_argument(
        "--text-batch-size",
        type=int,
        default=256,
        help="Batch size for text encoding.",
    )
    parser.add_argument(
        "--warmup-batches",
        type=int,
        default=2,
        help="Number of warmup batches (excluded from timing).",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to JSON file where this run's result will be stored.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device string for torch.device, e.g. 'cuda', 'cuda:0', or 'cpu'.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (for reproducibility of any stochastic components).",
    )

    parser.add_argument(
        "--save-embeddings",
        action="store_true",
        help="If set, save image/text embeddings to .pt files.",
    )
    parser.add_argument(
        "--embeddings-dir",
        type=str,
        default=None,
        help="Directory in which to save embeddings (.pt). "
             "Defaults to the output JSON's directory.",
    )


    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Basic seeding (not crucial for retrieval, but keeps things tidy)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.embeddings_dir is not None:
        embeddings_dir = Path(args.embeddings_dir)
    else:
        # default: same directory as the JSON result
        embeddings_dir = output_path.parent


    result = run_single_benchmark(
        model_key=args.model,
        benchmark_key=args.benchmark,
        precision=args.precision,
        mode=args.mode,
        image_batch_size=args.image_batch_size,
        text_batch_size=args.text_batch_size,
        warmup_batches=args.warmup_batches,
        device=device,
        save_embeddings=args.save_embeddings,
        embeddings_dir=embeddings_dir,
    )
    
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved result to {output_path}")


if __name__ == "__main__":
    main()
