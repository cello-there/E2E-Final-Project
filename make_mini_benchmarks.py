#!/usr/bin/env python
"""
make_mini_benchmarks.py

Create a small "mini" subset of COCO for fast testing.

Assumes a full COCO index already exists at:
  data/coco/coco_val2014_index.json

and creates:
  data/coco/coco_mini_index.json
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict, Any


def load_items(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_items(items: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(items)} items → {path}")


def make_subset(full_path: Path, mini_path: Path, n_items: int, seed: int) -> None:
    print(f"Loading full index from {full_path}")
    items = load_items(full_path)
    total = len(items)
    print(f"Total items in full index: {total}")

    random.seed(seed)
    if n_items >= total:
        print(f"Requested {n_items}, but full index has only {total}. Using all.")
        subset = items
    else:
        subset = random.sample(items, n_items)

    save_items(subset, mini_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data", help="Dataset root folder")
    parser.add_argument("--coco-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    root = Path(args.data_root)

    make_subset(
        full_path=root / "coco" / "coco_val2014_index.json",
        mini_path=root / "coco" / "coco_mini_index.json",
        n_items=args.coco_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
