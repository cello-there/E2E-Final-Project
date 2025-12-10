#!/usr/bin/env python
"""
make_mini_flickr30k.py

Create a small "mini" subset of Flickr30k for fast testing.

Assumes a full Flickr30k index already exists, e.g.:
  datasets/fickr_30k/flickr30k_index.json

and creates something like:
  datasets/fickr_30k/flickr30k_mini_index.json
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
    parser.add_argument(
        "--full-index",
        type=str,
        default="datasets/fickr_30k/flickr30k_index.json",
        help="Path to full Flickr30k index JSON",
    )
    parser.add_argument(
        "--mini-index",
        type=str,
        default="datasets/fickr_30k/flickr30k_mini_index.json",
        help="Path to write mini Flickr30k index JSON",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=500,
        help="Number of items in the mini subset",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for sampling",
    )
    args = parser.parse_args()

    make_subset(
        full_path=Path(args.full_index),
        mini_path=Path(args.mini_index),
        n_items=args.size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
