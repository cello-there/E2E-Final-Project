#!/usr/bin/env python

import argparse
import json
import csv
from pathlib import Path
from typing import Dict, List, Any


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--flickr-root",
        type=str,
        required=True,
        help="Folder containing images/ and results.csv (e.g. D:/datasets/flickr30k)",
    )
    parser.add_argument(
        "--captions-file",
        type=str,
        default="results.csv",
        help="Relative path to Flickr30k captions CSV file.",
    )
    parser.add_argument(
        "--split-file",
        type=str,
        default=None,
        help="Optional text file containing the 1k test split list.",
    )
    parser.add_argument(
        "--out-index",
        type=str,
        default="data/flickr30k/flickr30k_1k_index.json",
        help="Where to write the full index JSON.",
    )
    parser.add_argument(
        "--max-captions-per-image",
        type=int,
        default=5,
        help="Truncate to at most this many captions per image.",
    )
    args = parser.parse_args()

    flickr_root = Path(args.flickr_root)
    captions_path = flickr_root / args.captions_file

    # If split_file provided, use only those images
    limit_to_images = None
    if args.split_file:
        with open(args.split_file, "r", encoding="utf-8") as f:
            limit_to_images = set(
                line.strip().split(".")[0] + ".jpg"
                for line in f
                if line.strip()
            )
        print(f"Loaded {len(limit_to_images)} images for 1k split")

    print(f"Loading captions from {captions_path}")
    id_to_caps: Dict[str, List[str]] = {}

    # CSV format (mostly): image_name| comment_number| comment
    with captions_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="|", skipinitialspace=True)

        for row in reader:
            if not row:
                continue

            fname_raw = row.get("image_name")
            num_raw = row.get("comment_number")
            cap_raw = row.get("comment")

            # --- Repair broken lines like:
            # 2199200615.jpg| 4   A dog runs across the grass .
            # which become:
            # 'comment_number': '4   A dog runs across the grass .', 'comment': None
            if (cap_raw is None or cap_raw == "") and isinstance(num_raw, str):
                fixed = num_raw.strip()
                parts = fixed.split(None, 1)  # split on first whitespace
                if len(parts) == 2:
                    # first token is the number, rest is the caption
                    num_fixed, cap_fixed = parts
                    row["comment_number"] = num_fixed
                    cap_raw = cap_fixed

            # Now proceed with cleaned values
            if not fname_raw or not cap_raw:
                continue

            fname = fname_raw.strip()
            cap = cap_raw.strip()

            if not fname or not cap:
                continue

            if limit_to_images and fname not in limit_to_images:
                continue

            id_to_caps.setdefault(fname, []).append(cap)

    # Build index items
    items: List[Dict[str, Any]] = []
    for fname, caps in id_to_caps.items():
        caps = caps[: args.max_captions_per_image]
        items.append(
            {
                "image": f"images/{fname}",  # matches your folder name
                "captions": caps,
            }
        )

    out_index = Path(args.out_index)
    out_index.parent.mkdir(parents=True, exist_ok=True)
    with out_index.open("w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(items)} items → {out_index}")


if __name__ == "__main__":
    main()
