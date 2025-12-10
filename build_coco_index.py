#!/usr/bin/env python

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--coco-root",
        type=str,
        required=True,
        help="Folder containing val2014/ and annotations/ (e.g. D:/datasets/coco2014)",
    )
    parser.add_argument(
        "--ann-file",
        type=str,
        default="annotations/captions_val2014.json",
        help="Relative path to captions file from coco-root.",
    )
    parser.add_argument(
        "--out-index",
        type=str,
        default="data/coco/coco_val2014_index.json",
        help="Where to write the full index JSON (relative to project root).",
    )
    parser.add_argument(
        "--max-captions-per-image",
        type=int,
        default=5,
        help="Truncate to at most this many captions per image.",
    )
    args = parser.parse_args()

    coco_root = Path(args.coco_root)
    ann_path = coco_root / args.ann_file
    out_index = Path(args.out_index)

    print(f"Loading annotations from {ann_path}")
    with ann_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Map image_id -> file_name
    id_to_filename: Dict[int, str] = {
        img["id"]: img["file_name"] for img in data["images"]
    }

    # Map image_id -> list of captions
    id_to_caps: Dict[int, List[str]] = {}
    for ann in data["annotations"]:
        img_id = ann["image_id"]
        cap = ann["caption"]
        id_to_caps.setdefault(img_id, []).append(cap)

    items: List[Dict[str, Any]] = []
    for img_id, fname in id_to_filename.items():
        caps = id_to_caps.get(img_id, [])
        if not caps:
            continue
        caps = caps[: args.max_captions_per_image]
        # IMPORTANT: relative to image_root in run_benchmarks.py
        items.append(
            {
                "image": f"val2014/{fname}",
                "captions": caps,
            }
        )

    out_index.parent.mkdir(parents=True, exist_ok=True)
    with out_index.open("w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(items)} entries → {out_index}")


if __name__ == "__main__":
    main()