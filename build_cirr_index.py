#!/usr/bin/env python

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any

def load_captions(json_path: Path) -> List[Any]:
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cirr-root",
        type=str,
        required=True,
        help="Folder containing images/ and captions/train.json, dev.json, test.json"
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train.json", "dev.json"],
        help="Which CIRR caption files to include (absolute captions only)."
    )
    parser.add_argument(
        "--out-index",
        type=str,
        default="data/cirr/cirr_index.json",
        help="Output path for the built CIRR index."
    )
    parser.add_argument(
        "--max-captions-per-image",
        type=int,
        default=5,
        help="Maximum captions per image."
    )
    args = parser.parse_args()

    cirr_root = Path(args.cirr_root)
    captions_dir = cirr_root / "captions"

    id_to_caps: Dict[str, List[str]] = {}

    print("Loading CIRR captions...")
    for split in args.splits:
        split_path = captions_dir / split
        caps = load_captions(split_path)
        print(f"  Loaded {len(caps)} items from {split_path}")

        # each cap entry: { "image_id": "...", "caption": "..." }
        for entry in caps:
            img_id = entry["image_id"]
            caption = entry["caption"]
            id_to_caps.setdefault(img_id, []).append(caption)

    # Build final index
    items: List[Dict[str, Any]] = []
    for img_id, caps in id_to_caps.items():
        fname = f"{img_id}.jpg"
        caps = caps[: args.max_captions_per_image]

        items.append({
            "image": f"images/{fname}",
            "captions": caps,
        })

    out_index = Path(args.out_index)
    out_index.parent.mkdir(parents=True, exist_ok=True)
    with out_index.open("w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(items)} entries → {out_index}")

if __name__ == "__main__":
    main()
