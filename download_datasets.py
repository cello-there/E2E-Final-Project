import argparse
import os
import zipfile
import requests
from pathlib import Path

# --------- Simple downloader with progress printing ---------
def download_file(url: str, dest: Path, chunk_size: int = 1 << 20):
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        print(f"[SKIP] {dest} already exists.")
        return

    print(f"[DOWNLOAD] {url}")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        downloaded = 0

        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = downloaded * 100 / total
                    print(f"\r  -> {downloaded/1e6:7.2f} MB / {total/1e6:7.2f} MB ({pct:5.1f}%)", end="")
        print()  # newline
    print(f"[DONE] Saved to {dest}")


def unzip_file(zip_path: Path, dest_dir: Path):
    print(f"[UNZIP] {zip_path.name} -> {dest_dir}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
    print(f"[DONE] Unzipped {zip_path.name}")


# --------- COCO 2014 ---------
def download_coco2014(root: Path):
    coco_root = root / "coco2014"
    coco_root.mkdir(parents=True, exist_ok=True)

    files = {
        "val2014.zip": "http://images.cocodataset.org/zips/val2014.zip",
        "annotations_trainval2014.zip": "http://images.cocodataset.org/annotations/annotations_trainval2014.zip",
    }

    for name, url in files.items():
        dest = coco_root / name
        download_file(url, dest)

    unzip_file(coco_root / "val2014.zip", coco_root)
    unzip_file(coco_root / "annotations_trainval2014.zip", coco_root)


# --------- Flickr30k ---------
def download_flickr30k(root: Path):
    """
    Uses the awsaf49 GitHub mirror that provides Flickr30k in three parts:
      flickr30k_part00, flickr30k_part01, flickr30k_part02
    which are concatenated into flickr30k.zip and then unzipped.
    """
    flickr_root = root / "flickr30k"
    flickr_root.mkdir(parents=True, exist_ok=True)

    base = "https://github.com/awsaf49/flickr-dataset/releases/download/v1.0"
    parts = ["flickr30k_part00", "flickr30k_part01", "flickr30k_part02"]

    part_paths = []
    for p in parts:
        url = f"{base}/{p}"
        dest = flickr_root / p
        part_paths.append(dest)
        download_file(url, dest)

    # Concatenate into a single zip
    zip_path = flickr_root / "flickr30k.zip"
    if not zip_path.exists():
        print("[MERGE] Combining parts into flickr30k.zip")
        with open(zip_path, "wb") as out_f:
            for part_path in part_paths:
                with open(part_path, "rb") as in_f:
                    while True:
                        chunk = in_f.read(1 << 20)
                        if not chunk:
                            break
                        out_f.write(chunk)
        print("[DONE] Created", zip_path)

    # Clean up parts (optional)
    for part_path in part_paths:
        try:
            part_path.unlink()
        except OSError:
            pass

    # Unzip
    unzip_file(zip_path, flickr_root)


def main():
    parser = argparse.ArgumentParser(description="Download COCO 2014 and Flickr30k datasets.")
    parser.add_argument("--data-root", type=str, default="data",
                        help="Root directory to store datasets (default: data/)")
    parser.add_argument("--coco2014", action="store_true", help="Download COCO 2014 dataset")
    parser.add_argument("--flickr30k", action="store_true", help="Download Flickr30k dataset")

    args = parser.parse_args()
    root = Path(args.data_root)

    # If neither flag is set, download both
    if not args.coco2014 and not args.flickr30k:
        args.coco2014 = True
        args.flickr30k = True

    if args.coco2014:
        print("=== Downloading COCO 2014 ===")
        download_coco2014(root)

    if args.flickr30k:
        print("=== Downloading Flickr30k ===")
        download_flickr30k(root)


if __name__ == "__main__":
    main()
