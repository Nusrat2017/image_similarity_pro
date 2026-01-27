import os
import json
import numpy as np
from tqdm import tqdm

from utils import iter_images, phash_packed_bytes, save_paths_jsonl

def build_index(db_folder: str, out_dir: str, hash_size: int = 16, max_side: int | None = 1024):
    os.makedirs(out_dir, exist_ok=True)

    paths: list[str] = []
    packed_hashes: list[np.ndarray] = []

    image_list = list(iter_images(db_folder))
    for p in tqdm(image_list, desc="Indexing images"):
        try:
            ph = phash_packed_bytes(p, hash_size=hash_size, max_side=max_side)
            paths.append(os.path.abspath(p))
            packed_hashes.append(ph)
        except Exception:
            continue

    if not paths:
        raise SystemExit("No images indexed. Check your folder path and file extensions.")

    hashes_arr = np.vstack(packed_hashes).astype(np.uint8)  # (N, B)

    paths_file = os.path.join(out_dir, "paths.jsonl")
    hashes_file = os.path.join(out_dir, "hashes.npy")
    meta_file = os.path.join(out_dir, "meta.json")

    save_paths_jsonl(paths_file, paths)
    np.save(hashes_file, hashes_arr)

    meta = {
        "count": int(hashes_arr.shape[0]),
        "hash_size": int(hash_size),
        "bytes_per_hash": int(hashes_arr.shape[1]),
        "db_folder": os.path.abspath(db_folder),
        "max_side": int(max_side) if max_side else None,
    }
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"✅ Indexed {meta['count']} images")
    print(f"✅ Saved: {paths_file}")
    print(f"✅ Saved: {hashes_file}")
    print(f"✅ Saved: {meta_file}")

def run_build_index():
    db_folder = "image_database/stored_image"  # Specify the path to your image database folder here
    out_dir = "index"  # Specify the output index folder here
    hash_size = 32  # pHash hash size (default: 16 -> 256 bits, increase to 32 for better accuracy)
    max_side = None  # Downscale images so max side <= this (set to None to disable)

    build_index(db_folder, out_dir, hash_size=hash_size, max_side=max_side)

if __name__ == "__main__":
    run_build_index()
