import os
import json
import numpy as np
from PIL import Image
import imagehash

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

# Precompute popcount for bytes 0..255 (fast Hamming distance for packed uint8 arrays)
POPCOUNT_LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

def iter_images(folder: str):
    for root, _, files in os.walk(folder):
        for name in files:
            ext = os.path.splitext(name.lower())[1]
            if ext in IMAGE_EXTS:
                yield os.path.join(root, name)

def open_image_rgb(path: str) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")

def resize_max_side(img: Image.Image, max_side: int | None) -> Image.Image:
    if not max_side:
        return img
    w, h = img.size
    m = max(w, h)
    if m <= max_side:
        return img
    scale = max_side / m
    nw, nh = int(round(w * scale)), int(round(h * scale))
    return img.resize((nw, nh))

def phash_packed_bytes(path: str, hash_size: int = 16, max_side: int | None = None) -> np.ndarray:
    """Return packed bytes of pHash.
    Output shape: (hash_size*hash_size/8,) uint8.
    For hash_size=16 => 256 bits => 32 bytes.
    """
    img = open_image_rgb(path)
    img = resize_max_side(img, max_side)
    h = imagehash.average_hash(img, hash_size=hash_size)  # Average hash
    bits = np.asarray(h.hash, dtype=np.uint8).reshape(-1)  # 0/1
    packed = np.packbits(bits)  # uint8
    return packed

def hamming_distances_packed(query_packed: np.ndarray, db_packed: np.ndarray) -> np.ndarray:
    """Vectorized Hamming distance for packed uint8 hashes.
    query_packed: (B,)
    db_packed:    (N,B)
    returns: (N,) bit distances
    """
    xor = np.bitwise_xor(db_packed, query_packed)  # (N,B)
    return POPCOUNT_LUT[xor].sum(axis=1).astype(np.int32)

def dist_to_percent(dist_bits: int, hash_size: int) -> float:
    total_bits = hash_size * hash_size
    return max(0.0, (1.0 - dist_bits / total_bits) * 100.0)

def load_paths_jsonl(paths_file: str) -> list[str]:
    paths: list[str] = []
    with open(paths_file, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            paths.append(obj["path"])
    return paths

def save_paths_jsonl(paths_file: str, paths: list[str]):
    with open(paths_file, "w", encoding="utf-8") as f:
        for p in paths:
            f.write(json.dumps({"path": p}, ensure_ascii=False) + "\n")
