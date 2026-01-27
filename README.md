# Image Similarity Pro (10k–100k images) — pHash + OpenCV ORB rerank

This project is optimized for large image databases by using:

1) **Average hash (aHash)** to scan the entire database quickly (vectorized byte-wise Hamming distance). aHash is better for images with similar colors and overall structure.
2) **OpenCV ORB** feature matching to rerank only the top candidates (fast + robust for crops/rotation).

It returns **Top-N** results with a **pHash similarity %** and an **ORB match score**.

## Folder layout

- `image_database/`  → put your 10k–100k images here (subfolders allowed)
- `samples/`         → put your query images here
- `index/`           → generated index files

## Install

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Build the index (run once)

```bash
python build_index.py --db image_database --out index
```

Creates:
- `index/paths.jsonl`
- `index/hashes.npy`  (N x 128 uint8 packed bytes for 1024-bit aHash)
- `index/meta.json`

## Search Top-N matches

```bash
python search.py --query samples/sample.jpg --index index --top 10
```

### With ORB reranking (recommended)
```bash
python search.py --query samples/sample.jpg --index index --top 10 --rerank 50
```

- `--top` = number of final results
- `--rerank` = take best K by pHash, then rerank those using ORB (0 disables)

## Notes on scores

- **pHash similarity %** is derived from Hamming distance over 1024 bits (0 distance → 100%). Using aHash for better color-based similarity.
- **ORB score %** is a heuristic (% of good feature matches), useful when crops/rotations happen.

## Tips

- If your database has many near-duplicates, increase `--hash-size` at index time (default 32 → 1024 bits).
- If indexing is slow, keep `--max-side None` (default None to preserve original resolution).
- aHash works well for images with similar colors; if frequency details matter more, switch back to pHash.

