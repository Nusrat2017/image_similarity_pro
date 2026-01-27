import os
import json
import numpy as np
import cv2

from utils import (
    phash_packed_bytes,
    hamming_distances_packed,
    dist_to_percent,
    load_paths_jsonl,
)

def orb_rerank(query_path: str, candidate_paths: list[str], max_side: int = 1024, nfeatures: int = 3000) -> list[tuple[str, float]]:
    q = cv2.imread(query_path, cv2.IMREAD_GRAYSCALE)
    if q is None:
        raise ValueError(f"Cannot read query image: {query_path}")

    if max_side and max(q.shape[:2]) > max_side:
        scale = max_side / max(q.shape[:2])
        q = cv2.resize(q, (int(q.shape[1] * scale), int(q.shape[0] * scale)))

    orb = cv2.ORB_create(nfeatures=nfeatures)
    qk, qd = orb.detectAndCompute(q, None)
    if qd is None or len(qk) == 0:
        return [(p, 0.0) for p in candidate_paths]

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    scored: list[tuple[str, float]] = []
    for p in candidate_paths:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            scored.append((p, 0.0))
            continue

        if max_side and max(img.shape[:2]) > max_side:
            scale = max_side / max(img.shape[:2])
            img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))

        k, d = orb.detectAndCompute(img, None)
        if d is None or len(k) == 0:
            scored.append((p, 0.0))
            continue

        matches = bf.knnMatch(qd, d, k=2)
        good = []
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < 0.75 * n.distance:
                    good.append(m)
        denom = max(1, min(len(qd), len(d)))
        score = (len(good) / denom) * 100.0
        score = float(max(0.0, min(100.0, score)))
        scored.append((p, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored

def search(query: str, index_dir: str, top: int = 10, rerank: int = 50):
    meta_path = os.path.join(index_dir, "meta.json")
    paths_path = os.path.join(index_dir, "paths.jsonl")
    hashes_path = os.path.join(index_dir, "hashes.npy")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    hash_size = int(meta["hash_size"])
    max_side = meta.get("max_side", 1024)

    paths = load_paths_jsonl(paths_path)
    db_hashes = np.load(hashes_path, mmap_mode="r")

    qhash = phash_packed_bytes(query, hash_size=hash_size, max_side=max_side)
    dists = hamming_distances_packed(qhash, db_hashes)

    k = max(top, rerank if rerank else top)
    k = min(k, len(dists))

    idx = np.argpartition(dists, kth=k - 1)[:k]
    idx = idx[np.argsort(dists[idx])]

    results = []
    for i in idx:
        dist = int(dists[i])
        results.append({
            "path": paths[int(i)],
            "phash_distance_bits": dist,
            "phash_similarity_pct": dist_to_percent(dist, hash_size),
        })

    if rerank and len(results) > 0:
        cand = results[:rerank]
        cand_paths = [r["path"] for r in cand]
        orb_scored = orb_rerank(query, cand_paths, max_side=1024, nfeatures=3000)
        orb_map = {p: s for p, s in orb_scored}

        for r in results:
            r["orb_score_pct"] = orb_map.get(r["path"], 0.0)

        # Combine scores: 70% pHash, 30% ORB
        for r in results:
            r["combined_score"] = 0.7 * r["phash_similarity_pct"] + 0.3 * r["orb_score_pct"]

        results.sort(key=lambda r: r["combined_score"], reverse=True)
    else:
        for r in results:
            r["orb_score_pct"] = None
            r["combined_score"] = r["phash_similarity_pct"]

        results.sort(key=lambda r: r["phash_similarity_pct"], reverse=True)

    return results[:top]

def run_search_with_defaults():
    query_image_path = "test_image/lavender_garden.jpg"  # Specify your query image path here
    index_dir = "index"  # Specify your index directory here
    top_results = 5  # Number of top results to return
    rerank_candidates = 5  # Number of candidates to rerank with ORB

    res = search(query_image_path, index_dir, top=top_results, rerank=rerank_candidates)

    # show top 5 match
    print(f"Query: {os.path.abspath(query_image_path)}")
    print(f"Top {len(res)} matches:")
    for n, r in enumerate(res, 1):
        orb = r["orb_score_pct"]
        combined = r["combined_score"]
        orb_str = f"{orb:6.2f}%" if orb is not None else "  n/a "
        print(f"{n:02d}. {r['phash_similarity_pct']:6.2f}% (pHash dist {r['phash_distance_bits']:3d}) ORB {orb_str} Combined {combined:6.2f}% -> {r['path']}")

    # Open the query image and the top match
    if res:
        import subprocess
        try:
            subprocess.run(["xdg-open", query_image_path], check=True)
            subprocess.run(["xdg-open", res[0]['path']], check=True)
        except subprocess.CalledProcessError:
            print("Could not open images automatically. Please open them manually:")
            print(f"Query image: {os.path.abspath(query_image_path)}")
            print(f"Top match: {os.path.abspath(res[0]['path'])}")

if __name__ == "__main__":
    run_search_with_defaults()
