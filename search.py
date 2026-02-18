import os
import json
import numpy as np

from utils import (
    phash_packed_bytes,
    hamming_distances_packed,
    dist_to_percent,
    load_paths_jsonl,
    extract_deep_features,
    compare_deep_features,
    orb_rerank,
)

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
        orb_scored = orb_rerank(query, cand_paths, max_side=1024, nfeatures=5000)
        orb_map = {p: s for p, s in orb_scored}

        for r in results:
            r["orb_score_pct"] = orb_map.get(r["path"], 0.0)

        # Add deep learning feature comparison
        try:
            print("Extracting deep learning features...")
            query_features = extract_deep_features(query)
            for r in results[:rerank]:
                try:
                    img_features = extract_deep_features(r["path"])
                    r["deep_score_pct"] = compare_deep_features(query_features, img_features)
                except:
                    r["deep_score_pct"] = 0.0
        except Exception as e:
            print(f"Deep learning feature extraction failed: {e}")
            for r in results[:rerank]:
                r["deep_score_pct"] = 0.0

        # Combine scores: 25% pHash, 20% ORB, 55% Deep Learning (content-aware)
        for r in results[:rerank]:
            deep_score = r.get("deep_score_pct", 0.0)  
            r["combined_score"] = 0.25 * r["phash_similarity_pct"] + 0.2 * r["orb_score_pct"] + 0.55 * deep_score
        
        # For non-reranked items, use just pHash score
        for r in results[rerank:]:
            r["deep_score_pct"] = 0.0
            r["combined_score"] = r["phash_similarity_pct"]

        # Sort all results by combined score (highest first)
        results.sort(key=lambda r: r["combined_score"], reverse=True)
    else:
        for r in results:
            r["orb_score_pct"] = None
            r["deep_score_pct"] = 0.0
            r["combined_score"] = r["phash_similarity_pct"]

        results.sort(key=lambda r: r["phash_similarity_pct"], reverse=True)

    return results[:top]

def run_search_with_defaults():
    query_image_path = "test_image/asha2-50R-Fade.png"  # Specify your query image path here
    index_dir = "index"  # Specify your index directory here
    top_results = 5  # Number of top results to return
    rerank_candidates = 20  # Number of candidates to rerank with ORB

    res = search(query_image_path, index_dir, top=top_results, rerank=rerank_candidates)

    # show top 5 match
    print(f"Query: {os.path.abspath(query_image_path)}")
    print(f"Top {len(res)} matches:")
    for n, r in enumerate(res, 1):
        orb = r["orb_score_pct"]
        deep = r.get("deep_score_pct", 0.0)
        combined = r["combined_score"]
        orb_str = f"{orb:5.1f}%" if orb is not None else " n/a "
        deep_str = f"{deep:5.1f}%"
        print(f"{n:02d}. Hash:{r['phash_similarity_pct']:5.1f}% Deep:{deep_str} ORB:{orb_str} => Combined:{combined:5.1f}% | {r['path']}")

    # Open the query image and the top match
    if res:
        import subprocess
        import sys
        try:
            if sys.platform == "win32":
                subprocess.run(["cmd", "/c", "start", "", query_image_path], check=True)
                subprocess.run(["cmd", "/c", "start", "", res[0]['path']], check=True)
            else:
                subprocess.run(["xdg-open", query_image_path], check=True)
                subprocess.run(["xdg-open", res[0]['path']], check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Could not open images automatically. Please open them manually:")
            print(f"Query image: {os.path.abspath(query_image_path)}")
            print(f"Top match: {os.path.abspath(res[0]['path'])}")

if __name__ == "__main__":
    run_search_with_defaults()
