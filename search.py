"""
Simple Image Search Runner
This is the main script you run to search for similar images.
The actual search algorithms are in search_engine.py and utils.py
"""

import argparse
import os
from log_utils import log_info, log_ok, log_section, log_warn
from search_engine import search  # Import the core search function
from utils import classify_image_content  # For image content analysis
from visualization import show_side_by_side_comparison  # Image comparison display
from negative_search import handle_negative_search_result  # Negative search detection
from positive_search import handle_positive_search_result  # Positive search detection
from build_index import needs_reindex, build_index  # Auto-indexing


def run_search_with_defaults(
    query_path: str = "test_image/asha2-5R.jpg",
    index_folder: str = "index",
    limit: int = 10,
    filter_size: int = 100,
    source_folder: str = "image_database/stored_image",
    similarity_threshold: float = 50.0,
):
    """
    Main function to run an image search with default settings.
    Modify these values to search for different images or adjust search behavior.
    """
    # =========================================================================
    # STEP 0: Auto-indexing - Check if index needs to be updated
    # =========================================================================
    needs_update, changes = needs_reindex(source_folder, index_folder)
    
    if needs_update:
        log_section("INDEX STATUS")
        if 'reason' in changes:
            log_info(f"Indexing required: {changes['reason']}")
        else:
            messages = []
            if changes['new'] > 0:
                messages.append(f" {changes['new']} new image(s) added")
            if changes['renamed'] > 0:
                messages.append(f" {changes['renamed']} image(s) renamed")
            if changes['deleted'] > 0 and changes['renamed'] == 0:
                messages.append(f"  {changes['deleted']} image(s) deleted")
            log_info("Indexing required:")
            for msg in messages:
                log_info(msg.strip())
        build_index(source_folder, index_folder, hash_size=32, max_size=None)
        log_ok("Indexing completed - ready to search")
    else:
        log_section("INDEX STATUS")
        log_ok(f"Index up to date - {changes['reason']}")

    # =========================================================================
    # STEP 1: Analyze what's in the query image
    # =========================================================================
    # Identify the content (e.g., "eyes", "animal", "flower")
    # This helps understand what the deep learning model is looking for
    log_section("QUERY IMAGE CONTENT ANALYSIS")
    try:
        # Get top 5 predictions about what's in the image
        predictions = classify_image_content(query_path, top_k=5)
        log_info(f"Query image: {os.path.abspath(query_path)}")
        log_info("Detected content:")
        for i, (label, confidence) in enumerate(predictions, 1):
            log_info(f"{i}. {label.replace('_', ' ').title()}: {confidence:.1f}%")
    except Exception as e:
        log_warn(f"Could not classify image: {e}")

    # =========================================================================
    # STEP 2: Run the 3-stage search pipeline
    # =========================================================================
    results = search(query_path, index_folder, num_results=limit, candidates=filter_size)
    # this search function is from search_engine.py

    # =========================================================================
    # STEP 3: Display results with all scores
    # =========================================================================
    log_section("TOP MATCHING IMAGES")

    # Check for negative search result (no good matches)
    is_negative_match = handle_negative_search_result(results, query_path, similarity_threshold=similarity_threshold)

    # If it IS a good match, show a green popup (similar to the red popup for negative search)
    if not is_negative_match:
        handle_positive_search_result(results, query_path, similarity_threshold=similarity_threshold)
    
    log_info(f"Top {len(results)} matches")
    log_info("Score breakdown:")
    log_info("- Deep: Content similarity from neural network")
    log_info("- Hash: Average hash (aHash) similarity (structure)")
    log_info("- ORB: Keypoint matching (geometric similarity)")
    log_info("- Combined: Final score (33% each)")
    
    for rank, match in enumerate(results, 1):
        orb_score = match["orb_score_pct"]
        deep_score = match.get("deep_score_pct", 0.0)
        final_score = match["combined_score"]
        orb_text = f"{orb_score:5.1f}%" if orb_score is not None else " n/a "
        deep_text = f"{deep_score:5.1f}%"
        
        # Highlight results based on the 50% threshold
        if final_score >= similarity_threshold:
            log_ok(f"{rank:02d}. Hash:{match['hash_similarity_pct']:5.1f}% Deep:{deep_text} ORB:{orb_text} => Combined:{final_score:5.1f}% | {match['path']}")
        else:
            log_warn(f"{rank:02d}. Hash:{match['hash_similarity_pct']:5.1f}% Deep:{deep_text} ORB:{orb_text} => Combined:{final_score:5.1f}% | {match['path']}")

    # =========================================================================
    # STEP 4: Show side-by-side comparison of query and best match
    # =========================================================================
    if results:
        log_section("VISUAL COMPARISON")
        
        # Get similarity score from best match
        best_match_score = results[0]['combined_score']
        
        # Display message
        if not is_negative_match:
            log_info("Displaying query image vs. best match side by side")
        
        # Display side-by-side comparison using OpenCV with difference visualization
        show_side_by_side_comparison(query_path, results[0]['path'], 
                                     similarity_score=best_match_score,
                                     title="Query vs Best Match")


def parse_args() -> argparse.Namespace:
    """Parse optional CLI args.

    Defaults intentionally match previous hardcoded values (no behavior change
    when running `python search.py` without arguments).
    """
    parser = argparse.ArgumentParser(description="Run image similarity search")
    parser.add_argument("--query", default="test_image/asha2-5R.jpg", help="Path to query image")
    parser.add_argument("--index-folder", default="index", help="Index folder path")
    parser.add_argument("--source-folder", default="image_database/stored_image", help="Image database folder")
    parser.add_argument("--limit", type=int, default=10, help="Number of results to return")
    parser.add_argument("--filter-size", type=int, default=100, help="Stage-1 candidate count")
    parser.add_argument("--similarity-threshold", type=float, default=50.0, help="Threshold for positive/negative result")
    return parser.parse_args()

# Entry point: Run the search when this script is executed directly
if __name__ == "__main__":
    args = parse_args()
    run_search_with_defaults(
        query_path=args.query,
        index_folder=args.index_folder,
        limit=args.limit,
        filter_size=args.filter_size,
        source_folder=args.source_folder,
        similarity_threshold=args.similarity_threshold,
    )
