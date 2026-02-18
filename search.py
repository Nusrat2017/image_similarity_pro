"""
Simple Image Search Runner
This is the main script you run to search for similar images.
The actual search algorithms are in search_engine.py and utils.py
"""

import os
from search_engine import search  # Import the core search function
from utils import classify_image_content  # For image content analysis
from visualization import show_side_by_side_comparison  # Image comparison display
from negative_search import handle_negative_search_result  # Negative search detection
from positive_search import handle_positive_search_result  # Positive search detection
from build_index import needs_reindex, build_index  # Auto-indexing


def run_search_with_defaults():
    """
    Main function to run an image search with default settings.
    Modify these values to search for different images or adjust search behavior.
    """
    # =========================================================================
    # SEARCH SETTINGS - Change these to customize your search
    # =========================================================================
    query_path = "test_image/human_ear.jpg"  # Which image to search for
    index_folder = "index"  # Where the index files are stored
    limit = 10  # How many results to show
    filter_size = 100  # How many candidates to check in Stage 1 (higher = more thorough)
    
    # =========================================================================
    # STEP 0: Auto-indexing - Check if index needs to be updated
    # =========================================================================
    source_folder = "image_database/stored_image"
    needs_update, changes = needs_reindex(source_folder, index_folder)
    
    if needs_update:
        print(f"\n{'='*70}")
        if 'reason' in changes:
            print(f"🔄 INDEXING REQUIRED: {changes['reason']}")
        else:
            messages = []
            if changes['new'] > 0:
                messages.append(f"📥 {changes['new']} new image(s) added")
            if changes['renamed'] > 0:
                messages.append(f"📝 {changes['renamed']} image(s) renamed")
            if changes['deleted'] > 0 and changes['renamed'] == 0:
                messages.append(f"🗑️  {changes['deleted']} image(s) deleted")
            print(f"🔄 INDEXING REQUIRED:")
            for msg in messages:
                print(f"   {msg}")
        print(f"{'='*70}")
        build_index(source_folder, index_folder, hash_size=32, max_size=None)
        print(f"{'='*70}")
        print(f"✅ INDEXING COMPLETED - Ready to search")
        print(f"{'='*70}\n")
    else:
        print(f"\n{'='*70}")
        print(f"✅ INDEX UP TO DATE - {changes['reason']}")
        print(f"{'='*70}\n")

    # =========================================================================
    # STEP 1: Analyze what's in the query image
    # =========================================================================
    # Identify the content (e.g., "eyes", "animal", "flower")
    # This helps understand what the deep learning model is looking for
    print(f"\n{'='*70}")
    print(f"QUERY IMAGE CONTENT ANALYSIS:")
    print(f"{'='*70}")
    try:
        # Get top 5 predictions about what's in the image
        predictions = classify_image_content(query_path, top_k=5)
        print(f"Query image: {os.path.abspath(query_path)}")
        print(f"\nDetected content:")
        for i, (label, confidence) in enumerate(predictions, 1):
            print(f"  {i}. {label.replace('_', ' ').title()}: {confidence:.1f}%")
    except Exception as e:
        print(f"Could not classify image: {e}")
    print(f"{'='*70}\n")

    # =========================================================================
    # STEP 2: Run the 3-stage search pipeline
    # =========================================================================
    results = search(query_path, index_folder, num_results=limit, candidates=filter_size)
    # this search function is from search_engine.py

    # =========================================================================
    # STEP 3: Display results with all scores
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"TOP MATCHING IMAGES:")
    print(f"{'='*70}")
    
    similarity_threshold = 50.0

    # Check for negative search result (no good matches)
    is_negative_match = handle_negative_search_result(results, query_path, similarity_threshold=similarity_threshold)

    # If it IS a good match, show a green popup (similar to the red popup for negative search)
    if not is_negative_match:
        handle_positive_search_result(results, query_path, similarity_threshold=similarity_threshold)
    
    print(f"Top {len(results)} matches:")
    print(f"\nScore breakdown:")
    print(f"  - Deep: Content similarity from neural network")
    print(f"  - Hash: Perceptual hash similarity (structure)")
    print(f"  - ORB: Keypoint matching (geometric similarity)")
    print(f"  - Combined: Final score (33% each)\n")
    
    for rank, match in enumerate(results, 1):
        orb_score = match["orb_score_pct"]
        deep_score = match.get("deep_score_pct", 0.0)
        final_score = match["combined_score"]
        orb_text = f"{orb_score:5.1f}%" if orb_score is not None else " n/a "
        deep_text = f"{deep_score:5.1f}%"
        
        # Highlight results based on the 50% threshold
        prefix = "✅ " if final_score >= similarity_threshold else "❌ "
        print(f"{prefix}{rank:02d}. Hash:{match['phash_similarity_pct']:5.1f}% Deep:{deep_text} ORB:{orb_text} => Combined:{final_score:5.1f}% | {match['path']}")

    # =========================================================================
    # STEP 4: Show side-by-side comparison of query and best match
    # =========================================================================
    if results:
        print(f"\n{'='*70}")
        print(f"VISUAL COMPARISON:")
        print(f"{'='*70}")
        
        # Get similarity score from best match
        best_match_score = results[0]['combined_score']
        
        # Display message
        if not is_negative_match:
            print(f"Displaying query image vs. best match side by side...")
        
        # Display side-by-side comparison using OpenCV with difference visualization
        show_side_by_side_comparison(query_path, results[0]['path'], 
                                     similarity_score=best_match_score,
                                     title="Query vs Best Match")

# Entry point: Run the search when this script is executed directly
if __name__ == "__main__":
    run_search_with_defaults()
