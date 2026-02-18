"""
Positive Search Detection Module
Handles detection and display of match scenarios when a good match is found.

This mirrors negative_search.py but for successful matches.
"""

import os

import cv2
import numpy as np


def _format_file_size(num_bytes: int) -> str:
    if num_bytes < 0:
        return "unknown"
    units = ["B", "KB", "MB", "GB"]
    size = float(num_bytes)
    unit = units[0]
    for u in units:
        unit = u
        if size < 1024.0 or u == units[-1]:
            break
        size /= 1024.0
    if unit == "B":
        return f"{int(size)} {unit}"
    return f"{size:.1f} {unit}"


def _get_image_info(path: str) -> tuple[str, str]:
    """Return (size_str, resolution_str) for an image path."""
    try:
        size_str = _format_file_size(os.path.getsize(path)) if path and os.path.exists(path) else "missing"
    except Exception:
        size_str = "unknown"

    try:
        img = cv2.imread(path)
        if img is None:
            return size_str, "unreadable"
        h, w = img.shape[:2]
        return size_str, f"{w}x{h}"
    except Exception:
        return size_str, "unknown"


def check_positive_search(search_results, similarity_threshold=50.0):
    """Check if search results indicate a positive match (good match found).

    Args:
        search_results: List of search results with combined_score
        similarity_threshold: Minimum similarity threshold (default 50%)

    Returns:
        tuple: (is_positive_match, best_similarity_score)
    """
    if not search_results:
        return False, 0.0

    best_similarity_score = float(search_results[0].get("combined_score", 0.0))
    is_positive_match = best_similarity_score >= float(similarity_threshold)
    return is_positive_match, best_similarity_score


def display_console_success(best_similarity_score: float):
    """Display console message when a matching image is found."""
    print(f"\n✅ MATCH FOUND ✅")
    print(f"Best match similarity: {best_similarity_score:.1f}%")


def show_match_popup(best_similarity_score=0.0, query_image_path="", match_image_path=""):
    """Display a visual success popup when a good match is found."""
    popup_window_width = 1200
    popup_window_height = 500
    popup = np.zeros((popup_window_height, popup_window_width, 3), dtype=np.uint8)

    # Green background
    popup[:] = (0, 140, 0)  # Dark green in BGR

    # White border
    cv2.rectangle(popup, (10, 10), (popup_window_width - 10, popup_window_height - 10), (255, 255, 255), 5)

    font = cv2.FONT_HERSHEY_SIMPLEX

    title = "MATCH FOUND!"
    cv2.putText(popup, title, (50, 80), font, 1.4, (255, 255, 255), 3)

    query_name = os.path.basename(query_image_path) if query_image_path else "(unknown)"
    match_name = os.path.basename(match_image_path) if match_image_path else "(unknown)"

    query_size, query_res = _get_image_info(query_image_path)
    match_size, match_res = _get_image_info(match_image_path)

    lines = [
        f"Best match similarity: {best_similarity_score:.1f}%",
        "",
        f"Query: {query_name}",
        f"  Size: {query_size} | Resolution: {query_res}",
        f"Match: {match_name}",
        f"  Size: {match_size} | Resolution: {match_res}",
        "",
        "Press any key to close...",
    ]

    y = 160
    for line in lines:
        if line == "":
            y += 12
            continue
        cv2.putText(popup, line, (80, y), font, 0.8, (255, 255, 255), 2)
        y += 45

    cv2.imshow("Search Result - Match Found", popup)
    cv2.waitKey(0)
    cv2.destroyWindow("Search Result - Match Found")


def handle_positive_search_result(search_results, query_image_path, similarity_threshold=50.0):
    """Complete handler for positive search results.

    Shows a green popup when the best match is >= threshold.

    Returns:
        bool: True if positive match detected, False otherwise
    """
    is_positive_match, best_similarity_score = check_positive_search(search_results, similarity_threshold)

    if is_positive_match and search_results:
        display_console_success(best_similarity_score)
        best_match_path = search_results[0].get("path", "")
        show_match_popup(best_similarity_score, query_image_path, best_match_path)

    return is_positive_match
