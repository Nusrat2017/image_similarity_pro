"""
Negative Search Detection Module
Handles detection and display of no-match scenarios when query images don't exist in database.
"""

import cv2
import numpy as np


def check_negative_search(search_results, similarity_threshold=50.0):
    """
    Check if search results indicate a negative match (no good matches found).
    
    Args:
        search_results: List of search results with combined_score
        similarity_threshold: Minimum similarity threshold (default 50%)
    
    Returns:
        tuple: (is_negative_match, best_similarity_score) - True if no good match found, best score value
    """
    if not search_results:
        return True, 0.0
    
    best_similarity_score = search_results[0]["combined_score"]
    is_negative_match = best_similarity_score < similarity_threshold
    return is_negative_match, best_similarity_score


def display_console_warning(best_similarity_score):
    """
    Display console warning when no matching images are found.
    
    Args:
        best_similarity_score: The best similarity score found
    """
    print(f"\n⚠️  NO MATCHING IMAGES FOUND ⚠️")
    print(f"\nThe best match is only {best_similarity_score:.1f}% similar.")
    print(f"This suggests the query image does NOT exist in the database.")
    print(f"\nShowing closest matches anyway for reference:\n")


def show_no_match_popup(best_similarity_score=0.0, query_image_path=""):
    """
    Display a visual warning popup when no good matches are found.
    
    Args:
        best_similarity_score: The best similarity score found (below threshold)
        query_image_path: Path to the query image
    """
    # Create a red warning image
    popup_window_width = 1200
    popup_window_height = 500
    warning_popup_image = np.zeros((popup_window_height, popup_window_width, 3), dtype=np.uint8)
    
    # Red background for warning
    warning_popup_image[:] = (0, 0, 180)  # Dark red in BGR
    
    # Add yellow border
    cv2.rectangle(warning_popup_image, (10, 10), (popup_window_width-10, popup_window_height-10), (0, 255, 255), 5)
    
    # Add warning text
    text_font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Title
    warning_title_text = "WARNING: NO MATCHING IMAGE FOUND!"
    cv2.putText(warning_popup_image, warning_title_text, (50, 80), text_font, 1.2, (255, 255, 255), 3)
    
    # Message lines
    warning_messages = [
        f"Best match similarity: {best_similarity_score:.1f}%",
        "",
        "This query image does NOT exist in the database.",
        "",
        "Possible reasons:",
        "  - Image was never indexed",
        "  - Image is significantly different from database",
        "  - Database needs to be rebuilt",
        "",
        "Press any key to close...",
        "",
        "NOTE: This warning is only shown when the best match is below the similarity threshold.",
        "",
    ]
    
    current_y_position = 150
    for message_line in warning_messages:
        if message_line.startswith("  -"):
            cv2.putText(warning_popup_image, message_line, (100, current_y_position), text_font, 0.6, (255, 255, 150), 1)
        elif message_line == "":
            current_y_position += 10
            continue
        else:
            cv2.putText(warning_popup_image, message_line, (80, current_y_position), text_font, 0.7, (255, 255, 255), 2)
        current_y_position += 35
    
    # Display popup
    cv2.imshow("Search Result - No Match", warning_popup_image)
    cv2.waitKey(0)
    cv2.destroyWindow("Search Result - No Match")


def handle_negative_search_result(search_results, query_image_path, similarity_threshold=50.0):
    """
    Complete handler for negative search results.
    Checks if results are below threshold and displays appropriate warnings.
    
    Args:
        search_results: List of search results
        query_image_path: Path to the query image
        similarity_threshold: Minimum similarity threshold (default 50%)
    
    Returns:
        bool: True if negative search detected, False otherwise
    """
    is_negative_match, best_similarity_score = check_negative_search(search_results, similarity_threshold)
    
    if is_negative_match:
        # Display console warning
        display_console_warning(best_similarity_score)
        
        # If there's at least one result, show popup before visual comparison
        if search_results:
            print(f"⚠️  Displaying comparison with LOW similarity match ({best_similarity_score:.1f}%)")
            print(f"    This is likely NOT the same image!")
            show_no_match_popup(best_similarity_score, query_image_path)
    
    return is_negative_match
