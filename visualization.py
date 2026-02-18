"""
Image Visualization Module
Side-by-side comparison and difference display functions for the image search system.
"""

import os
import cv2
import numpy as np


def show_side_by_side_comparison(query_image_path, match_image_path, similarity_score=100.0, title="Image Comparison"):
    """
    Display two images side by side with OpenCV for visual comparison.
    Shows a difference map when similarity is not 100%.
    
    Args:
        query_image_path: Path to the first image (query image)
        match_image_path: Path to the second image (matching image)
        similarity_score: Similarity percentage (0-100)
        title: Window title
    """
    # Load both images
    query_image = cv2.imread(query_image_path)
    match_image = cv2.imread(match_image_path)
    
    if query_image is None or match_image is None:
        print(f"Error: Could not load images for comparison")
        return

    def _shorten_label(text: str, max_chars: int = 28) -> str:
        text = str(text)
        if len(text) <= max_chars:
            return text
        return text[: max(0, max_chars - 1)] + "…"

    def _put_bottom_right_label(img: np.ndarray, label: str, *, pad: int = 8) -> np.ndarray:
        """Draw a readable label in the bottom-right corner of a BGR image."""
        label = _shorten_label(label)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        thickness = 2
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        h, w = img.shape[:2]
        x2 = max(pad, w - pad)
        y2 = max(pad, h - pad)
        x1 = max(pad, x2 - tw - (pad * 2))
        y1 = max(pad, y2 - th - baseline - (pad * 2))

        # Background rectangle for contrast
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), -1)
        # Text
        cv2.putText(img, label, (x1 + pad, y2 - pad), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        return img
    
    # Set maximum display dimensions to fit on screen (adjust as needed)
    # Keep the old vertical sizing behavior, but reduce horizontal overflow by
    # constraining each of the 3 side-by-side panels to a max width.
    max_display_height = 800  # Maximum height for the comparison window
    max_display_width = 1400  # Maximum total width for the comparison window

    divider_w = 5

    # Choose a target height that keeps Query | Match | Heatmap within max_display_width.
    # We keep aspect ratio for query/match by shrinking target_height if needed.
    panel_w = max(1, (max_display_width - (2 * divider_w)) // 3)

    def _height_limit(img, max_w: int) -> float:
        h, w = img.shape[:2]
        return (h * max_w) / max(1, w)

    target_height = min(
        max(query_image.shape[0], match_image.shape[0]),
        max_display_height,
        int(_height_limit(query_image, panel_w)),
        int(_height_limit(match_image, panel_w)),
    )
    target_height = max(200, int(target_height))

    # Calculate new widths maintaining aspect ratio at the chosen height
    query_width = max(1, int(query_image.shape[1] * target_height / max(1, query_image.shape[0])))
    match_width = max(1, int(match_image.shape[1] * target_height / max(1, match_image.shape[0])))

    # Resize images
    query_resized = cv2.resize(query_image, (query_width, target_height))
    match_resized = cv2.resize(match_image, (match_width, target_height))
    
    # If not 100% match, create a difference visualization
    if similarity_score < 100.0:
        # Resize match image to exact same size as query for pixel-wise comparison
        match_for_diff = cv2.resize(match_image, (query_resized.shape[1], query_resized.shape[0]))
        
        # Calculate absolute difference between images
        difference = cv2.absdiff(query_resized, match_for_diff)
        
        # Convert to grayscale for better visualization
        difference_gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
        
        # Threshold to highlight significant differences (adjust threshold as needed)
        _, threshold_diff = cv2.threshold(difference_gray, 30, 255, cv2.THRESH_BINARY)
        
        # Create matching areas mask (inverse of differences)
        matching_mask = cv2.bitwise_not(threshold_diff)
        
        # Create a heatmap of differences (red = different, black = same)
        difference_heatmap = cv2.applyColorMap(difference_gray, cv2.COLORMAP_HOT)
        
        # Create highlighted versions showing both matches (green) and differences (red)
        query_highlighted = query_resized.copy()
        match_highlighted = match_for_diff.copy()
        
        # Find contours for differences (red outline)
        diff_contours, _ = cv2.findContours(threshold_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find contours for matching areas (green overlay)
        match_contours, _ = cv2.findContours(matching_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create green overlay for matching areas
        green_overlay_query = query_resized.copy()
        green_overlay_match = match_for_diff.copy()
        cv2.drawContours(green_overlay_query, match_contours, -1, (0, 255, 0), -1)  # Fill with green
        cv2.drawContours(green_overlay_match, match_contours, -1, (0, 255, 0), -1)  # Fill with green
        
        # Blend the green overlay with original images (30% green, 70% original)
        query_highlighted = cv2.addWeighted(query_resized, 0.7, green_overlay_query, 0.3, 0)
        match_highlighted = cv2.addWeighted(match_for_diff, 0.7, green_overlay_match, 0.3, 0)
        
        # Draw red contours for differences on top
        cv2.drawContours(query_highlighted, diff_contours, -1, (0, 0, 255), 2)
        cv2.drawContours(match_highlighted, diff_contours, -1, (0, 0, 255), 2)
        
        # Resize difference panel to a fixed panel width (keeps total window width smaller)
        difference_heatmap_resized = cv2.resize(difference_heatmap, (panel_w, target_height))

        # Add filename labels at the bottom-right of each panel
        _put_bottom_right_label(query_highlighted, os.path.basename(query_image_path))
        _put_bottom_right_label(match_highlighted, os.path.basename(match_image_path))
        _put_bottom_right_label(difference_heatmap_resized, "Differences")

        # Create white dividers
        white_divider = np.ones((target_height, divider_w, 3), dtype=np.uint8) * 255

        # Concatenate: Query | Match | Difference Heatmap
        side_by_side_comparison = np.hstack((
            query_highlighted,
            white_divider,
            match_highlighted,
            white_divider,
            difference_heatmap_resized,
        ))

        # Add text labels
        label_font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(side_by_side_comparison, "Query Image", (10, 30), label_font, 0.8, (255, 255, 255), 2)
        cv2.putText(side_by_side_comparison, "Best Match", (query_width + 15, 30), label_font, 0.8, (255, 255, 255), 2)
        cv2.putText(side_by_side_comparison, "Differences (Hot)", (query_width + match_width + 25, 30), label_font, 0.8, (255, 255, 255), 2)
        cv2.putText(side_by_side_comparison, f"Similarity: {similarity_score:.1f}%", (10, target_height - 10), label_font, 0.8, (0, 255, 255), 2)
        cv2.putText(side_by_side_comparison, "Green=Match Red=Diff", (10, target_height - 40), label_font, 0.6, (255, 255, 255), 2)
        
    else:
        # For 100% match, just show side by side without difference map
        white_divider = np.ones((target_height, divider_w, 3), dtype=np.uint8) * 255

        # Add filename labels at the bottom-right of each panel
        _put_bottom_right_label(query_resized, os.path.basename(query_image_path))
        _put_bottom_right_label(match_resized, os.path.basename(match_image_path))

        side_by_side_comparison = np.hstack((query_resized, white_divider, match_resized))
        
        label_font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(side_by_side_comparison, "Query Image", (10, 30), label_font, 0.8, (0, 255, 0), 2)
        cv2.putText(side_by_side_comparison, "Best Match - 100% Identical!", (query_width + 15, 30), label_font, 0.8, (0, 255, 0), 2)
    
    # Scale down the final comparison if it's too wide to fit on screen
    comparison_height, comparison_width = side_by_side_comparison.shape[:2]
    if comparison_width > max_display_width or comparison_height > max_display_height:
        scale_factor = min(max_display_width / comparison_width, max_display_height / comparison_height)
        new_width = int(comparison_width * scale_factor)
        new_height = int(comparison_height * scale_factor)
        side_by_side_comparison = cv2.resize(side_by_side_comparison, (new_width, new_height))
        print(f"Scaled comparison to fit screen: {new_width}x{new_height}")
    
    # Display the comparison
    cv2.imshow(title, side_by_side_comparison)
    if similarity_score < 100.0:
        print(f"\nSide-by-side comparison with difference map displayed.")
        print(f"Green tinted areas = Matching regions")
        print(f"Red outlines = Different regions")
        print(f"Heatmap: Red/Yellow = Differences, Dark = Same")
    else:
        print(f"\nPerfect match! Images are identical.")
    print(f"Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
