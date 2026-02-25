import os

import cv2


def format_file_size(num_bytes: int) -> str:
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


def get_image_info(path: str) -> tuple[str, str]:
    """Return (size_str, resolution_str) for an image path."""
    try:
        size_str = format_file_size(os.path.getsize(path)) if path and os.path.exists(path) else "missing"
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
