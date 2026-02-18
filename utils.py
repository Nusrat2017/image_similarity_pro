import os
import json
import numpy as np
from PIL import Image
import imagehash
import cv2
import torch
import torchvision.models as models
import torchvision.transforms as transforms

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

# =============================================================================
# HELPER FUNCTIONS - Common utilities used by all algorithms
# =============================================================================

def iter_images(folder: str):
    """Iterate through all image files in a folder recursively."""
    for root, _, files in os.walk(folder):
        for name in files:
            ext = os.path.splitext(name.lower())[1]
            if ext in IMAGE_EXTS:
                yield os.path.join(root, name)

def open_image_rgb(path: str) -> Image.Image:
    """Open an image and convert to RGB format."""
    with Image.open(path) as img:
        return img.convert("RGB")

def resize_max_side(img: Image.Image, max_side: int | None) -> Image.Image:
    """Resize image so that the longest side is max_side pixels."""
    if not max_side:
        return img
    width, height = img.size
    max_dimension = max(width, height)
    if max_dimension <= max_side:
        return img
    scale = max_side / max_dimension
    new_width, new_height = int(round(width * scale)), int(round(height * scale))
    return img.resize((new_width, new_height))

def load_paths_jsonl(paths_file: str) -> list[str]:
    """Load image paths from JSONL file."""
    paths: list[str] = []
    with open(paths_file, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            paths.append(obj["path"])
    return paths

def save_paths_jsonl(paths_file: str, paths: list[str]):
    """Save image paths to JSONL file."""
    with open(paths_file, "w", encoding="utf-8") as f:
        for p in paths:
            f.write(json.dumps({"path": p}, ensure_ascii=False) + "\n")

# =============================================================================
# ALGORITHM 1: HASH-BASED COMPARISON (Average Hash)
# Fast initial filtering using perceptual hashing
# Weight: 25% in final score
# =============================================================================

# Precompute bit count for bytes 0..255 (fast Hamming distance for packed uint8 arrays)
BIT_COUNT_LOOKUP_TABLE = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

def phash_packed_bytes(image_path: str, hash_size: int = 16, max_side: int | None = None) -> np.ndarray:
    """Return packed bytes of pHash.
    Output shape: (hash_size*hash_size/8,) uint8.
    For hash_size=16 => 256 bits => 32 bytes.
    """
    img = open_image_rgb(image_path)
    img = resize_max_side(img, max_side)
    img_hash = imagehash.average_hash(img, hash_size=hash_size)  # Average hash - works better for color images
    bits = np.asarray(img_hash.hash, dtype=np.uint8).reshape(-1)  # 0/1
    packed = np.packbits(bits)  # uint8
    return packed

def hamming_distances_packed(query_hash: np.ndarray, stored_hashes: np.ndarray) -> np.ndarray:
    """Vectorized Hamming distance for packed uint8 hashes.
    query_hash: (B,)
    stored_hashes:    (N,B)
    returns: (N,) bit distances
    """
    diff = np.bitwise_xor(stored_hashes, query_hash)  # (N,B)
    return BIT_COUNT_LOOKUP_TABLE[diff].sum(axis=1).astype(np.int32)

def dist_to_percent(dist_bits: int, hash_size: int) -> float:
    """Convert bit distance to similarity percentage."""
    total_bits = hash_size * hash_size
    return max(0.0, (1.0 - dist_bits / total_bits) * 100.0)

# =============================================================================
# ALGORITHM 2: ORB FEATURE MATCHING
# Oriented FAST and Rotated BRIEF - Keypoint-based matching
# Weight: 20% in final score
# =============================================================================

def orb_rerank(query_path: str, candidate_paths: list[str], max_side: int = 1024, nfeatures: int = 5000) -> list[tuple[str, float]]:
    """Rerank candidate images using ORB feature matching."""
    query_img = cv2.imread(query_path, cv2.IMREAD_GRAYSCALE)
    if query_img is None:
        raise ValueError(f"Cannot read query image: {query_path}")

    if max_side and max(query_img.shape[:2]) > max_side:
        scale = max_side / max(query_img.shape[:2])
        query_img = cv2.resize(query_img, (int(query_img.shape[1] * scale), int(query_img.shape[0] * scale)))

    detector = cv2.ORB_create(nfeatures=nfeatures, scaleFactor=1.2, nlevels=8, edgeThreshold=15, patchSize=31)
    query_kp, query_desc = detector.detectAndCompute(query_img, None)
    if query_desc is None or len(query_kp) == 0:
        return [(path, 0.0) for path in candidate_paths]

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    results: list[tuple[str, float]] = []
    for path in candidate_paths:
        candidate = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if candidate is None:
            results.append((path, 0.0))
            continue

        if max_side and max(candidate.shape[:2]) > max_side:
            scale = max_side / max(candidate.shape[:2])
            candidate = cv2.resize(candidate, (int(candidate.shape[1] * scale), int(candidate.shape[0] * scale)))

        kp, desc = detector.detectAndCompute(candidate, None)
        if desc is None or len(kp) == 0:
            results.append((path, 0.0))
            continue

        all_matches = matcher.knnMatch(query_desc, desc, k=2)
        good = []
        for pair in all_matches:
            if len(pair) == 2:
                best, second = pair
                if best.distance < 0.7 * second.distance:
                    good.append(best)
        min_count = max(1, min(len(query_desc), len(desc)))
        score = (len(good) / min_count) * 100.0
        score = float(max(0.0, min(100.0, score)))
        results.append((path, score))

    results.sort(key=lambda x: x[1], reverse=True)
    return results


def orb_rerank_with_info(
    query_path: str,
    candidate_paths: list[str],
    max_side: int = 1024,
    nfeatures: int = 5000,
) -> tuple[list[tuple[str, float]], bool]:
    """Like orb_rerank(), but also reports whether the query had usable ORB features.

    For very small / low-detail query images ORB may produce zero keypoints.
    In that case, returning a 0% ORB score for everything can incorrectly drag
    down the combined similarity score. This helper lets the caller adapt the
    weighting when ORB is not applicable.

    Returns:
        (results, query_has_features)
    """
    query_img = cv2.imread(query_path, cv2.IMREAD_GRAYSCALE)
    if query_img is None:
        raise ValueError(f"Cannot read query image: {query_path}")

    if max_side and max(query_img.shape[:2]) > max_side:
        scale = max_side / max(query_img.shape[:2])
        query_img = cv2.resize(query_img, (int(query_img.shape[1] * scale), int(query_img.shape[0] * scale)))

    detector = cv2.ORB_create(nfeatures=nfeatures, scaleFactor=1.2, nlevels=8, edgeThreshold=15, patchSize=31)
    query_kp, query_desc = detector.detectAndCompute(query_img, None)
    if query_desc is None or len(query_kp) == 0:
        return ([(path, 0.0) for path in candidate_paths], False)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    results: list[tuple[str, float]] = []
    for path in candidate_paths:
        candidate = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if candidate is None:
            results.append((path, 0.0))
            continue

        if max_side and max(candidate.shape[:2]) > max_side:
            scale = max_side / max(candidate.shape[:2])
            candidate = cv2.resize(candidate, (int(candidate.shape[1] * scale), int(candidate.shape[0] * scale)))

        kp, desc = detector.detectAndCompute(candidate, None)
        if desc is None or len(kp) == 0:
            results.append((path, 0.0))
            continue

        all_matches = matcher.knnMatch(query_desc, desc, k=2)
        good = []
        for pair in all_matches:
            if len(pair) == 2:
                best, second = pair
                if best.distance < 0.7 * second.distance:
                    good.append(best)
        min_count = max(1, min(len(query_desc), len(desc)))
        score = (len(good) / min_count) * 100.0
        score = float(max(0.0, min(100.0, score)))
        results.append((path, score))

    results.sort(key=lambda x: x[1], reverse=True)
    return (results, True)

# =============================================================================
# ALGORITHM 3: DEEP LEARNING (ResNet18)
# Content-aware feature extraction using pre-trained neural network
# Weight: 55% in final score (MOST IMPORTANT)
# =============================================================================

# Global model for deep learning features (lazy loaded)
_deep_model = None
_deep_model_full = None  # Full model with classification head
_deep_transform = None

def get_deep_learning_model():
    """Get or initialize the ResNet model for feature extraction."""
    global _deep_model, _deep_model_full, _deep_transform
    if _deep_model is None:
        # Disable SSL verification for model download
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        
        # Use ResNet18 pre-trained on ImageNet
        full_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        _deep_model_full = full_model  # Keep full model for classification
        _deep_model_full.eval()
        
        # Remove the final classification layer to get features
        _deep_model = torch.nn.Sequential(*list(full_model.children())[:-1])
        _deep_model.eval()
        
        # Standard ImageNet preprocessing
        _deep_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return _deep_model, _deep_transform

def extract_deep_features(image_path: str) -> np.ndarray:
    """Extract deep learning features from an image using ResNet."""
    model, transform = get_deep_learning_model()
    img = open_image_rgb(image_path)
    tensor = transform(img).unsqueeze(0)  # Add batch dimension
    
    with torch.no_grad():
        output = model(tensor)
    
    # Flatten to 1D array
    vector = output.squeeze().numpy()
    # Normalize
    normalized = vector / (np.linalg.norm(vector) + 1e-7)
    return normalized

def compare_deep_features(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compare deep features using cosine similarity. Returns similarity 0-100%."""
    similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-7)
    # Convert from [-1, 1] to [0, 100]
    percentage = ((similarity + 1) / 2) * 100
    return max(0.0, min(100.0, percentage))

def classify_image_content(image_path: str, top_k: int = 5) -> list[tuple[str, float]]:
    """Classify image content using ResNet18 pre-trained on ImageNet.
    
    Returns top_k predictions as list of (label, confidence_percentage) tuples.
    Example: [('golden_retriever', 85.2), ('Labrador_retriever', 8.3), ...]
    """
    global _deep_model_full, _deep_transform
    
    # Initialize model if needed
    if _deep_model_full is None:
        get_deep_learning_model()
    
    # Load and preprocess image
    img = open_image_rgb(image_path)
    tensor = _deep_transform(img).unsqueeze(0)  # Add batch dimension
    
    # Get predictions
    with torch.no_grad():
        logits = _deep_model_full(tensor)
        probs = torch.nn.functional.softmax(logits[0], dim=0)
    
    # Get top k predictions
    top_probs, top_idx = torch.topk(probs, top_k)
    
    # Get ImageNet class labels
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    labels = weights.meta["categories"]
    
    predictions = []
    for i in range(top_k):
        idx = top_idx[i].item()
        conf = top_probs[i].item() * 100
        name = labels[idx]
        predictions.append((name, conf))
    
    return predictions
