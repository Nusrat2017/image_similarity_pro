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

def phash_packed_bytes(path: str, hash_size: int = 16, max_side: int | None = None) -> np.ndarray:
    """Return packed bytes of pHash.
    Output shape: (hash_size*hash_size/8,) uint8.
    For hash_size=16 => 256 bits => 32 bytes.
    """
    img = open_image_rgb(path)
    img = resize_max_side(img, max_side)
    image_hash = imagehash.average_hash(img, hash_size=hash_size)  # Average hash - works better for color images
    hash_bits = np.asarray(image_hash.hash, dtype=np.uint8).reshape(-1)  # 0/1
    packed_bytes = np.packbits(hash_bits)  # uint8
    return packed_bytes

def hamming_distances_packed(query_packed: np.ndarray, db_packed: np.ndarray) -> np.ndarray:
    """Vectorized Hamming distance for packed uint8 hashes.
    query_packed: (B,)
    db_packed:    (N,B)
    returns: (N,) bit distances
    """
    xor_result = np.bitwise_xor(db_packed, query_packed)  # (N,B)
    return BIT_COUNT_LOOKUP_TABLE[xor_result].sum(axis=1).astype(np.int32)

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
    query_image = cv2.imread(query_path, cv2.IMREAD_GRAYSCALE)
    if query_image is None:
        raise ValueError(f"Cannot read query image: {query_path}")

    if max_side and max(query_image.shape[:2]) > max_side:
        scale = max_side / max(query_image.shape[:2])
        query_image = cv2.resize(query_image, (int(query_image.shape[1] * scale), int(query_image.shape[0] * scale)))

    orb_detector = cv2.ORB_create(nfeatures=nfeatures, scaleFactor=1.2, nlevels=8, edgeThreshold=15, patchSize=31)
    query_keypoints, query_descriptors = orb_detector.detectAndCompute(query_image, None)
    if query_descriptors is None or len(query_keypoints) == 0:
        return [(image_path, 0.0) for image_path in candidate_paths]

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    scored_results: list[tuple[str, float]] = []
    for image_path in candidate_paths:
        candidate_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if candidate_image is None:
            scored_results.append((image_path, 0.0))
            continue

        if max_side and max(candidate_image.shape[:2]) > max_side:
            scale = max_side / max(candidate_image.shape[:2])
            candidate_image = cv2.resize(candidate_image, (int(candidate_image.shape[1] * scale), int(candidate_image.shape[0] * scale)))

        keypoints, descriptors = orb_detector.detectAndCompute(candidate_image, None)
        if descriptors is None or len(keypoints) == 0:
            scored_results.append((image_path, 0.0))
            continue

        matches = matcher.knnMatch(query_descriptors, descriptors, k=2)
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                best_match, second_best_match = match_pair
                if best_match.distance < 0.7 * second_best_match.distance:
                    good_matches.append(best_match)
        min_features_count = max(1, min(len(query_descriptors), len(descriptors)))
        similarity_score = (len(good_matches) / min_features_count) * 100.0
        similarity_score = float(max(0.0, min(100.0, similarity_score)))
        scored_results.append((image_path, similarity_score))

    scored_results.sort(key=lambda x: x[1], reverse=True)
    return scored_results

# =============================================================================
# ALGORITHM 3: DEEP LEARNING (ResNet18)
# Content-aware feature extraction using pre-trained neural network
# Weight: 55% in final score (MOST IMPORTANT)
# =============================================================================

# Global model for deep learning features (lazy loaded)
_deep_model = None
_deep_transform = None

def get_deep_learning_model():
    """Get or initialize the ResNet model for feature extraction."""
    global _deep_model, _deep_transform
    if _deep_model is None:
        # Disable SSL verification for model download
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        
        # Use ResNet18 pre-trained on ImageNet
        _deep_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # Remove the final classification layer to get features
        _deep_model = torch.nn.Sequential(*list(_deep_model.children())[:-1])
        _deep_model.eval()
        
        # Standard ImageNet preprocessing
        _deep_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return _deep_model, _deep_transform

def extract_deep_features(path: str) -> np.ndarray:
    """Extract deep learning features from an image using ResNet."""
    model, transform = get_deep_learning_model()
    img = open_image_rgb(path)
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
    
    with torch.no_grad():
        features = model(img_tensor)
    
    # Flatten to 1D array
    features = features.squeeze().numpy()
    # Normalize
    features = features / (np.linalg.norm(features) + 1e-7)
    return features

def compare_deep_features(features1: np.ndarray, features2: np.ndarray) -> float:
    """Compare deep features using cosine similarity. Returns similarity 0-100%."""
    cosine_similarity = np.dot(features1, features2) / (np.linalg.norm(features1) * np.linalg.norm(features2) + 1e-7)
    # Convert from [-1, 1] to [0, 100]
    similarity_percentage = ((cosine_similarity + 1) / 2) * 100
    return max(0.0, min(100.0, similarity_percentage))
