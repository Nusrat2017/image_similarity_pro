import os
import json
import numpy as np
from tqdm import tqdm

from log_utils import log_info, log_ok, log_warn
from utils import iter_images, ahash_packed_bytes, save_paths_jsonl, extract_deep_features, load_paths_jsonl

def build_index(source_folder: str, output_folder: str, hash_size: int = 16, max_size: int | None = 1024):
    """
    Build an image search index from all images in a folder.
    This pre-computes all the features needed for fast image searching later.
    
    Handles renamed files intelligently by detecting content matches and updating paths
    instead of creating duplicate entries.
    
    Args:
        source_folder: Folder containing images to index (searches recursively)
        output_folder: Where to save the index files
        hash_size: Size of perceptual hash (32 = 1024 bits, more accurate but slower)
        max_size: Resize images to this max dimension (None = keep original size)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Try to load existing index for incremental updates
    existing_paths = []
    existing_hashes = None
    existing_features = None
    
    paths_file = os.path.join(output_folder, "paths.jsonl")
    hashes_file = os.path.join(output_folder, "hashes.npy")
    features_file = os.path.join(output_folder, "deep_features.npy")
    
    if os.path.exists(paths_file) and os.path.exists(hashes_file):
        try:
            existing_paths = load_paths_jsonl(paths_file)
            existing_hashes = np.load(hashes_file)
            existing_features = np.load(features_file)
            log_info(f"Loaded existing index with {len(existing_paths)} images")
        except Exception as e:
            log_warn(f"Could not load existing index: {e}")
            existing_paths = []
            existing_hashes = None
            existing_features = None

    existing_path_to_index: dict[str, int] = {}
    for idx, path in enumerate(existing_paths):
        if path not in existing_path_to_index:
            existing_path_to_index[path] = idx

    # Lists to collect data for all images
    image_paths: list[str] = []  # Store file paths
    hashes: list[np.ndarray] = []  # Store perceptual hashes for quick filtering
    features: list[np.ndarray] = []  # Store deep learning features for content matching

    # Track which existing index entries have been matched (to detect deleted files)
    matched_indices = set()

    # Find all images in the source folder
    all_images = list(iter_images(source_folder))
    
    # Process each image and extract its features
    for img_path in tqdm(all_images, desc="Indexing images"):
        try:
            img_abs_path = os.path.abspath(img_path)
            
            # Extract perceptual hash (fast, captures overall structure)
            img_hash = ahash_packed_bytes(img_path, hash_size=hash_size, max_side=max_size)
            
            # Check if this image already exists in the index
            found_match = False
            if existing_hashes is not None:
                # First check by path (fastest - file wasn't renamed)
                if img_abs_path in existing_path_to_index:
                    idx = existing_path_to_index[img_abs_path]
                    matched_indices.add(idx)
                    image_paths.append(img_abs_path)
                    hashes.append(existing_hashes[idx])
                    features.append(existing_features[idx])
                    found_match = True
                else:
                    # Check for renamed files by comparing hashes (content-based match)
                    for idx, (old_path, old_hash) in enumerate(zip(existing_paths, existing_hashes)):
                        if idx in matched_indices:
                            continue
                        # If hashes match exactly, it's the same image with a different path
                        if np.array_equal(img_hash, old_hash):
                            matched_indices.add(idx)
                            image_paths.append(img_abs_path)  # Use new path
                            hashes.append(old_hash)  # Reuse old hash
                            features.append(existing_features[idx])  # Reuse old features
                            log_info(f"Renamed: {os.path.basename(old_path)} -> {os.path.basename(img_abs_path)}")
                            found_match = True
                            break
            
            # If no match found, this is a new image - extract features
            if not found_match:
                # Extract deep learning features (slow, captures content/meaning)
                img_features = extract_deep_features(img_path)
                
                # Save everything for this image
                image_paths.append(img_abs_path)
                hashes.append(img_hash)
                features.append(img_features)
                
        except Exception as e:
            # Skip images that can't be processed (corrupted, wrong format, etc.)
            log_warn(f"Skipping {img_path}: {e}")
            continue

    # Report deleted files (existed in index but not found in folder)
    if existing_paths:
        deleted_count = len(existing_paths) - len(matched_indices)
        if deleted_count > 0:
            log_info(f"Removed {deleted_count} deleted/moved images from index")

    # Make sure we found at least some images
    if not image_paths:
        raise SystemExit("No images indexed. Check your folder path and file extensions.")

    # Convert lists to numpy arrays for efficient storage and loading
    hash_array = np.vstack(hashes).astype(np.uint8)  # Stack all hashes into one array
    feature_array = np.vstack(features).astype(np.float32)  # Stack all features (each image = 512 numbers)

    # Define where to save each component of the index
    paths_file = os.path.join(output_folder, "paths.jsonl")  # Image file paths
    hashes_file = os.path.join(output_folder, "hashes.npy")  # Perceptual hashes
    features_file = os.path.join(output_folder, "deep_features.npy")  # Deep learning features
    config_file = os.path.join(output_folder, "meta.json")  # Metadata about the index

    # Save all the data to disk
    save_paths_jsonl(paths_file, image_paths)
    np.save(hashes_file, hash_array)
    np.save(features_file, feature_array)

    # Save configuration and statistics about the index
    config = {
        "count": int(hash_array.shape[0]),  # How many images were indexed
        "hash_size": int(hash_size),  # Hash size used
        "bytes_per_hash": int(hash_array.shape[1]),  # Bytes needed per hash
        "db_folder": os.path.abspath(source_folder),  # Source folder
        "max_side": int(max_size) if max_size else None,  # Image resize setting
        "deep_feature_dim": int(feature_array.shape[1]),  # Feature vector size (usually 512)
    }
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    log_ok(f"Indexed {config['count']} images")
    log_ok(f"Saved: {paths_file}")
    log_ok(f"Saved: {hashes_file}")
    log_ok(f"Saved: {features_file}")
    log_ok(f"Saved: {config_file}")

def needs_reindex(source_folder: str, output_folder: str) -> tuple[bool, dict]:
    """
    Check if re-indexing is needed by comparing current files with the index.
    
    Returns:
        (needs_reindex: bool, changes: dict with counts of each change type)
    """
    paths_file = os.path.join(output_folder, "paths.jsonl")
    meta_file = os.path.join(output_folder, "meta.json")
    
    changes = {
        'new': 0,
        'deleted': 0,
        'renamed': 0,
        'total_current': 0,
        'total_indexed': 0
    }
    
    # If index doesn't exist, we need to build it
    if not os.path.exists(paths_file) or not os.path.exists(meta_file):
        return True, {'reason': 'Index not found - creating new index'}
    
    try:
        # Load existing index info
        existing_paths = load_paths_jsonl(paths_file)
        existing_paths_set = set(existing_paths)
        changes['total_indexed'] = len(existing_paths)
        
        # Get current images in folder
        current_images = set(os.path.abspath(p) for p in iter_images(source_folder))
        changes['total_current'] = len(current_images)
        
        # Check for new images
        new_images = current_images - existing_paths_set
        changes['new'] = len(new_images)
        
        # Check for deleted images
        deleted_images = existing_paths_set - current_images
        changes['deleted'] = len(deleted_images)
        
        # If counts are different but some files exist, likely renamed
        if changes['total_current'] == changes['total_indexed'] and changes['deleted'] > 0:
            changes['renamed'] = changes['deleted']  # Likely renames
        
        # Check if any changes exist
        needs_update = changes['new'] > 0 or changes['deleted'] > 0
        
        if needs_update:
            return True, changes
        else:
            return False, {'reason': 'Index is up to date'}
        
    except Exception as e:
        return True, {'reason': f'Error checking index: {e}'}

def run_build_index():
    source = "image_database/stored_image"  # Specify the path to your image database folder here
    output = "index"  # Specify the output index folder here
    hash_size = 32  # aHash size (default: 16 -> 256 bits, increase to 32 for better accuracy)
    resize_limit = None  # Downscale images so max side <= this (set to None to disable)

    # Smart indexing - only re-index if needed
    needs_update, changes = needs_reindex(source, output)
    
    if needs_update:
        if 'reason' in changes:
            log_info(changes['reason'])
        else:
            messages = []
            if changes['new'] > 0:
                messages.append(f"{changes['new']} new image(s)")
            if changes['renamed'] > 0:
                messages.append(f"{changes['renamed']} renamed image(s)")
            if changes['deleted'] > 0 and changes['renamed'] == 0:
                messages.append(f"{changes['deleted']} deleted image(s)")
            log_info(f"Detected: {', '.join(messages)}")
        
        build_index(source, output, hash_size=hash_size, max_size=resize_limit)
    else:
        log_info(f"{changes['reason']} - Skipping indexing")

if __name__ == "__main__":
    run_build_index()
