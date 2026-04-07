"""
Search Qdrant for detections similar to a query image.

Usage:
    python search_image.py /path/to/image.jpg
    python search_image.py /path/to/image.jpg --top 10
    python search_image.py /path/to/image.jpg --collection my_collection
"""

import argparse
import sys

import cv2
import numpy as np
import open_clip
import torch
from PIL import Image
from qdrant_client import QdrantClient

# --- Config (edit these or pass via args) ---
QDRANT_URL = "https://nikita-qdrant.fly.dev"
QDRANT_API_KEY = None  # set this or import from config
MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"

try:
    import config
    QDRANT_API_KEY = config.qdrant_api_key
except ImportError:
    pass


def main():
    parser = argparse.ArgumentParser(description="Search Qdrant by image similarity")
    parser.add_argument("image", help="Path to query image")
    parser.add_argument("--top", type=int, default=5, help="Number of results")
    parser.add_argument("--collection", default=None, help="Qdrant collection name")
    parser.add_argument("--url", default=QDRANT_URL, help="Qdrant URL")
    parser.add_argument("--api-key", default=QDRANT_API_KEY, help="Qdrant API key")
    args = parser.parse_args()

    # Load image
    img = cv2.imread(args.image)
    if img is None:
        print(f"ERROR: Cannot read image: {args.image}")
        sys.exit(1)
    print(f"Loaded image: {args.image} ({img.shape[1]}x{img.shape[0]})")

    # Pick device
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # Load model
    print(f"Loading OpenCLIP {MODEL_NAME} on {device}...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED, device=device
    )
    model.eval()

    # Vectorize
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    tensor = preprocess(pil).unsqueeze(0).to(device)

    with torch.no_grad():
        features = model.encode_image(tensor)
        features /= features.norm(dim=-1, keepdim=True)

    query_vector = features.cpu().numpy().astype("float32")[0].tolist()
    print(f"Embedding: {len(query_vector)} dimensions")

    # Connect to Qdrant
    client = QdrantClient(url=args.url, api_key=args.api_key, port=None, timeout=30)

    # Auto-detect collection if not specified
    collection = args.collection
    if not collection:
        cols = [c.name for c in client.get_collections().collections]
        if len(cols) == 1:
            collection = cols[0]
            print(f"Using collection: {collection}")
        elif len(cols) == 0:
            print("ERROR: No collections found in Qdrant")
            sys.exit(1)
        else:
            print(f"Multiple collections found: {cols}")
            print("Specify one with --collection")
            sys.exit(1)

    # Search
    results = client.query_points(
        collection_name=collection,
        query=query_vector,
        limit=args.top,
        with_payload=True,
    )

    # Print results
    print(f"\n{'='*60}")
    print(f"Top {args.top} matches in '{collection}':")
    print(f"{'='*60}")

    for i, point in enumerate(results.points):
        score = point.score
        p = point.payload
        track_id = p.get("track_id", "?")
        obj_type = p.get("type", "?")
        device_id = p.get("device_id", "?")
        ts = p.get("timestamp_us", 0)
        attrs = p.get("attributes", [])

        print(f"\n  #{i+1}  score={score:.4f}")
        print(f"       track_id:  {track_id}")
        print(f"       type:      {obj_type}")
        print(f"       device_id: {device_id}")
        print(f"       timestamp: {ts}")
        if attrs:
            print(f"       attrs:     {attrs}")

    if not results.points:
        print("\n  No matches found.")


if __name__ == "__main__":
    main()
