"""
Vectorizer module — crops detected objects from frames and produces
OpenCLIP embeddings.

Designed to be used standalone or plugged into DeviceAgent's processing loop.

Usage:
    from vectorizer import Vectorizer

    v = Vectorizer(model_name="ViT-B-32", pretrained="laion2b_s34b_b79k")
    results = v.process_frame(frame, detections)
    # results = [{"track_id": ..., "type": ..., "embedding": np.ndarray, "crop": np.ndarray}, ...]
"""

import logging
import uuid
from typing import Optional

import cv2
import numpy as np
import open_clip
import torch
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

# Optional background removal — import lazily to avoid hard dependency
_rembg_session = None


def _get_rembg_session():
    """Lazily initialise a rembg session (downloads model on first call)."""
    global _rembg_session
    if _rembg_session is None:
        from rembg import new_session
        _rembg_session = new_session(model_name="u2net")
        _log("rembg session initialised (u2net)")
    return _rembg_session

logger = logging.getLogger(__name__)


def _log(msg):
    print(f"[VECTORIZER] {msg}", flush=True)


class Vectorizer:
    """Crops bounding-box regions from frames and encodes them with OpenCLIP.

    Parameters
    ----------
    model_name : str
        OpenCLIP model architecture, e.g. "ViT-B-32", "ViT-L-14".
    pretrained : str
        Pretrained weights tag, e.g. "laion2b_s34b_b79k".
    device : str or None
        "cuda", "cpu", or None for auto-detect.
    min_crop_px : int
        Minimum crop dimension (width or height) in pixels.  Crops smaller
        than this are skipped — too tiny to produce useful embeddings.
    qdrant_url : str or None
        Qdrant server URL, e.g. "https://qdrant.dev.internal:6333".
        If None, storage is disabled and embeddings are only returned
        in-memory.
    qdrant_api_key : str or None
        Qdrant API key / token for authenticated clusters.
    qdrant_collection : str
        Name of the Qdrant collection to upsert vectors into.
        Created automatically if it doesn't exist.
    qdrant_verify_ssl : bool or str
        SSL verification for the Qdrant connection.  True = verify
        against system CA bundle (default), False = skip verification
        (self-signed certs), or a path to a CA bundle / cert file.
    remove_bg : bool
        If True, remove background from each crop before computing
        the embedding.  Uses rembg (u2net).  Adds ~100-300ms per crop
        on CPU, ~20-50ms on GPU.  Background pixels are replaced with
        white, which is neutral for CLIP models.  Default: False.
    """

    def __init__(
        self,
        model_name: str = "hf-hub:timm/ViT-gopt-16-SigLIP2-384",
        pretrained: str = "hf-hub:timm/ViT-gopt-16-SigLIP2-384",
        device: Optional[str] = None,
        min_crop_px: int = 32,
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        qdrant_collection: str = "detections",
        qdrant_verify_ssl: bool | str = True,
        bbox_increase_rate: float = 0.0,
        remove_bg: bool = False,
    ):
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        self.bbox_increase_rate = bbox_increase_rate
        self.remove_bg = remove_bg
        if remove_bg:
            _get_rembg_session()  # warm-up: download model now, not at first frame

        _log(f"Loading OpenCLIP {model_name} ({pretrained}) on {self.device}...")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        self.model.eval()
        _log("Model loaded")

        self.min_crop_px = min_crop_px

        # Qdrant
        self.qdrant_collection = qdrant_collection
        if qdrant_url:
            kwargs = {
                "url": qdrant_url,
                "api_key": qdrant_api_key,
                "port": None,        # don't append default :6333
                "timeout": 30,
            }
            if qdrant_verify_ssl is False:
                kwargs["verify"] = False
            elif isinstance(qdrant_verify_ssl, str):
                kwargs["verify"] = qdrant_verify_ssl
            self.qdrant = QdrantClient(**kwargs)
            self._ensure_collection()
            _log(f"Qdrant connected: {qdrant_url}  collection={qdrant_collection}")
        else:
            self.qdrant = None

    def _ensure_collection(self):
        """Create the Qdrant collection if it doesn't exist, or verify dimension matches."""
        # Infer vector size from the current model using a dummy image with correct preprocessing
        dummy_pil = Image.new('RGB', (384, 384))  # Create a dummy PIL image
        dummy_tensor = self.preprocess(dummy_pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            current_dim = self.model.encode_image(dummy_tensor).shape[-1]

        collections = {c.name: c for c in self.qdrant.get_collections().collections}

        if self.qdrant_collection in collections:
            # Collection exists — verify dimension
            collection_info = self.qdrant.get_collection(self.qdrant_collection)
            existing_dim = collection_info.config.params.vectors.size

            if existing_dim != current_dim:
                _log(f"⚠️  Dimension mismatch: collection '{self.qdrant_collection}' "
                     f"has dim={existing_dim}, but model produces dim={current_dim}")
                _log(f"🗑️  Deleting old collection and recreating with correct dimension...")

                self.qdrant.delete_collection(self.qdrant_collection)
                self.qdrant.create_collection(
                    collection_name=self.qdrant_collection,
                    vectors_config=VectorParams(size=current_dim, distance=Distance.COSINE),
                )
                _log(f"✅ Recreated collection '{self.qdrant_collection}' (dim={current_dim})")
            else:
                _log(f"Collection '{self.qdrant_collection}' exists with correct dim={current_dim}")
        else:
            # Collection doesn't exist — create it
            self.qdrant.create_collection(
                collection_name=self.qdrant_collection,
                vectors_config=VectorParams(size=current_dim, distance=Distance.COSINE),
            )
            _log(f"✅ Created Qdrant collection '{self.qdrant_collection}' (dim={current_dim})")

    # ------------------------------------------------------------------
    # Cropping helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_bbox(bbox, frame_h: int, frame_w: int) -> tuple[int, int, int, int]:
        """Convert any supported bbox format to (x_min, y_min, x_max, y_max) in pixels.

        Supported formats:
          - Normalized rect dict: {"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4}
            Values in [0..1], relative to frame size.  This is the NX VMS
            objectRegion format.
          - Pixel rect dict: {"x": 100, "y": 200, "w": 300, "h": 400}
            Auto-detected when any value > 1.
          - 4-corner list: [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
            Pixel coordinates; axis-aligned bounding rect extracted.
          - Simple list/tuple: [x_min, y_min, x_max, y_max]
            Either normalized (all <= 1) or pixel values.
        """
        if isinstance(bbox, dict):
            x = bbox.get("x", 0)
            y = bbox.get("y", 0)
            w = bbox.get("w", bbox.get("width", 0))
            h = bbox.get("h", bbox.get("height", 0))
            # Heuristic: if all values <= 1.0, treat as normalized
            if all(v <= 1.0 for v in (x, y, w, h)):
                x_min = int(x * frame_w)
                y_min = int(y * frame_h)
                x_max = int((x + w) * frame_w)
                y_max = int((y + h) * frame_h)
            else:
                x_min, y_min = int(x), int(y)
                x_max, y_max = int(x + w), int(y + h)

        elif isinstance(bbox, (list, tuple)):
            if len(bbox) == 4 and not isinstance(bbox[0], (list, tuple)):
                # [x_min, y_min, x_max, y_max]
                vals = [float(v) for v in bbox]
                if all(v <= 1.0 for v in vals):
                    x_min = int(vals[0] * frame_w)
                    y_min = int(vals[1] * frame_h)
                    x_max = int(vals[2] * frame_w)
                    y_max = int(vals[3] * frame_h)
                else:
                    x_min, y_min, x_max, y_max = (int(v) for v in vals)
            else:
                # 4-corner list: [(x,y), ...]
                xs = [p[0] for p in bbox]
                ys = [p[1] for p in bbox]
                x_min, x_max = int(min(xs)), int(max(xs))
                y_min, y_max = int(min(ys)), int(max(ys))
        else:
            raise ValueError(f"Unsupported bbox format: {type(bbox)}")

        # Clamp to frame boundaries
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(frame_w, x_max)
        y_max = min(frame_h, y_max)

        return x_min, y_min, x_max, y_max

    def crop_objects(self, frame: np.ndarray, detections: list[dict]) -> list[dict]:
        """Crop bounding-box regions from a BGR numpy frame.

        Parameters
        ----------
        frame : np.ndarray
            Full camera frame (H, W, 3) in BGR.
        detections : list[dict]
            Each dict must contain a ``bbox`` key (see _normalize_bbox for formats).
            Extra keys (track_id, type, attributes, etc.) are passed through.

        Returns
        -------
        list[dict]
            Each dict contains the original detection keys plus:
            - "crop": np.ndarray — the BGR crop
            - "crop_rect": (x_min, y_min, x_max, y_max) in pixels
        """
        h, w = frame.shape[:2]
        results = []

        _log(f"Cropping {len(detections)} detection(s) from {w}x{h} frame")

        for det in detections:
            bbox = det.get("bbox")
            if bbox is None:
                continue

            try:
                x_min, y_min, x_max, y_max = self._normalize_bbox(bbox, h, w)
            except (KeyError, ValueError, IndexError) as e:
                _log(f"WARN: bad bbox {bbox}: {e}")
                continue

            crop_w = x_max - x_min
            crop_h = y_max - y_min
            if crop_w < self.min_crop_px or crop_h < self.min_crop_px:
                _log(f"SKIP: crop too small {crop_w}x{crop_h} < {self.min_crop_px}px")
                continue

            #increase crop size
            if self.bbox_increase_rate > 0.0:
                # Calculate bbox dimensions
                bbox_w = x_max - x_min
                bbox_h = y_max - y_min

                # Calculate expansion amount based on bbox SIZE, not position
                expand_x = int(bbox_w * self.bbox_increase_rate)
                expand_y = int(bbox_h * self.bbox_increase_rate)

                # Expand in all directions and clamp to image boundaries
                x_min = max(0, x_min - expand_x)
                x_max = min(w, x_max + expand_x)
                y_min = max(0, y_min - expand_y)
                y_max = min(h, y_max + expand_y)

            crop = frame[y_min:y_max, x_min:x_max]

            entry = {k: v for k, v in det.items() if k != "bbox"}
            entry["bbox"] = bbox
            entry["crop"] = crop
            entry["crop_rect"] = (x_min, y_min, x_max, y_max)
            results.append(entry)

        _log(f"Cropped {len(results)} valid objects")
        return results

    # ------------------------------------------------------------------
    # Background removal
    # ------------------------------------------------------------------

    @staticmethod
    def _remove_background(bgr_crop: np.ndarray) -> np.ndarray:
        """Remove background from a BGR crop using rembg.

        Returns a BGR image where background pixels are replaced with
        a uniform white fill.  White is a neutral choice for CLIP models
        that were trained on web images with varied backgrounds.
        """
        from rembg import remove as rembg_remove

        session = _get_rembg_session()
        rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
        pil_in = Image.fromarray(rgb)
        pil_out = rembg_remove(pil_in, session=session)  # returns RGBA

        # Composite onto white background
        bg = Image.new("RGB", pil_out.size, (255, 255, 255))
        bg.paste(pil_out, mask=pil_out.split()[3])  # alpha channel as mask

        result = cv2.cvtColor(np.array(bg), cv2.COLOR_RGB2BGR)
        return result

    # ------------------------------------------------------------------
    # Vectorization
    # ------------------------------------------------------------------

    def vectorize_crops(self, crops: list[dict]) -> list[dict]:
        """Compute OpenCLIP embeddings for a list of crop dicts.

        Modifies each dict in-place, adding an "embedding" key (np.ndarray
        of shape (D,), L2-normalized).

        Parameters
        ----------
        crops : list[dict]
            Output of crop_objects(). Each must have a "crop" key (BGR ndarray).

        Returns
        -------
        list[dict]
            Same list, with "embedding" added to each entry.
        """
        if not crops:
            return crops

        _log(f"Vectorizing {len(crops)} crop(s)...")

        # Batch preprocessing
        tensors = []
        for entry in crops:
            bgr = entry["crop"]
            if self.remove_bg:
                bgr = self._remove_background(bgr)
                entry["crop_nobg"] = bgr  # keep for debugging / saving
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            tensors.append(self.preprocess(pil_img))

        batch = torch.stack(tensors).to(self.device)

        with torch.no_grad():
            features = self.model.encode_image(batch)
            features /= features.norm(dim=-1, keepdim=True)

        embeddings = features.cpu().numpy().astype("float32")

        for i, entry in enumerate(crops):
            entry["embedding"] = embeddings[i]

        _log(f"Computed {len(embeddings)} embedding(s), dim={embeddings.shape[1]}")
        return crops

    # ------------------------------------------------------------------
    # All-in-one convenience
    # ------------------------------------------------------------------

    def process_frame(
        self, frame: np.ndarray, detections: list[dict],
        extra_payload: dict | None = None,
    ) -> list[dict]:
        """Crop + vectorize + store in one call.

        Parameters
        ----------
        frame : np.ndarray
            BGR frame from the camera.
        detections : list[dict]
            Parsed metadata with bbox info (see crop_objects).
        extra_payload : dict or None
            Additional fields to include in every Qdrant point payload,
            e.g. {"device_id": "...", "engine_id": "..."}.

        Returns
        -------
        list[dict]
            Each entry has: bbox, crop, crop_rect, embedding,
            plus any pass-through keys (track_id, type, attributes, ...).
        """
        crops = self.crop_objects(frame, detections)
        if not crops:
            return []
        results = self.vectorize_crops(crops)
        if results and self.qdrant:
            self.store_to_qdrant(results, extra_payload)
        return results

    # ------------------------------------------------------------------
    # Qdrant storage
    # ------------------------------------------------------------------

    def store_to_qdrant(self, results: list[dict], extra_payload: dict | None = None):
        """Upsert vectorized detections into Qdrant.

        Each result becomes a point with:
          - vector: the L2-normalized OpenCLIP embedding
          - payload: track_id, type, timestamp_us, attributes,
            plus anything from extra_payload (device_id, engine_id, etc.)
        """
        import os
        from datetime import datetime

        # Create crops directory if saving is enabled
        crops_dir = os.path.expanduser("~/crops")  # Change path as needed
        os.makedirs(crops_dir, exist_ok=True)

        points = []
        for r in results:
            point_id = str(uuid.uuid4())
            payload = {
                "track_id": r.get("track_id"),
                "type": r.get("type"),
                "timestamp_us": r.get("timestamp_us"),
                "attributes": r.get("attributes", []),
            }
            if extra_payload:
                payload.update(extra_payload)

            # Debug: save crop images (original + background-removed side by side)
            if "crop" in r:
                try:
                    track_id = r.get("track_id", "unknown").replace("{", "").replace("}", "")
                    timestamp = r.get("timestamp_us", 0)
                    device_id = (extra_payload or {}).get("device_id", "unknown").replace("{", "").replace("}", "")
                    obj_type = r.get("type", "unknown").replace(".", "_")
                    prefix = f"{device_id}_{track_id}_{timestamp}_{obj_type}"

                    # # Always save original crop
                    # cv2.imwrite(os.path.join(crops_dir, f"{prefix}_orig.jpg"), r["crop"])
                    #
                    # # Save bg-removed version if available
                    # if "crop_nobg" in r:
                    #     cv2.imwrite(os.path.join(crops_dir, f"{prefix}_nobg.jpg"), r["crop_nobg"])
                    #     _log(f"Saved debug crops: {prefix}_orig.jpg + _nobg.jpg")
                    # else:
                    #     _log(f"Saved crop: {prefix}_orig.jpg")
                except Exception as e:
                    _log(f"WARN: Failed to save crop: {e}")

            points.append(
                PointStruct(
                    id=point_id,
                    vector=r["embedding"].tolist(),
                    payload=payload,
                )
            )

        try:
            self.qdrant.upsert(
                collection_name=self.qdrant_collection,
                points=points,
            )
            _log(f"Stored {len(points)} vector(s) to Qdrant, collection: {self.qdrant_collection}")
        except Exception as e:
            _log(f"ERROR: Qdrant upsert failed: {e}")

    # ------------------------------------------------------------------
    # Text encoding (for similarity search)
    # ------------------------------------------------------------------

    def encode_text(self, text: str) -> np.ndarray:
        """Encode a text query into the same embedding space as images.

        Returns
        -------
        np.ndarray
            Shape (D,), L2-normalized.
        """
        tokens = open_clip.tokenize([text]).to(self.device)
        with torch.no_grad():
            features = self.model.encode_text(tokens)
            features /= features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().astype("float32")[0]
