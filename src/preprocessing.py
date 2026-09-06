from typing import Any, Dict, Optional, Tuple
import cv2
import numpy as np

from utils.constants import MODEL_INPUT_SIZE


def extract_character_roi(
    mask: np.ndarray,
    padding: int = 15
) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
    """
    Find bounding box enclosing all drawn strokes, crop with padding,
    and center on a square canvas preserving stroke aspect ratio.

    Returns:
        (square_roi, (x, y, w, h)) or (None, None) if no strokes found.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    # Filter out tiny noise specks (< 20 px area)
    valid_contours = [c for c in contours if cv2.contourArea(c) > 20]
    if not valid_contours:
        valid_contours = contours

    # Compute union bounding box across all strokes
    x_min = min(cv2.boundingRect(c)[0] for c in valid_contours)
    y_min = min(cv2.boundingRect(c)[1] for c in valid_contours)
    x_max = max(cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2] for c in valid_contours)
    y_max = max(cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3] for c in valid_contours)

    w = max(x_max - x_min, 1)
    h = max(y_max - y_min, 1)

    # Apply padding within frame boundaries
    H, W = mask.shape[:2]
    x0 = max(0, x_min - padding)
    y0 = max(0, y_min - padding)
    x1 = min(W, x_max + padding)
    y1 = min(H, y_max + padding)

    cropped = mask[y0:y1, x0:x1]
    ch, cw = cropped.shape[:2]

    # Create square canvas preserving aspect ratio
    max_side = max(ch, cw)
    square_canvas = np.zeros((max_side, max_side), dtype=np.uint8)

    # Center crop inside square canvas
    offset_y = (max_side - ch) // 2
    offset_x = (max_side - cw) // 2
    square_canvas[offset_y:offset_y + ch, offset_x:offset_x + cw] = cropped

    return square_canvas, (x_min, y_min, w, h)


def segment_canvas_characters(
    canvas_bgr_or_gray: np.ndarray,
    min_contour_area: int = 25,
    padding: int = 10,
    target_size: Tuple[int, int] = MODEL_INPUT_SIZE
) -> list:
    """
    Segments individual characters in a word or multi-character writing from left to right.
    Merges disjoint strokes belonging to the same character (e.g. crossbars of 'A', 'H', dots).

    Returns:
        List of tuples: [(normalized_64x64_img, (x, y, w, h)), ...]
    """
    if canvas_bgr_or_gray is None:
        return []

    if len(canvas_bgr_or_gray.shape) == 3:
        gray = cv2.cvtColor(canvas_bgr_or_gray, cv2.COLOR_BGR2GRAY)
    else:
        gray = canvas_bgr_or_gray.copy()

    _, thresh = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
    if cv2.countNonZero(thresh) < 30:
        return []

    # Morphological closing to bridge small involuntary tremors / stroke breaks
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        if cv2.contourArea(c) >= min_contour_area:
            boxes.append(list(cv2.boundingRect(c)))

    if not boxes:
        return []

    # Sort boxes left-to-right
    boxes.sort(key=lambda b: b[0])

    med_h = float(np.median([b[3] for b in boxes]))

    def is_complete_char(b):
        w, h = b[2], b[3]
        return (w >= 0.40 * med_h and h >= 0.65 * med_h)

    # Cluster stroke fragments into complete characters
    clusters = []
    curr = boxes[0]

    for nxt in boxes[1:]:
        curr_x0, curr_y0, curr_w, curr_h = curr
        curr_x1 = curr_x0 + curr_w
        nxt_x0, nxt_y0, nxt_w, nxt_h = nxt
        nxt_x1 = nxt_x0 + nxt_w

        h_overlap = min(curr_x1, nxt_x1) - max(curr_x0, nxt_x0)
        gap = nxt_x0 - curr_x1

        both_complete = is_complete_char(curr) and is_complete_char(nxt)

        if both_complete:
            should_merge = False
        else:
            should_merge = (h_overlap > -8) or (gap <= 8)

        if should_merge:
            x_min = min(curr_x0, nxt_x0)
            y_min = min(curr_y0, nxt_y0)
            x_max = max(curr_x1, nxt_x1)
            y_max = max(curr_y0 + curr_h, nxt_y0 + nxt_h)
            curr = [x_min, y_min, x_max - x_min, y_max - y_min]
        else:
            clusters.append(curr)
            curr = nxt
    clusters.append(curr)

    # Extract, square-pad, and normalize each segmented character
    results = []
    H, W = gray.shape[:2]
    for b in clusters:
        bx, by, bw, bh = b
        x0 = max(0, bx - padding)
        y0 = max(0, by - padding)
        x1 = min(W, bx + bw + padding)
        y1 = min(H, by + bh + padding)

        cropped = thresh[y0:y1, x0:x1]
        ch, cw = cropped.shape[:2]
        if ch == 0 or cw == 0:
            continue

        max_side = max(ch, cw)
        sq = np.zeros((max_side, max_side), dtype=np.uint8)
        oy = (max_side - ch) // 2
        ox = (max_side - cw) // 2
        sq[oy:oy + ch, ox:ox + cw] = cropped

        resized = cv2.resize(sq, target_size, interpolation=cv2.INTER_AREA)
        norm = resized.astype(np.float32) / 255.0
        results.append((norm, tuple(b)))

    return results



def preprocess_canvas(
    canvas_bgr: np.ndarray,
    target_size: Tuple[int, int] = MODEL_INPUT_SIZE,
    median_ksize: int = 3,
    padding: int = 15
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Handwriting Preprocessing Pipeline (Module 8):
    1. Grayscale conversion.
    2. Thresholding to isolate drawing from background.
    3. Noise reduction using Median filtering and morphological closing.
    4. Character segmentation (bounding box crop & aspect-ratio square centering).
    5. Resize to fixed dimensions (e.g. 64x64 or 28x28).
    6. Normalization to [0.0, 1.0] float32 tensor.

    Returns:
        normalized_img: np.ndarray of shape (target_size[0], target_size[1]), float32 in [0, 1]
        metadata: dictionary with intermediate images and bounding box info.
    """
    metadata: Dict[str, Any] = {
        "has_content": False,
        "bbox": None,
        "raw_gray": None,
        "filtered": None,
        "segmented": None,
    }

    if canvas_bgr is None:
        empty = np.zeros(target_size, dtype=np.float32)
        return empty, metadata

    # 1. Convert to grayscale
    if len(canvas_bgr.shape) == 3:
        gray = cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = canvas_bgr.copy()
    metadata["raw_gray"] = gray

    # 2. Thresholding: strokes are non-zero
    _, thresh = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)

    # Check if canvas has any drawn strokes
    if cv2.countNonZero(thresh) < 30:
        empty = np.zeros(target_size, dtype=np.float32)
        return empty, metadata

    # 3. Noise Reduction: Median filtering handles isolated tremor spikes
    if median_ksize > 1:
        filtered = cv2.medianBlur(thresh, median_ksize)
    else:
        filtered = thresh

    # Morphological closing to bridge small involuntary tremors / stroke breaks
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    filtered = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, kernel)
    metadata["filtered"] = filtered

    # 4. Character Segmentation
    square_roi, bbox = extract_character_roi(filtered, padding=padding)

    if square_roi is None or square_roi.size == 0:
        empty = np.zeros(target_size, dtype=np.float32)
        return empty, metadata

    # 5. Multi-character word segmentation
    metadata["has_content"] = True
    metadata["bbox"] = bbox
    metadata["segmented"] = square_roi
    metadata["segmented_characters"] = segment_canvas_characters(filtered, padding=padding, target_size=target_size)

    # 6. Resize to target dimension
    resized = cv2.resize(square_roi, target_size, interpolation=cv2.INTER_AREA)

    # 7. Normalization to [0, 1] float32
    normalized = resized.astype(np.float32) / 255.0

    return normalized, metadata

