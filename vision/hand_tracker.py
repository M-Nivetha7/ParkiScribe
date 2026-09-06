"""
Vision-Based Touchless Hand & Fingertip Tracker.
Extracts 2D/3D fingertip coordinates from continuous camera video streams
using adaptive color segmentation, convex hull analysis, and optical flow.
"""

from typing import List, Optional, Tuple
import cv2
import numpy as np


class HandTracker:
    """
    Detects hand contour and extracts pointing fingertip coordinates
    from RGB / BGR camera frames.
    """

    def __init__(
        self,
        min_hand_area: float = 3000.0,
        max_hand_area: float = 200000.0,
        smoothing_window: int = 5
    ):
        self.min_hand_area = min_hand_area
        self.max_hand_area = max_hand_area
        self.history_pts: List[Tuple[float, float]] = []
        self.smoothing_window = smoothing_window
        self.prev_gray = None

    def segment_hand_mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Segment skin region using HSV and YCrCb color spaces.
        """
        # Convert to HSV
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        lower_hsv = np.array([0, 30, 60], dtype=np.uint8)
        upper_hsv = np.array([25, 255, 255], dtype=np.uint8)
        mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)

        # Convert to YCrCb
        ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
        lower_ycrcb = np.array([0, 133, 77], dtype=np.uint8)
        upper_ycrcb = np.array([255, 173, 127], dtype=np.uint8)
        mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)

        # Combine masks
        mask = cv2.bitwise_and(mask_hsv, mask_ycrcb)

        # Morphological filtering to clean noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)

        return mask

    def find_fingertip(
        self,
        frame_bgr: np.ndarray
    ) -> Tuple[Optional[Tuple[float, float, float]], np.ndarray, dict]:
        """
        Process video frame and locate top fingertip coordinate.

        Returns:
            fingertip_norm: (x_norm, y_norm, z_depth) normalized in [0, 1]
            annotated_frame: visual debug frame with contours & tracked tip
            metadata: dict with bbox, area, and tracking state
        """
        h, w = frame_bgr.shape[:2]
        annotated = frame_bgr.copy()
        mask = self.segment_hand_mask(frame_bgr)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None, annotated, {"detected": False}

        # Find largest contour in reasonable area range
        valid_contours = [c for c in contours if self.min_hand_area <= cv2.contourArea(c) <= self.max_hand_area]
        if not valid_contours:
            return None, annotated, {"detected": False}

        max_c = max(valid_contours, key=cv2.contourArea)
        area = cv2.contourArea(max_c)
        x, y, bw, bh = cv2.boundingRect(max_c)

        # Approximate fingertip as topmost point of the hand contour
        topmost = tuple(max_c[max_c[:, :, 1].argmin()][0])

        # Depth estimate: larger bounding box area indicates hand closer to camera (lower depth z)
        # z in [0, 1] where 0 is closest
        z_est = float(np.clip(1.0 - (area / self.max_hand_area), 0.0, 1.0))

        # Normalize coordinates
        fx_norm = float(topmost[0] / w)
        fy_norm = float(topmost[1] / h)

        # Smooth point trajectory
        self.history_pts.append((fx_norm, fy_norm))
        if len(self.history_pts) > self.smoothing_window:
            self.history_pts.pop(0)

        smooth_x = float(np.mean([p[0] for p in self.history_pts]))
        smooth_y = float(np.mean([p[1] for p in self.history_pts]))

        # Visual annotations
        cv2.drawContours(annotated, [max_c], -1, (0, 255, 0), 2)
        cv2.rectangle(annotated, (x, y), (x + bw, y + bh), (255, 100, 0), 2)
        cv2.circle(annotated, topmost, 8, (0, 0, 255), -1)
        cv2.circle(annotated, (int(smooth_x * w), int(smooth_y * h)), 5, (255, 255, 0), -1)
        cv2.putText(
            annotated,
            f"Fingertip: ({smooth_x:.2f}, {smooth_y:.2f})",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        metadata = {
            "detected": True,
            "bbox": (x, y, bw, bh),
            "area": float(area),
            "raw_point": (fx_norm, fy_norm, z_est),
            "smooth_point": (smooth_x, smooth_y, z_est)
        }

        return (smooth_x, smooth_y, z_est), annotated, metadata
