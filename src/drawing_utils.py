import time
from typing import List, Optional, Tuple
import cv2
import numpy as np

from utils.constants import BLACK, BLUE, DEFAULT_BRUSH_THICKNESS, DEFAULT_ERASER_THICKNESS


class AirCanvas:
    """
    Virtual Air Canvas for touchless finger drawing and trajectory tracking.
    Renders strokes onto a dedicated canvas, merges with webcam frames,
    and logs timestamped (t, x, y) coordinates for Parkinsonian motor analysis.
    """

    def __init__(self):
        self.prev_x, self.prev_y = 0, 0
        self.canvas: Optional[np.ndarray] = None
        self.color = BLUE
        self.brush_thickness = DEFAULT_BRUSH_THICKNESS
        self.eraser_thickness = DEFAULT_ERASER_THICKNESS

        # Trajectory logging for Parkinson's motor analysis: list of (timestamp, x, y, is_drawing)
        self.trajectory_log: List[Tuple[float, float, float, bool]] = []
        self._start_time: Optional[float] = None

    def initialize_canvas(self, frame: np.ndarray) -> None:
        """Initialize blank black canvas matching frame dimensions if not yet created."""
        if self.canvas is None:
            self.canvas = np.zeros_like(frame)

    def draw(self, frame: np.ndarray, x: int, y: int, timestamp: Optional[float] = None) -> None:
        """
        Draw a smooth continuous stroke from previous position to (x, y)
        and log trajectory point.
        """
        self.initialize_canvas(frame)

        curr_time = time.time() if timestamp is None else timestamp
        if self._start_time is None:
            self._start_time = curr_time

        # If starting new stroke
        if self.prev_x == 0 and self.prev_y == 0:
            self.prev_x, self.prev_y = x, y

        # Draw smooth line
        cv2.line(
            self.canvas,
            (self.prev_x, self.prev_y),
            (x, y),
            self.color,
            self.brush_thickness
        )

        # Log trajectory point
        rel_time = curr_time - self._start_time
        self.trajectory_log.append((rel_time, float(x), float(y), True))

        self.prev_x, self.prev_y = x, y

    def erase(self, x: int, y: int) -> None:
        """Erase stroke at (x, y) using black color and eraser thickness."""
        if self.canvas is None:
            return

        if self.prev_x == 0 and self.prev_y == 0:
            self.prev_x, self.prev_y = x, y

        cv2.line(
            self.canvas,
            (self.prev_x, self.prev_y),
            (x, y),
            BLACK,
            self.eraser_thickness
        )

        self.prev_x, self.prev_y = x, y

    def reset_position(self, timestamp: Optional[float] = None) -> None:
        """Lift pen (important to stop unwanted connecting lines between strokes)."""
        if self.prev_x != 0 or self.prev_y != 0:
            # Mark pen lift in trajectory
            if self._start_time is not None:
                curr_time = time.time() if timestamp is None else timestamp
                rel_time = curr_time - self._start_time
                self.trajectory_log.append((rel_time, float(self.prev_x), float(self.prev_y), False))
        self.prev_x, self.prev_y = 0, 0

    def clear_canvas(self) -> None:
        """Clear canvas and reset recorded trajectory."""
        if self.canvas is not None:
            self.canvas[:] = 0
        self.reset_position()
        self.trajectory_log.clear()
        self._start_time = None

    def merge_canvas(self, frame: np.ndarray) -> np.ndarray:
        """
        Merge drawing canvas onto video frame using binary bitwise masking.
        Preserves original frame colors outside drawn strokes.
        """
        if self.canvas is None:
            return frame

        # Convert canvas to grayscale
        gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)

        # Create mask
        _, mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        # Convert to 3 channel
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask_inv = cv2.cvtColor(mask_inv, cv2.COLOR_GRAY2BGR)

        # Black-out area on frame
        frame_bg = cv2.bitwise_and(frame, mask_inv)

        # Take only drawing
        canvas_fg = cv2.bitwise_and(self.canvas, mask)

        # Combine
        combined = cv2.add(frame_bg, canvas_fg)

        return combined

    def get_trajectory_array(self) -> np.ndarray:
        """
        Get recorded drawing points as numpy array of shape (N, 3): [timestamp, x, y].
        Filters to only active drawing points.
        """
        drawing_pts = [(t, x, y) for (t, x, y, is_draw) in self.trajectory_log if is_draw]
        if not drawing_pts:
            return np.empty((0, 3), dtype=np.float32)
        return np.array(drawing_pts, dtype=np.float32)

    def get_strokes(self) -> List[np.ndarray]:
        """
        Get list of separate stroke segments.
        Each stroke is an array of shape (M, 3) [timestamp, x, y].
        """
        strokes: List[List[Tuple[float, float, float]]] = []
        curr_stroke: List[Tuple[float, float, float]] = []

        for t, x, y, is_draw in self.trajectory_log:
            if is_draw:
                curr_stroke.append((t, x, y))
            else:
                if len(curr_stroke) > 0:
                    strokes.append(curr_stroke)
                    curr_stroke = []

        if len(curr_stroke) > 0:
            strokes.append(curr_stroke)

        return [np.array(s, dtype=np.float32) for s in strokes if len(s) > 1]
