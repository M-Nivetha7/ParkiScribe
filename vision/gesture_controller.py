"""
Air-Writing Gesture Controller & Stroke Segmenter.
State machine managing touchless Pen-Down / Pen-Up / Clear triggers
and segmenting continuous video streams into isolated character trajectories.
"""

import time
import enum
from typing import Callable, List, Optional, Tuple
import numpy as np


class WritingState(str, enum.Enum):
    IDLE = "idle"
    WRITING = "writing"
    STROKE_END = "stroke_end"


class AirWritingGestureController:
    """
    Monitors fingertip movements, segments strokes based on kinematic dwell time / pauses,
    and dispatches completed trajectories to the recognition engine.
    """

    def __init__(
        self,
        dwell_time_sec: float = 0.55,
        min_stroke_points: int = 5,
        max_stroke_points: int = 250,
        speed_threshold: float = 0.015
    ):
        self.dwell_time_sec = dwell_time_sec
        self.min_stroke_points = min_stroke_points
        self.max_stroke_points = max_stroke_points
        self.speed_threshold = speed_threshold

        self.current_state = WritingState.IDLE
        self.current_stroke: List[Tuple[float, float, float]] = []
        self.last_move_time = time.time()
        self.last_point: Optional[Tuple[float, float, float]] = None

    def reset(self):
        """Reset controller state."""
        self.current_state = WritingState.IDLE
        self.current_stroke.clear()
        self.last_move_time = time.time()
        self.last_point = None

    def process_point(
        self,
        point: Optional[Tuple[float, float, float]]
    ) -> Tuple[WritingState, Optional[np.ndarray]]:
        """
        Feed consecutive fingertip point (x, y, z) into the gesture state machine.

        Returns:
            state: Current WritingState
            completed_stroke: Trajectory np.ndarray if a stroke was just completed, else None
        """
        now = time.time()

        if point is None:
            if self.current_state == WritingState.WRITING:
                if len(self.current_stroke) >= self.min_stroke_points:
                    stroke_arr = np.array(self.current_stroke, dtype=np.float32)
                    self.reset()
                    return WritingState.STROKE_END, stroke_arr
                else:
                    self.reset()
            return WritingState.IDLE, None

        if self.last_point is None:
            self.last_point = point
            self.last_move_time = now
            self.current_state = WritingState.WRITING
            self.current_stroke = [point]
            return WritingState.WRITING, None

        dist = np.linalg.norm(np.array(point[:2]) - np.array(self.last_point[:2]))

        if dist > self.speed_threshold:
            # Active movement
            self.last_move_time = now
            self.current_state = WritingState.WRITING
            self.current_stroke.append(point)
            self.last_point = point

            if len(self.current_stroke) >= self.max_stroke_points:
                stroke_arr = np.array(self.current_stroke, dtype=np.float32)
                self.reset()
                return WritingState.STROKE_END, stroke_arr

            return WritingState.WRITING, None

        else:
            # Stationary / pause
            if self.current_state == WritingState.WRITING:
                self.current_stroke.append(point)
                self.last_point = point

                # Check if paused long enough to trigger stroke completion
                if (now - self.last_move_time) > self.dwell_time_sec:
                    if len(self.current_stroke) >= self.min_stroke_points:
                        stroke_arr = np.array(self.current_stroke, dtype=np.float32)
                        self.reset()
                        return WritingState.STROKE_END, stroke_arr
                    else:
                        self.reset()
                        return WritingState.IDLE, None

                return WritingState.WRITING, None

            return WritingState.IDLE, None
