"""
Constants for Air Writing Recognition and Parkinson's Disease Motor Analysis System.
Defines colors (BGR format for OpenCV), geometry parameters, landmark indices,
and Parkinsonian biomarker thresholds.
"""

from typing import Tuple

# --- BGR Colors for OpenCV ---
BLUE: Tuple[int, int, int] = (255, 0, 0)
BLACK: Tuple[int, int, int] = (0, 0, 0)
WHITE: Tuple[int, int, int] = (255, 255, 255)
GREEN: Tuple[int, int, int] = (0, 255, 0)
RED: Tuple[int, int, int] = (0, 0, 255)
YELLOW: Tuple[int, int, int] = (0, 255, 255)
CYAN: Tuple[int, int, int] = (255, 255, 0)
PURPLE: Tuple[int, int, int] = (255, 0, 255)
ORANGE: Tuple[int, int, int] = (0, 165, 255)
LIGHT_GRAY: Tuple[int, int, int] = (200, 200, 200)
DARK_GRAY: Tuple[int, int, int] = (40, 40, 40)

# --- Drawing & Canvas Defaults ---
DEFAULT_BRUSH_THICKNESS: int = 8
DEFAULT_ERASER_THICKNESS: int = 30
CANVAS_WIDTH: int = 640
CANVAS_HEIGHT: int = 480
MODEL_INPUT_SIZE: Tuple[int, int] = (64, 64)

# --- MediaPipe Hand Landmark Indices (21 points) ---
LANDMARK_WRIST: int = 0
LANDMARK_THUMB_CMC: int = 1
LANDMARK_THUMB_MCP: int = 2
LANDMARK_THUMB_IP: int = 3
LANDMARK_THUMB_TIP: int = 4

LANDMARK_INDEX_MCP: int = 5
LANDMARK_INDEX_PIP: int = 6
LANDMARK_INDEX_DIP: int = 7
LANDMARK_INDEX_TIP: int = 8

LANDMARK_MIDDLE_MCP: int = 9
LANDMARK_MIDDLE_PIP: int = 10
LANDMARK_MIDDLE_DIP: int = 11
LANDMARK_MIDDLE_TIP: int = 12

LANDMARK_RING_MCP: int = 13
LANDMARK_RING_PIP: int = 14
LANDMARK_RING_DIP: int = 15
LANDMARK_RING_TIP: int = 16

LANDMARK_PINKY_MCP: int = 17
LANDMARK_PINKY_PIP: int = 18
LANDMARK_PINKY_DIP: int = 19
LANDMARK_PINKY_TIP: int = 20

# --- Classes for Character Recognition ---
DIGITS = [str(i) for i in range(10)]
UPPERCASE_LETTERS = [chr(c) for c in range(ord('A'), ord('Z') + 1)]
ALL_CLASSES = DIGITS + UPPERCASE_LETTERS
NUM_CLASSES = len(ALL_CLASSES)
CLASS_TO_IDX = {c: i for i, c in enumerate(ALL_CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(ALL_CLASSES)}

# --- Parkinson's Motor Biomarker Settings ---
TREMOR_FREQ_BAND: Tuple[float, float] = (4.0, 7.0)  # Classical resting/postural tremor band in Hz
NORMAL_WRITING_SPEED_MIN: float = 80.0  # px/s
NORMAL_SMOOTHNESS_THRESHOLD: float = 70.0  # %

# --- Medical & Ethical Disclaimer ---
CLINICAL_DISCLAIMER: str = (
    "ASSISTIVE COMPUTATION DISCLAIMER: This system is designed for handwriting "
    "accessibility assistance and kinematic motor pattern characterization. It is "
    "NOT intended or certified to replace professional clinical diagnosis of Parkinson's Disease."
)
