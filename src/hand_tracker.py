from typing import List, Optional, Tuple
import cv2
import numpy as np

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except (ImportError, Exception):
    MEDIAPIPE_AVAILABLE = False
    mp = None


class HandTracker:
    """
    Tracks hand landmarks using MediaPipe Hands (21 keypoints)
    and extracts pixel coordinates of the index fingertip for air drawing.
    """

    def __init__(self, max_hands: int = 1, detection_conf: float = 0.7, track_conf: float = 0.7):
        self.max_hands = max_hands
        self.detection_conf = detection_conf
        self.track_conf = track_conf
        self.results = None

        if MEDIAPIPE_AVAILABLE:
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                max_num_hands=max_hands,
                min_detection_confidence=detection_conf,
                min_tracking_confidence=track_conf
            )
            self.mp_draw = mp.solutions.drawing_utils
        else:
            self.mp_hands = None
            self.hands = None
            self.mp_draw = None

    def find_hands(self, frame: np.ndarray):
        """Process BGR frame and locate hands."""
        if not MEDIAPIPE_AVAILABLE or self.hands is None:
            return None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(rgb)
        return self.results

    def get_landmarks(self, frame: np.ndarray) -> List[Tuple[int, int, int]]:
        """
        Extract list of 21 landmarks as (id, cx, cy) in frame pixel coordinates.
        """
        h, w, _ = frame.shape
        lm_list: List[Tuple[int, int, int]] = []

        if self.results and self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                for idx, lm in enumerate(hand_landmarks.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lm_list.append((idx, cx, cy))

        return lm_list

    def draw_hands(self, frame: np.ndarray) -> None:
        """Draw hand skeletal connections on the frame."""
        if not MEDIAPIPE_AVAILABLE or self.mp_draw is None or not self.results:
            return
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )


def fingers_up(lm_list: List[Tuple[int, int, int]]) -> List[bool]:
    """
    Determine which fingers are extended upwards (Index, Middle, Ring, Pinky).
    A finger is considered up if its tip landmark y-coord is above (smaller than)
    the PIP joint y-coord.

    Returns:
        [index_up, middle_up, ring_up, pinky_up]
    """
    if len(lm_list) < 21:
        return []

    fingers = []

    # Landmark IDs:
    # Index:  Tip=8,  PIP=6
    # Middle: Tip=12, PIP=10
    # Ring:   Tip=16, PIP=14
    # Pinky:  Tip=20, PIP=18

    # Index
    fingers.append(lm_list[8][2] < lm_list[6][2])
    # Middle
    fingers.append(lm_list[12][2] < lm_list[10][2])
    # Ring
    fingers.append(lm_list[16][2] < lm_list[14][2])
    # Pinky
    fingers.append(lm_list[20][2] < lm_list[18][2])

    return fingers


def classify_gesture(fingers: List[bool]) -> str:
    """
    Classify gesture based on extended fingers:
    - 'fist': All fingers down -> Clear canvas
    - 'pen_lift': Index and Middle up -> Stop drawing / Pen lifted (move freely)
    - 'draw': Index up and Middle down -> Smooth on-live drawing mode
    - 'erase': Pinky up with index and middle down -> Eraser mode
    - 'idle': Other postures
    """
    if not fingers or len(fingers) < 4:
        return 'none'

    index_up, middle_up, ring_up, pinky_up = fingers[0], fingers[1], fingers[2], fingers[3]

    # Fist: All 4 fingers down -> Clear canvas (matches user starter code)
    if not (index_up or middle_up or ring_up or pinky_up):
        return 'fist'

    # Two fingers up (Index + Middle) -> Stop drawing / Pen lift (matches user starter code)
    if index_up and middle_up:
        return 'pen_lift'

    # Index finger up and Middle finger down -> Draw! (matches user starter code: fingers[0] and not fingers[1])
    if index_up and not middle_up:
        return 'draw'

    # Eraser mode (optional): Pinky extended with index and middle down
    if pinky_up and not index_up and not middle_up:
        return 'erase'

    return 'idle'

