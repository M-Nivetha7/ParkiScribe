"""
Main Interactive Real-Time Application for:
AI-Based Air Writing Recognition and Parkinson's Disease Motor Analysis System.

Combines:
- OpenCV webcam capture with mirror interaction
- MediaPipe Hands 21-landmark tracking
- Gesture interaction:
  * ✍️ Only Index up: Draw stroke & log trajectory
  * ✋ Index + Middle up: Stop drawing / Lift pen (allows multi-stroke letters)
  * ✊ Closed hand / Fist: Clear virtual canvas
  * 🧹 Index + Pinky up: Erase
- Handwriting preprocessing & character segmentation (Module 8)
- Tremor-robust 2D CNN Character Recognition (Modules 9 & 10)
- Parkinsonian Motor Pattern Analysis (Module 11) with live HUD metrics
"""

import os
import sys
import time
from typing import Optional
import cv2
import numpy as np

# Ensure project directory is in python path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from utils.constants import (
    BLACK, BLUE, CYAN, GREEN, ORANGE, PURPLE, RED, WHITE, YELLOW,
    LANDMARK_INDEX_TIP
)
from src.hand_tracker import HandTracker, fingers_up, classify_gesture
from src.drawing_utils import AirCanvas
from src.preprocessing import preprocess_canvas
from src.motor_analysis import ParkinsonMotorAnalyzer, MotorBiomarkers
from src.recognizer import AirWritingRecognizer


def draw_hud(
    frame: np.ndarray,
    gesture_state: str,
    prediction: Optional[dict],
    biomarkers: Optional[MotorBiomarkers],
    cursor_pos: Optional[Tuple[int, int]] = None,
    current_color_name: str = "Blue",
    constructed_text: str = "",
    current_word: str = ""
) -> np.ndarray:
    """
    Renders a sleek, non-intrusive clinical & word recognition status overlay.
    Keeps the main drawing area unblocked so live strokes are completely visible.
    """
    h, w = frame.shape[:2]

    # --- 1. Top Status Bar (Height: 50px) ---
    top_overlay = frame[0:50, 0:w].copy()
    cv2.rectangle(top_overlay, (0, 0), (w, 50), (18, 18, 18), -1)
    frame[0:50, 0:w] = cv2.addWeighted(top_overlay, 0.85, frame[0:50, 0:w], 0.15, 0)
    cv2.line(frame, (0, 50), (w, 50), (55, 55, 55), 1)

    # Gesture Badge (Top Left)
    badge_info = {
        "draw": (GREEN, BLACK, "  DRAWING  "),
        "pen_lift": (CYAN, BLACK, " PEN LIFTED "),
        "fist": (RED, WHITE, "FIST (CLEAR)"),
        "erase": (ORANGE, BLACK, "   ERASER  "),
        "idle": ((130, 130, 130), WHITE, "    IDLE    "),
        "none": ((70, 70, 70), (200, 200, 200), "  NO HAND  ")
    }
    bg_col, txt_col, label = badge_info.get(gesture_state, ((70, 70, 70), WHITE, gesture_state.upper()))
    cv2.rectangle(frame, (12, 10), (145, 40), bg_col, -1, cv2.LINE_AA)
    cv2.putText(frame, label, (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.52, txt_col, 2, cv2.LINE_AA)

    # AI Recognition Result (Center of Top Bar)
    if prediction:
        is_word = prediction.get("is_word", False)
        word = prediction.get("predicted_word", prediction.get("predicted_char", "-"))
        conf = prediction.get("confidence", 0.0)
        cands = prediction.get("top_candidates", [])[1:3]
        cand_str = " ".join([f"{c['char']}:{c['confidence']:.0f}%" for c in cands])

        tag = "Word:" if is_word else "Char:"
        cv2.putText(frame, f"AI {tag} \"{word}\"", (160, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, YELLOW, 2, cv2.LINE_AA)
        cv2.putText(frame, f"({conf:.1f}%)", (320 if len(word) > 2 else 260, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.52, GREEN if conf > 70 else ORANGE, 1, cv2.LINE_AA)
        if cand_str:
            cv2.putText(frame, f"[{cand_str}]", (400 if len(word) > 2 else 330, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (170, 170, 170), 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, "AI: Write letters or words in air (pause 1.4s to decode)", (160, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1, cv2.LINE_AA)

    # Parkinson's Motor Biomarkers (Right of Top Bar)
    if biomarkers:
        bio_txt = f"Speed: {biomarkers.average_speed_px_sec:.0f}px/s | Tremor: {biomarkers.tremor_intensity_score:.0f}%"
        cv2.putText(frame, bio_txt, (w - 300, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.46, WHITE, 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, f"Brush: {current_color_name}", (w - 150, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (160, 160, 160), 1, cv2.LINE_AA)

    # --- 2. Constructed Assistive Sentence Strip (Height: 32px) ---
    sub_overlay = frame[52:84, 0:w].copy()
    cv2.rectangle(sub_overlay, (0, 0), (w, 32), (24, 24, 24), -1)
    frame[52:84, 0:w] = cv2.addWeighted(sub_overlay, 0.82, frame[52:84, 0:w], 0.18, 0)
    cv2.line(frame, (0, 84), (w, 84), (45, 45, 45), 1)

    disp_sent = constructed_text if constructed_text else "[Empty Sentence]"
    disp_col = CYAN if constructed_text else (130, 130, 130)
    cv2.putText(frame, f"Text: {disp_sent}", (15, 73), cv2.FONT_HERSHEY_SIMPLEX, 0.55, disp_col, 2 if constructed_text else 1, cv2.LINE_AA)

    if current_word:
        cv2.putText(frame, f"Buffer: [{current_word}_]", (w - 180, 73), cv2.FONT_HERSHEY_SIMPLEX, 0.55, YELLOW, 2, cv2.LINE_AA)

    # --- 3. Sleek Bottom Control Bar (Height: 28px) ---
    bot_overlay = frame[h - 28:h, 0:w].copy()
    cv2.rectangle(bot_overlay, (0, 0), (w, 28), (18, 18, 18), -1)
    frame[h - 28:h, 0:w] = cv2.addWeighted(bot_overlay, 0.85, frame[h - 28:h, 0:w], 0.15, 0)
    cv2.line(frame, (0, h - 28), (w, h - 28), (55, 55, 55), 1)

    hint_text = "[P] Force Predict | [Space] Add Space | [B] Backspace | [C] Clear | [X] Reset Text | [Q] Quit"
    cv2.putText(frame, hint_text, (15, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1, cv2.LINE_AA)

    # --- 4. Floating Mini-Card for Recognized Word / Letter ---
    if prediction:
        word = prediction.get("predicted_word", prediction.get("predicted_char", "-"))
        conf = prediction.get("confidence", 0.0)
        card_w = max(95, 18 * len(word) + 30)
        card = frame[92:155, 12:12 + card_w].copy()
        cv2.rectangle(card, (0, 0), (card_w, 63), (25, 25, 25), -1)
        frame[92:155, 12:12 + card_w] = cv2.addWeighted(card, 0.82, frame[92:155, 12:12 + card_w], 0.18, 0)
        cv2.rectangle(frame, (12, 92), (12 + card_w, 155), (80, 80, 80), 1, cv2.LINE_AA)
        
        font_scale = 1.3 if len(word) <= 2 else (0.9 if len(word) <= 4 else 0.7)
        cv2.putText(frame, word, (20, 134), cv2.FONT_HERSHEY_SIMPLEX, font_scale, YELLOW, 2, cv2.LINE_AA)
        cv2.putText(frame, f"{conf:.0f}%", (card_w - 38, 146), cv2.FONT_HERSHEY_SIMPLEX, 0.40, GREEN if conf > 70 else ORANGE, 1, cv2.LINE_AA)

    # --- 5. Interactive Live Fingertip Cursor Overlay ---
    if cursor_pos is not None:
        cx, cy = cursor_pos
        if gesture_state == "draw":
            cv2.circle(frame, (cx, cy), 12, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1, cv2.LINE_AA)
            cv2.putText(frame, "Drawing", (cx + 14, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
        elif gesture_state == "pen_lift":
            cv2.circle(frame, (cx, cy), 11, CYAN, 2, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 3, CYAN, -1, cv2.LINE_AA)
            cv2.putText(frame, "Pen Up", (cx + 14, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, CYAN, 1, cv2.LINE_AA)
        elif gesture_state == "fist":
            cv2.circle(frame, (cx, cy), 16, RED, 2, cv2.LINE_AA)
            cv2.putText(frame, "Clear", (cx + 16, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, RED, 1, cv2.LINE_AA)

    return frame


def main():
    print("===================================================================")
    print("  AI-Based Air Writing Recognition & Parkinson's Motor Analysis")
    print("===================================================================")
    print("Live Air Gestures (matches your starter code):")
    print("  [☝] Index Finger Only : Draw strokes / letters / words live on-screen")
    print("  [✌] Index + Middle    : Lift pen / Move freely between letters")
    print("  [✊] Closed Fist      : Clear virtual canvas")
    print("Keyboard Hotkeys:")
    print("  [P] : Trigger Recognition & Motor Analysis Now")
    print("  [Space] : Commit current word / Add space to sentence")
    print("  [B] or [Backspace] : Delete last letter")
    print("  [C] : Clear Canvas")
    print("  [X] : Reset Full Sentence")
    print("  [1] Blue | [2] Cyan | [3] Green | [4] Yellow : Brush Colors")
    print("  [Q] or [ESC] : Quit")
    print("===================================================================\n")

    # Initialize Camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[WARNING] Could not open webcam device 0.")
        print("          If running on macOS, check camera permissions in Terminal.")
        print("          Or run the web application dashboard at http://localhost:5050")
        sys.exit(1)

    tracker = HandTracker()
    canvas = AirCanvas()
    recognizer = AirWritingRecognizer()
    motor_analyzer = ParkinsonMotorAnalyzer()

    # Color palette choices
    color_palette = {
        ord('1'): (BLUE, "Blue"),
        ord('2'): (CYAN, "Cyan"),
        ord('3'): (GREEN, "Green"),
        ord('4'): (YELLOW, "Yellow")
    }
    current_color_name = "Blue"

    current_prediction = None
    current_biomarkers = None

    constructed_text = ""
    current_word = ""
    needs_clear_on_next_draw = False

    last_draw_time = 0.0
    auto_predict_timeout = 1.4  # Auto-predict 1.4s after pen lift
    has_drawn_since_last_predict = False

    while True:
        success, frame = cap.read()
        if not success:
            print("[INFO] Camera stream ended.")
            break

        frame = cv2.flip(frame, 1)
        canvas.initialize_canvas(frame)

        tracker.find_hands(frame)
        lm_list = tracker.get_landmarks(frame)
        tracker.draw_hands(frame)

        gesture_state = "none"
        cursor_pos = None

        if len(lm_list) != 0:
            x1, y1 = lm_list[LANDMARK_INDEX_TIP][1], lm_list[LANDMARK_INDEX_TIP][2]
            cursor_pos = (x1, y1)
            fingers = fingers_up(lm_list)

            # ✊ Fist -> Clear screen
            if fingers == [False, False, False, False] or not any(fingers):
                gesture_state = "fist"
                canvas.clear_canvas()
                current_prediction = None
                current_biomarkers = None
                has_drawn_since_last_predict = False
                needs_clear_on_next_draw = False

            # ✋ Two fingers -> Pen lift / Stop drawing (move freely)
            elif fingers[0] and fingers[1]:
                gesture_state = "pen_lift"
                canvas.reset_position()

            # ✍️ Only index up -> Draw smooth on-live line!
            elif fingers[0] and not fingers[1]:
                gesture_state = "draw"
                if needs_clear_on_next_draw:
                    canvas.clear_canvas()
                    needs_clear_on_next_draw = False

                canvas.draw(frame, x1, y1)
                last_draw_time = time.time()
                has_drawn_since_last_predict = True

            else:
                gesture_state = "idle"
                canvas.reset_position()
        else:
            canvas.reset_position()

        # Check for auto-prediction after finishing drawing
        curr_time = time.time()
        if has_drawn_since_last_predict and (curr_time - last_draw_time > auto_predict_timeout):
            traj = canvas.get_trajectory_array()
            if len(traj) > 8:
                pred, meta = recognizer.predict_canvas(canvas.canvas)
                if pred:
                    current_prediction = pred
                    current_biomarkers = motor_analyzer.analyze_trajectory(traj, canvas.get_strokes())

                    # Accumulate into sentence or word buffer
                    if pred.get("is_word", False):
                        w = pred["predicted_word"]
                        constructed_text = (constructed_text + " " + w).strip()
                        current_word = ""
                    else:
                        c = pred["predicted_char"]
                        current_word += c

                    needs_clear_on_next_draw = True

            has_drawn_since_last_predict = False

        # 1. Merge canvas drawing with live camera frame (EXACT user starter code logic)
        frame_merged = canvas.merge_canvas(frame)

        # 2. Overlay sleek status bar & sentence strip
        display_frame = draw_hud(
            frame_merged,
            gesture_state,
            current_prediction,
            current_biomarkers,
            cursor_pos=cursor_pos,
            current_color_name=current_color_name,
            constructed_text=constructed_text,
            current_word=current_word
        )

        cv2.imshow("Air Writing Recognition & Parkinson's Motor Analysis", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break
        elif key == 32:  # Space key
            if current_word:
                constructed_text = (constructed_text + " " + current_word).strip()
                current_word = ""
            else:
                constructed_text += " "
            canvas.clear_canvas()
            needs_clear_on_next_draw = False
        elif key == 8 or key == ord('b'):  # Backspace
            if current_word:
                current_word = current_word[:-1]
            elif constructed_text:
                constructed_text = constructed_text[:-1]
        elif key == ord('c'):  # Clear canvas
            canvas.clear_canvas()
            current_prediction = None
            current_biomarkers = None
            current_word = ""
            has_drawn_since_last_predict = False
            needs_clear_on_next_draw = False
        elif key == ord('x'):  # Reset text & sentence
            constructed_text = ""
            current_word = ""
            canvas.clear_canvas()
        elif key == ord('p'):  # Force predict
            traj = canvas.get_trajectory_array()
            if len(traj) > 5:
                pred, meta = recognizer.predict_canvas(canvas.canvas)
                if pred:
                    current_prediction = pred
                    current_biomarkers = motor_analyzer.analyze_trajectory(traj, canvas.get_strokes())
                    if pred.get("is_word", False):
                        w = pred["predicted_word"]
                        constructed_text = (constructed_text + " " + w).strip()
                        current_word = ""
                    else:
                        current_word += pred["predicted_char"]
                    needs_clear_on_next_draw = True
        elif key in color_palette:
            canvas.color, current_color_name = color_palette[key]

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


