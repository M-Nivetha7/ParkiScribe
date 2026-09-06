"""
Unit and Integration Test Suite for:
AI-Based Air Writing Recognition and Parkinson's Disease Motor Analysis System.
"""

import os
import sys
import unittest
import numpy as np
import torch

# Project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from utils.constants import (
    ALL_CLASSES, BLUE, BLACK, CLINICAL_DISCLAIMER, MODEL_INPUT_SIZE, NUM_CLASSES
)
from src.drawing_utils import AirCanvas
from src.hand_tracker import HandTracker, fingers_up, classify_gesture
from src.preprocessing import preprocess_canvas, extract_character_roi
from src.motor_analysis import ParkinsonMotorAnalyzer, MotorBiomarkers
from src.model import AirWritingCNN, augment_tremor_image
from src.recognizer import AirWritingRecognizer


class TestAirCanvas(unittest.TestCase):
    def setUp(self):
        self.canvas = AirCanvas()
        self.mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    def test_initialization(self):
        self.canvas.initialize_canvas(self.mock_frame)
        self.assertIsNotNone(self.canvas.canvas)
        self.assertEqual(self.canvas.canvas.shape, (480, 640, 3))

    def test_drawing_and_trajectory_logging(self):
        self.canvas.draw(self.mock_frame, 100, 100, timestamp=0.0)
        self.canvas.draw(self.mock_frame, 120, 110, timestamp=0.033)
        self.canvas.draw(self.mock_frame, 140, 120, timestamp=0.066)

        traj = self.canvas.get_trajectory_array()
        self.assertEqual(len(traj), 3)
        self.assertEqual(traj.shape[1], 3)
        # Check non-black pixels in canvas
        gray = self.canvas.canvas[:, :, 0]
        self.assertGreater(np.count_nonzero(gray), 0)

    def test_pen_lift_and_strokes(self):
        # Stroke 1
        self.canvas.draw(self.mock_frame, 100, 100, timestamp=0.0)
        self.canvas.draw(self.mock_frame, 120, 110, timestamp=0.033)
        self.canvas.reset_position(timestamp=0.05)

        # Stroke 2
        self.canvas.draw(self.mock_frame, 200, 200, timestamp=0.10)
        self.canvas.draw(self.mock_frame, 220, 210, timestamp=0.13)

        strokes = self.canvas.get_strokes()
        self.assertEqual(len(strokes), 2)
        self.assertEqual(len(strokes[0]), 2)
        self.assertEqual(len(strokes[1]), 2)

    def test_erase_and_clear(self):
        self.canvas.draw(self.mock_frame, 50, 50, timestamp=0.0)
        self.canvas.erase(50, 50)
        self.canvas.clear_canvas()
        self.assertEqual(np.count_nonzero(self.canvas.canvas), 0)
        self.assertEqual(len(self.canvas.get_trajectory_array()), 0)

    def test_merge_canvas(self):
        frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        self.canvas.draw(frame, 100, 100, timestamp=0.0)
        self.canvas.draw(frame, 150, 150, timestamp=0.033)
        merged = self.canvas.merge_canvas(frame)
        self.assertEqual(merged.shape, frame.shape)


class TestHandTrackerAndGestures(unittest.TestCase):
    def test_fingers_up_and_classify_gesture(self):
        # Synthetic landmark list: 21 points
        # Each landmark: (id, cx, cy)
        # In frame coordinates, smaller cy means higher (up)

        # 1. Fist (all tips lower than PIPs)
        lm_fist = [(i, 100, 200) for i in range(21)]
        # Tips (8, 12, 16, 20) below PIPs (6, 10, 14, 18)
        lm_fist[8] = (8, 100, 250)
        lm_fist[6] = (6, 100, 200)
        lm_fist[12] = (12, 110, 250)
        lm_fist[10] = (10, 110, 200)
        lm_fist[16] = (16, 120, 250)
        lm_fist[14] = (14, 120, 200)
        lm_fist[20] = (20, 130, 250)
        lm_fist[18] = (18, 130, 200)

        f_fist = fingers_up(lm_fist)
        self.assertEqual(f_fist, [False, False, False, False])
        self.assertEqual(classify_gesture(f_fist), "fist")

        # 2. Draw mode (only index tip up)
        lm_draw = list(lm_fist)
        lm_draw[8] = (8, 100, 150)  # tip above PIP 200
        f_draw = fingers_up(lm_draw)
        self.assertEqual(f_draw, [True, False, False, False])
        self.assertEqual(classify_gesture(f_draw), "draw")

        # 3. Pen lift (index + middle up)
        lm_lift = list(lm_draw)
        lm_lift[12] = (12, 110, 150)  # middle tip above PIP
        f_lift = fingers_up(lm_lift)
        self.assertEqual(f_lift, [True, True, False, False])
        self.assertEqual(classify_gesture(f_lift), "pen_lift")

        # 4. Draw mode with relaxed ring or pinky (vital for smooth writing!)
        self.assertEqual(classify_gesture([True, False, True, False]), "draw")
        self.assertEqual(classify_gesture([True, False, False, True]), "draw")
        self.assertEqual(classify_gesture([True, False, True, True]), "draw")



class TestPreprocessing(unittest.TestCase):
    def test_preprocess_canvas_with_drawing(self):
        canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        # Draw a cross simulating character 'X'
        import cv2
        cv2.line(canvas, (100, 100), (200, 200), (255, 0, 0), 8)
        cv2.line(canvas, (200, 100), (100, 200), (255, 0, 0), 8)

        norm_img, meta = preprocess_canvas(canvas, target_size=MODEL_INPUT_SIZE)
        self.assertTrue(meta["has_content"])
        self.assertIsNotNone(meta["bbox"])
        self.assertEqual(norm_img.shape, MODEL_INPUT_SIZE)
        self.assertGreaterEqual(norm_img.min(), 0.0)
        self.assertLessEqual(norm_img.max(), 1.0)

    def test_preprocess_empty_canvas(self):
        canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        norm_img, meta = preprocess_canvas(canvas)
        self.assertFalse(meta["has_content"])
        self.assertEqual(np.count_nonzero(norm_img), 0)


class TestMotorAnalysis(unittest.TestCase):
    def test_motor_biomarkers_extraction(self):
        analyzer = ParkinsonMotorAnalyzer(sampling_rate=30.0)

        # Generate a synthetic 2-second trajectory with 4 Hz oscillation (Parkinsonian tremor)
        t = np.linspace(0, 2.0, 60, dtype=np.float32)
        freq = 5.0  # 5 Hz tremor
        x = 200.0 + 100.0 * t + 10.0 * np.sin(2 * np.pi * freq * t)
        y = 200.0 + 50.0 * t + 10.0 * np.cos(2 * np.pi * freq * t)

        traj = np.column_stack([t, x, y])
        biomarkers = analyzer.analyze_trajectory(traj)

        self.assertIsInstance(biomarkers, MotorBiomarkers)
        self.assertAlmostEqual(biomarkers.writing_duration_sec, 2.0, delta=0.1)
        self.assertGreater(biomarkers.average_speed_px_sec, 0.0)
        self.assertGreater(biomarkers.tremor_intensity_score, 0.0)
        self.assertIn(biomarkers.trajectory_variation, ["Low", "Moderate", "High"])
        self.assertIn("ASSISTIVE COMPUTATION DISCLAIMER", biomarkers.disclaimer)


class TestAirWritingCNN(unittest.TestCase):
    def test_model_forward_and_prediction(self):
        model = AirWritingCNN(num_classes=NUM_CLASSES)
        dummy_input = torch.randn(2, 1, 64, 64)
        logits = model(dummy_input)
        self.assertEqual(logits.shape, (2, NUM_CLASSES))

        probs = model.predict_proba(dummy_input)
        self.assertTrue(torch.allclose(probs.sum(dim=1), torch.tensor([1.0, 1.0]), atol=1e-4))

    def test_recognizer_inference(self):
        recognizer = AirWritingRecognizer()
        mock_canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        import cv2
        cv2.circle(mock_canvas, (320, 240), 60, (255, 0, 0), 8)

        pred, meta = recognizer.predict_canvas(mock_canvas)
        self.assertTrue(meta["has_content"])
        self.assertIsNotNone(pred)
        self.assertIn("predicted_char", pred)
        self.assertIn(pred["predicted_char"], ALL_CLASSES)
        self.assertGreaterEqual(pred["confidence"], 0.0)
        self.assertLessEqual(pred["confidence"], 100.0)


if __name__ == "__main__":
    unittest.main()
