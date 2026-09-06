import unittest
import numpy as np
import cv2

from src.preprocessing import segment_canvas_characters
from src.recognizer import AirWritingRecognizer
from nlp.word_recognizer import AssistiveNLPDecoder


class TestWordRecognition(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.recognizer = AirWritingRecognizer()
        cls.decoder = AssistiveNLPDecoder()

    def test_segment_multi_character_canvas(self):
        # Draw "HI"
        canvas = np.zeros((300, 300, 3), dtype=np.uint8)
        cv2.putText(canvas, "H", (40, 200), cv2.FONT_HERSHEY_SIMPLEX, 3.5, (255, 255, 255), 8)
        cv2.putText(canvas, "I", (160, 200), cv2.FONT_HERSHEY_SIMPLEX, 3.5, (255, 255, 255), 8)

        characters = segment_canvas_characters(canvas)
        self.assertEqual(len(characters), 2, "Expected exactly 2 segmented characters for HI")

        box_h = characters[0][1]
        box_i = characters[1][1]
        self.assertLess(box_h[0], box_i[0], "First character box should be to the left of the second")

    def test_beam_search_vocabulary_boost(self):
        posteriors = [
            [("H", 0.7), ("N", 0.3)],
            [("E", 0.8), ("F", 0.2)],
            [("L", 0.9), ("I", 0.1)],
            [("P", 0.75), ("D", 0.25)],
        ]
        results = self.decoder.beam_search_decode(posteriors, beam_width=5)
        self.assertTrue(len(results) > 0)
        self.assertEqual(results[0][0], "HELP", "Beam search should select valid vocabulary word HELP")

    def test_digit_to_letter_lookalike_correction(self):
        posteriors = [
            [("0", 0.8), ("O", 0.2)],
            [("K", 0.9), ("X", 0.1)],
        ]
        results = self.decoder.beam_search_decode(posteriors, beam_width=5)
        top_words = [w for w, _ in results]
        self.assertIn("OK", top_words)
        self.assertEqual(results[0][0], "OK")

    def test_end_to_end_word_prediction(self):
        canvas = np.zeros((300, 500, 3), dtype=np.uint8)
        word = "HELP"
        for i, ch in enumerate(word):
            cv2.putText(canvas, ch, (40 + i * 110, 210), cv2.FONT_HERSHEY_SIMPLEX, 4.0, (255, 255, 255), 10)

        pred, meta = self.recognizer.predict_canvas(canvas)
        self.assertIsNotNone(pred)
        self.assertTrue(pred.get("is_word", False))
        self.assertEqual(pred.get("predicted_word"), "HELP")


if __name__ == "__main__":
    unittest.main()
