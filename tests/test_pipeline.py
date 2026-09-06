"""
Comprehensive Unit & Integration Test Suite for Air-Writing Recognition System.
Tests data generation, Parkinsonian simulation, preprocessing, models, and NLP decoding.
"""

import unittest
import numpy as np
import torch

from data.character_templates import ALL_CLASSES, NUM_CLASSES, get_canonical_trajectory, get_all_canonical_trajectories
from data.parkinson_simulator import ParkinsonSimulator, PDSeverity
from data.dataset_generator import AirWritingDatasetGenerator
from data.dataset_loader import AirWritingDataset, create_dataloaders
from preprocessing.trajectory_filter import TrajectoryFilter
from preprocessing.normalizer import TrajectoryNormalizer
from preprocessing.resampler import TrajectoryResampler
from preprocessing.feature_extractor import TrajectoryFeatureExtractor
from models.conv1d_bilstm import Conv1DBiLSTMAttention
from models.transformer_model import TrajectoryTransformer
from models.baseline_classifier import BaselineTrajectoryClassifier
from models.ensemble_recognizer import AirWritingRecognizer
from nlp.word_recognizer import AssistiveNLPDecoder, ASSISTIVE_VOCABULARY
from vision.gesture_controller import AirWritingGestureController, WritingState


class TestAirWritingPipeline(unittest.TestCase):

    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_character_templates(self):
        """Test canonical stroke templates for 36 classes (0-9, A-Z)."""
        self.assertEqual(NUM_CLASSES, 36)
        all_trajs = get_all_canonical_trajectories(num_points=64)
        for char, traj in all_trajs.items():
            self.assertEqual(traj.shape, (64, 3), f"Shape mismatch for character {char}")
            # Ensure coordinates are within normalized [0, 1] range
            self.assertTrue(np.all(traj[:, :2] >= -0.1) and np.all(traj[:, :2] <= 1.1))

    def test_parkinson_simulator(self):
        """Test biological Parkinsonian degradation across all severity levels."""
        sim = ParkinsonSimulator(sampling_rate=30.0, rng_seed=42)
        base_traj = get_canonical_trajectory("A", num_points=64)

        for sev in [PDSeverity.CONTROL, PDSeverity.MILD, PDSeverity.MODERATE, PDSeverity.SEVERE]:
            deg_traj, meta = sim.apply_parkinson_effects(base_traj, severity=sev)
            self.assertEqual(deg_traj.shape, (64, 3))
            self.assertEqual(meta["severity"], sev.value)
            if sev != PDSeverity.CONTROL:
                self.assertGreaterEqual(meta["tremor_freq_hz"], 3.5)
                self.assertLessEqual(meta["tremor_freq_hz"], 7.5)

    def test_preprocessing_modules(self):
        """Test filtering, normalization, resampling, and feature extraction."""
        traj = np.random.randn(80, 3).astype(np.float32)

        # 1. Filters
        sg_filtered = TrajectoryFilter.savitzky_golay(traj, window_length=5)
        self.assertEqual(sg_filtered.shape, traj.shape)

        bw_filtered = TrajectoryFilter.butterworth_lowpass(traj, cutoff_hz=10.0)
        self.assertEqual(bw_filtered.shape, traj.shape)

        # 2. Resampler
        resampled = TrajectoryResampler.resample_arc_length(traj, num_points=64)
        self.assertEqual(resampled.shape, (64, 3))

        # 3. Normalizer
        normalized = TrajectoryNormalizer.center_and_scale(resampled, target_size=1.0)
        self.assertEqual(normalized.shape, (64, 3))
        self.assertTrue(np.all(normalized >= 0.0) and np.all(normalized <= 1.0))

        # 4. Feature Extractor
        extractor = TrajectoryFeatureExtractor(fs=30.0)
        feats = extractor.extract_features(normalized)
        self.assertIsInstance(feats, np.ndarray)
        self.assertGreater(len(feats), 15)
        self.assertFalse(np.any(np.isnan(feats)))

    def test_conv1d_bilstm_model(self):
        """Test Conv1D-BiLSTM forward pass, attention weights, and gradient flow."""
        model = Conv1DBiLSTMAttention(in_channels=3, num_classes=NUM_CLASSES)
        x = torch.randn(4, 3, 64)
        logits, attn = model(x)

        self.assertEqual(logits.shape, (4, NUM_CLASSES))
        self.assertEqual(attn.shape, (4, 64, 1))

        # Attention weights should sum to 1 along sequence dimension
        sum_attn = torch.sum(attn, dim=1)
        self.assertTrue(torch.allclose(sum_attn, torch.ones_like(sum_attn), atol=1e-5))

        # Single prediction helper
        single_traj = np.random.randn(64, 3).astype(np.float32)
        preds, attn_arr = model.predict_single(single_traj, top_k=5)
        self.assertEqual(len(preds), 5)
        self.assertEqual(len(attn_arr), 64)

    def test_transformer_model(self):
        """Test Trajectory Transformer forward pass."""
        model = TrajectoryTransformer(in_channels=3, num_classes=NUM_CLASSES)
        x = torch.randn(4, 3, 64)
        logits, pooled = model(x)

        self.assertEqual(logits.shape, (4, NUM_CLASSES))
        self.assertEqual(pooled.shape, (4, 64))

    def test_baseline_classifier(self):
        """Test Random Forest baseline classifier on synthetic data."""
        clf = BaselineTrajectoryClassifier(model_type="rf", random_state=42)
        X = np.random.randn(10, 64, 3).astype(np.float32)
        y = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        clf.fit(X, y)
        preds = clf.predict(X)
        self.assertEqual(len(preds), 10)
        single_preds = clf.predict_single(X[0], top_k=3)
        self.assertEqual(len(single_preds), 3)

    def test_nlp_word_recognizer(self):
        """Test Levenshtein distance, auto-completion, and beam search decoding."""
        decoder = AssistiveNLPDecoder()

        # 1. Edit distance
        self.assertEqual(decoder.levenshtein_distance("HELP", "HELP"), 0)
        self.assertEqual(decoder.levenshtein_distance("HELP", "HELL"), 1)

        # 2. Vocabulary correction
        corrections = decoder.correct_word("HEKP", max_distance=1)
        self.assertTrue(any(w == "HELP" for w, _ in corrections))

        # 3. Autocomplete prefix
        autocomp = decoder.autocomplete_prefix("WAT", max_results=3)
        self.assertIn("WATER", autocomp)

        # 4. Beam Search Decoding for "HELP"
        posteriors = [
            [("H", 0.85), ("N", 0.10)],
            [("E", 0.90), ("F", 0.05)],
            [("L", 0.88), ("I", 0.08)],
            [("P", 0.92), ("D", 0.04)]
        ]
        decoded = decoder.beam_search_decode(posteriors, beam_width=4)
        self.assertEqual(decoded[0][0], "HELP")

    def test_gesture_controller_state_machine(self):
        """Test air-writing stroke state machine and dwell time pause detection."""
        ctrl = AirWritingGestureController(dwell_time_sec=0.2, min_stroke_points=5)

        # Start moving -> WRITING
        state, stroke = ctrl.process_point((0.1, 0.1, 0.0))
        self.assertEqual(state, WritingState.WRITING)

        # Continue moving
        for i in range(1, 10):
            state, stroke = ctrl.process_point((0.1 + i * 0.05, 0.1 + i * 0.05, 0.0))
            self.assertEqual(state, WritingState.WRITING)

        # Pause and wait > 0.25s
        import time
        time.sleep(0.25)
        state, stroke = ctrl.process_point((0.55, 0.55, 0.0))
        self.assertEqual(state, WritingState.STROKE_END)
        self.assertIsNotNone(stroke)
        self.assertGreaterEqual(len(stroke), 5)


if __name__ == "__main__":
    unittest.main()
