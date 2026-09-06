"""
Ensemble Air-Writing Recognizer & Inference Engine.
Combines Deep Learning (CNN-BiLSTM-Attention, Transformer) and Classical ML baselines
with adaptive calibration and preprocessing pipelines.
"""

import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F

from data.character_templates import ALL_CLASSES, IDX_TO_CLASS, NUM_CLASSES
from preprocessing.trajectory_filter import TrajectoryFilter
from preprocessing.normalizer import TrajectoryNormalizer
from preprocessing.resampler import TrajectoryResampler
from preprocessing.feature_extractor import TrajectoryFeatureExtractor
from models.conv1d_bilstm import Conv1DBiLSTMAttention
from models.transformer_model import TrajectoryTransformer
from models.baseline_classifier import BaselineTrajectoryClassifier


class AirWritingRecognizer:
    """
    End-to-End Recognizer combining deep neural networks and kinematic models
    for Parkinsonian air-writing character and word decoding.
    """

    def __init__(
        self,
        bilstm_model: Optional[Conv1DBiLSTMAttention] = None,
        transformer_model: Optional[TrajectoryTransformer] = None,
        baseline_model: Optional[BaselineTrajectoryClassifier] = None,
        device: Optional[torch.device] = None
    ):
        self.device = device or torch.device("cpu")
        self.bilstm_model = bilstm_model
        self.transformer_model = transformer_model
        self.baseline_model = baseline_model

        if self.bilstm_model:
            self.bilstm_model.to(self.device).eval()
        if self.transformer_model:
            self.transformer_model.to(self.device).eval()

        self.filter = TrajectoryFilter()
        self.normalizer = TrajectoryNormalizer()
        self.resampler = TrajectoryResampler()
        self.feature_extractor = TrajectoryFeatureExtractor()

        self.user_class_weights = np.ones(NUM_CLASSES, dtype=np.float32)

    def preprocess_trajectory(
        self,
        raw_trajectory: np.ndarray,
        apply_filtering: bool = True
    ) -> np.ndarray:
        """
        Full preprocessing pipeline:
        1. Adaptive Butterworth / Savitzky-Golay filtering
        2. Arc-length / temporal resampling to 64 points
        3. Aspect-ratio preserving spatial centering and normalization
        """
        traj = np.asarray(raw_trajectory, dtype=np.float32)
        if traj.ndim != 2 or len(traj) < 2:
            return traj

        # If 2D, pad with 0 for z
        if traj.shape[1] == 2:
            z = np.zeros((len(traj), 1), dtype=np.float32)
            traj = np.hstack([traj, z])

        # 1. Trajectory Denoising
        if apply_filtering and len(traj) >= 5:
            traj = self.filter.butterworth_lowpass(traj, cutoff_hz=10.0)
            traj = self.filter.savitzky_golay(traj, window_length=5)

        # 2. Uniform temporal & arc-length resampling to 64 points
        traj = self.resampler.resample_arc_length(traj, num_points=64)

        # 3. Spatial bounding box normalization & centering
        traj = self.normalizer.center_and_scale(traj, target_size=1.0, preserve_aspect_ratio=True)

        return traj

    def predict_probabilities(
        self,
        raw_trajectory: np.ndarray,
        weights: Optional[Dict[str, float]] = None
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, any]]:
        """
        Compute ensemble posterior probability distribution P(y|T).
        """
        clean_traj = self.preprocess_trajectory(raw_trajectory)

        if weights is None:
            weights = {"bilstm": 0.60, "transformer": 0.35, "baseline": 0.05}

        total_prob = np.zeros(NUM_CLASSES, dtype=np.float32)
        total_weight = 0.0
        attn_weights = np.zeros(len(clean_traj), dtype=np.float32)

        # 1. Conv1D-BiLSTM prediction
        if self.bilstm_model is not None and weights.get("bilstm", 0) > 0:
            w = weights["bilstm"]
            x_in = torch.from_numpy(clean_traj).transpose(0, 1).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits, attn = self.bilstm_model(x_in)
                probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
                attn_weights = attn.squeeze().cpu().numpy()
            total_prob += w * probs
            total_weight += w

        # 2. Transformer prediction
        if self.transformer_model is not None and weights.get("transformer", 0) > 0:
            w = weights["transformer"]
            x_in = torch.from_numpy(clean_traj).transpose(0, 1).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits, _ = self.transformer_model(x_in)
                probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            total_prob += w * probs
            total_weight += w

        # 3. Baseline ML prediction
        if self.baseline_model is not None and weights.get("baseline", 0) > 0:
            w = weights["baseline"]
            try:
                probs = self.baseline_model.predict_proba(np.array([clean_traj]))[0]
                total_prob += w * probs
                total_weight += w
            except Exception:
                pass

        if total_weight > 0:
            total_prob /= total_weight
        else:
            total_prob = np.ones(NUM_CLASSES) / NUM_CLASSES

        total_prob *= self.user_class_weights
        total_prob /= (np.sum(total_prob) + 1e-8)

        kinematic_meta = self.feature_extractor.extract_feature_dict(clean_traj)

        return total_prob, attn_weights, kinematic_meta

    def predict(
        self,
        raw_trajectory: np.ndarray,
        top_k: int = 5
    ) -> Dict[str, any]:
        """
        Predict intended character and return structured result.
        """
        probs, attn_weights, kin_meta = self.predict_probabilities(raw_trajectory)
        top_idx = np.argsort(probs)[::-1][:top_k]

        candidates = [
            {"char": IDX_TO_CLASS[i], "confidence": float(probs[i])}
            for i in top_idx
        ]

        top_pred = candidates[0]
        return {
            "predicted_char": top_pred["char"],
            "confidence": top_pred["confidence"],
            "candidates": candidates,
            "attention_weights": attn_weights.tolist() if isinstance(attn_weights, np.ndarray) else [],
            "kinematics": kin_meta
        }
