"""
Air-Writing Dataset Generator for Parkinsonian and Control Motor Patterns.
Generates multi-subject, multi-severity synthetic air-writing datasets
with consistent spatial normalization and temporal resampling.
"""

import os
from typing import Dict, List, Optional, Tuple
import numpy as np

from data.character_templates import ALL_CLASSES, CLASS_TO_IDX, get_canonical_trajectory
from data.parkinson_simulator import ParkinsonSimulator, PDSeverity
from preprocessing.normalizer import TrajectoryNormalizer
from preprocessing.resampler import TrajectoryResampler


class AirWritingDatasetGenerator:
    """
    Generates balanced datasets of air-writing trajectories across multiple subjects
    and Parkinsonian impairment levels.
    """

    def __init__(
        self,
        num_points: int = 64,
        sampling_rate: float = 30.0,
        rng_seed: int = 42
    ):
        self.num_points = num_points
        self.sampling_rate = sampling_rate
        self.simulator = ParkinsonSimulator(sampling_rate=sampling_rate, rng_seed=rng_seed)
        self.rng = np.random.default_rng(rng_seed)

    def generate_sample(
        self,
        char: str,
        severity: PDSeverity = PDSeverity.MODERATE,
        subject_id: Optional[str] = None
    ) -> Tuple[np.ndarray, int, dict]:
        """
        Generate a single air-writing sample for a given character and severity.

        Returns:
            trajectory: (num_points, 3) normalized array
            label: integer class index
            metadata: sample generation metadata
        """
        base_traj = get_canonical_trajectory(char, num_points=self.num_points)
        degraded_traj, meta = self.simulator.apply_parkinson_effects(base_traj, severity=severity)

        # Standardize via resampling and normalization
        norm_traj = TrajectoryResampler.resample_arc_length(degraded_traj, num_points=self.num_points)
        norm_traj = TrajectoryNormalizer.center_and_scale(norm_traj, target_size=1.0, preserve_aspect_ratio=True)

        meta["char"] = char
        meta["label"] = CLASS_TO_IDX[char]
        meta["subject_id"] = subject_id or f"subj_{self.rng.integers(100, 999)}"
        return norm_traj, meta["label"], meta

    def generate_dataset(
        self,
        samples_per_class: int = 50,
        severity_distribution: Optional[Dict[PDSeverity, float]] = None,
        num_subjects: int = 20
    ) -> Dict[str, np.ndarray]:
        """
        Generate a full balanced dataset.
        """
        if severity_distribution is None:
            severity_distribution = {
                PDSeverity.CONTROL: 0.25,
                PDSeverity.MILD: 0.25,
                PDSeverity.MODERATE: 0.30,
                PDSeverity.SEVERE: 0.20,
            }

        severities = list(severity_distribution.keys())
        probs = np.array([severity_distribution[s] for s in severities], dtype=np.float32)
        probs /= np.sum(probs)

        trajectories = []
        labels = []
        sev_list = []
        subj_list = []

        for char in ALL_CLASSES:
            for i in range(samples_per_class):
                sev = self.rng.choice(severities, p=probs)
                subj_id = f"subj_{self.rng.integers(0, num_subjects):03d}"
                traj, lbl, meta = self.generate_sample(char, severity=sev, subject_id=subj_id)
                trajectories.append(traj)
                labels.append(lbl)
                sev_list.append(meta["severity"])
                subj_list.append(subj_id)

        indices = np.arange(len(trajectories))
        self.rng.shuffle(indices)

        return {
            "trajectories": np.array(trajectories, dtype=np.float32)[indices],
            "labels": np.array(labels, dtype=np.int64)[indices],
            "severities": np.array(sev_list, dtype=object)[indices],
            "subject_ids": np.array(subj_list, dtype=object)[indices],
            "classes": np.array(ALL_CLASSES)
        }

    def save_dataset(self, dataset: Dict[str, np.ndarray], filepath: str):
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        np.savez_compressed(filepath, **dataset)
        print(f"Saved dataset with {len(dataset['labels'])} samples to {filepath}")

    @staticmethod
    def load_dataset(filepath: str) -> Dict[str, np.ndarray]:
        data = np.load(filepath, allow_pickle=True)
        return {key: data[key] for key in data.files}
