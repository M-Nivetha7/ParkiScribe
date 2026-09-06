"""
Spatial Normalization Module for Air-Writing Trajectories.
Normalizes translation, scale, and bounding box while preserving aspect ratio.
"""

import numpy as np


class TrajectoryNormalizer:
    """
    Normalizes spatial coordinates of air-writing trajectories.
    """

    @staticmethod
    def center_and_scale(
        trajectory: np.ndarray,
        target_size: float = 1.0,
        preserve_aspect_ratio: bool = True
    ) -> np.ndarray:
        """
        Translate trajectory centroid to (0.5, 0.5) and scale into [0, target_size].
        """
        traj = np.copy(trajectory).astype(np.float32)
        min_vals = np.min(traj[:, :2], axis=0)
        max_vals = np.max(traj[:, :2], axis=0)
        span = max_vals - min_vals

        if preserve_aspect_ratio:
            max_span = np.max(span)
            if max_span < 1e-6:
                traj[:, :2] = target_size / 2.0
            else:
                scale = target_size * 0.8 / max_span
                centroid = (min_vals + max_vals) / 2.0
                traj[:, :2] = (traj[:, :2] - centroid) * scale + (target_size / 2.0)
        else:
            span = np.where(span < 1e-6, 1.0, span)
            traj[:, :2] = (traj[:, :2] - min_vals) / span * target_size

        # Clamp to bounds [0, target_size]
        traj[:, :2] = np.clip(traj[:, :2], 0.0, target_size)

        # Normalize z dimension if present
        if traj.shape[1] > 2:
            min_z, max_z = np.min(traj[:, 2]), np.max(traj[:, 2])
            span_z = max_z - min_z
            if span_z > 1e-6:
                traj[:, 2] = (traj[:, 2] - min_z) / span_z
            else:
                traj[:, 2] = 0.0
            traj[:, 2] = np.clip(traj[:, 2], 0.0, 1.0)

        return traj

    @staticmethod
    def z_score_normalize(trajectory: np.ndarray) -> np.ndarray:
        """Z-score normalization (zero mean, unit variance)."""
        traj = np.copy(trajectory).astype(np.float32)
        mean = np.mean(traj, axis=0, keepdims=True)
        std = np.std(traj, axis=0, keepdims=True)
        std = np.where(std < 1e-6, 1.0, std)
        return (traj - mean) / std
