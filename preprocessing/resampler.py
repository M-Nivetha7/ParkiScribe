"""
Trajectory Resampling Module.
Performs uniform arc-length and temporal interpolation on stroke sequences.
"""

import numpy as np


class TrajectoryResampler:
    """
    Resamples variable-length trajectory sequences into fixed-length arrays.
    """

    @staticmethod
    def resample_arc_length(trajectory: np.ndarray, num_points: int = 64) -> np.ndarray:
        """
        Resample points uniformly along cumulative arc-length.
        """
        traj = np.asarray(trajectory, dtype=np.float32)
        N, D = traj.shape
        if N < 2:
            return np.repeat(traj[:1], num_points, axis=0)

        diffs = np.diff(traj[:, :2], axis=0)
        seg_lens = np.linalg.norm(diffs, axis=1)
        total_len = np.sum(seg_lens)

        if total_len < 1e-6:
            return np.repeat(traj[:1], num_points, axis=0)

        cum_dist = np.insert(np.cumsum(seg_lens), 0, 0.0)
        target_dist = np.linspace(0.0, total_len, num_points)

        resampled = np.zeros((num_points, D), dtype=np.float32)
        for d in range(D):
            resampled[:, d] = np.interp(target_dist, cum_dist, traj[:, d])

        return resampled

    @staticmethod
    def resample_temporal(trajectory: np.ndarray, num_points: int = 64) -> np.ndarray:
        """
        Resample points uniformly along the time dimension.
        """
        traj = np.asarray(trajectory, dtype=np.float32)
        N, D = traj.shape
        if N < 2:
            return np.repeat(traj[:1], num_points, axis=0)

        t_orig = np.linspace(0.0, 1.0, N)
        t_target = np.linspace(0.0, 1.0, num_points)

        resampled = np.zeros((num_points, D), dtype=np.float32)
        for d in range(D):
            resampled[:, d] = np.interp(t_target, t_orig, traj[:, d])

        return resampled
