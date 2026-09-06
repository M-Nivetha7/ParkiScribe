"""
Kinematic, Geometric, and Spectral Feature Extractor.
Extracts rich physiological and movement features from air-writing trajectories
for classical machine learning baselines and clinical tremor analysis.
"""

from typing import Dict, List, Tuple
import numpy as np
from scipy import signal, stats


class TrajectoryFeatureExtractor:
    """
    Extracts high-dimensional kinematic, spectral, and geometric feature vectors
    from a trajectory sequence.
    """

    def __init__(self, fs: float = 30.0):
        self.fs = fs

    def extract_features(self, trajectory: np.ndarray) -> np.ndarray:
        """
        Extract 1D feature vector from a trajectory array of shape (N, 2) or (N, 3).
        Returns a float32 vector suitable for Random Forest / SVM / XGBoost.
        """
        feat_dict = self.extract_feature_dict(trajectory)
        features = []
        for key in sorted(feat_dict.keys()):
            val = feat_dict[key]
            if isinstance(val, (list, np.ndarray)):
                features.extend(list(val))
            else:
                features.append(float(val))
        return np.array(features, dtype=np.float32)

    def extract_feature_dict(self, trajectory: np.ndarray) -> Dict[str, any]:
        """
        Extract structured dictionary of kinematic, spectral, and geometric features.
        """
        traj = np.asarray(trajectory, dtype=np.float32)
        N = len(traj)
        dt = 1.0 / self.fs
        feats = {}

        # 1. Kinematic First Derivatives: Velocity
        diffs = np.diff(traj[:, :2], axis=0)
        vel = diffs / dt  # (N-1, 2)
        speed = np.linalg.norm(vel, axis=1)  # (N-1,)

        feats["vel_mean_speed"] = float(np.mean(speed)) if len(speed) > 0 else 0.0
        feats["vel_std_speed"] = float(np.std(speed)) if len(speed) > 0 else 0.0
        feats["vel_max_speed"] = float(np.max(speed)) if len(speed) > 0 else 0.0
        feats["vel_median_speed"] = float(np.median(speed)) if len(speed) > 0 else 0.0
        feats["vel_skewness"] = float(stats.skew(speed)) if len(speed) > 2 else 0.0

        # 2. Kinematic Second Derivatives: Acceleration
        if len(vel) > 1:
            acc = np.diff(vel, axis=0) / dt  # (N-2, 2)
            acc_mag = np.linalg.norm(acc, axis=1)
            feats["acc_mean"] = float(np.mean(acc_mag))
            feats["acc_std"] = float(np.std(acc_mag))
            feats["acc_max"] = float(np.max(acc_mag))
            feats["acc_rms"] = float(np.sqrt(np.mean(acc_mag**2)))
        else:
            feats["acc_mean"] = feats["acc_std"] = feats["acc_max"] = feats["acc_rms"] = 0.0

        # 3. Kinematic Third Derivatives: Jerk (Key biomarker in PD motor rigidity)
        if len(vel) > 2:
            jerk = np.diff(acc, axis=0) / dt  # (N-3, 2)
            jerk_mag = np.linalg.norm(jerk, axis=1)
            mean_sq_jerk = np.mean(jerk_mag**2)
            feats["jerk_mean_sq"] = float(mean_sq_jerk)
            total_duration = N * dt
            total_path = np.sum(np.linalg.norm(diffs, axis=1)) + 1e-6
            # Dimensionless normalized jerk
            feats["jerk_normalized"] = float(np.sqrt(0.5 * np.sum(jerk_mag**2) * (total_duration**5) / (total_path**2)))
        else:
            feats["jerk_mean_sq"] = feats["jerk_normalized"] = 0.0

        # 4. Curvature & Angular Velocity
        if len(vel) > 1:
            dx = vel[:-1, 0]
            dy = vel[:-1, 1]
            ddx = acc[:, 0]
            ddy = acc[:, 1]
            denom = (dx**2 + dy**2)**1.5 + 1e-6
            curvature = np.abs(dx * ddy - dy * ddx) / denom
            feats["curv_mean"] = float(np.mean(curvature))
            feats["curv_max"] = float(np.max(curvature))
            feats["curv_bending_energy"] = float(np.sum(curvature**2))

            angles = np.arctan2(vel[:, 1], vel[:, 0])
            angle_diffs = np.abs(np.diff(angles))
            angle_diffs = np.where(angle_diffs > np.pi, 2 * np.pi - angle_diffs, angle_diffs)
            feats["angular_vel_mean"] = float(np.mean(angle_diffs) / dt) if len(angle_diffs) > 0 else 0.0
            feats["direction_reversals"] = float(np.sum(angle_diffs > np.pi / 2))
        else:
            feats["curv_mean"] = feats["curv_max"] = feats["curv_bending_energy"] = 0.0
            feats["angular_vel_mean"] = feats["direction_reversals"] = 0.0

        # 5. Spectral & Tremor Analysis (4-7 Hz band power)
        freqs = np.fft.rfftfreq(N, d=dt)
        fft_x = np.abs(np.fft.rfft(traj[:, 0]))**2
        fft_y = np.abs(np.fft.rfft(traj[:, 1]))**2
        power = fft_x + fft_y

        tremor_mask = (freqs >= 4.0) & (freqs <= 7.0)
        total_power = np.sum(power) + 1e-8
        tremor_power = np.sum(power[tremor_mask])
        feats["tremor_power_ratio_4_7hz"] = float(tremor_power / total_power)

        if np.any(tremor_mask) and np.max(power[tremor_mask]) > 0:
            peak_idx = np.argmax(power[tremor_mask])
            feats["tremor_peak_freq_hz"] = float(freqs[tremor_mask][peak_idx])
        else:
            feats["tremor_peak_freq_hz"] = 0.0

        # 6. Geometric & Spatial Morphology
        min_xy = np.min(traj[:, :2], axis=0)
        max_xy = np.max(traj[:, :2], axis=0)
        bbox_w = float(max_xy[0] - min_xy[0])
        bbox_h = float(max_xy[1] - min_xy[1])
        feats["bbox_width"] = bbox_w
        feats["bbox_height"] = bbox_h
        feats["bbox_aspect_ratio"] = float(bbox_w / (bbox_h + 1e-6))
        feats["bbox_diagonal"] = float(np.sqrt(bbox_w**2 + bbox_h**2))

        # Path efficiency (displacement / total arc length)
        endpoint_dist = float(np.linalg.norm(traj[-1, :2] - traj[0, :2]))
        total_arc = float(np.sum(np.linalg.norm(diffs, axis=1))) + 1e-6
        feats["path_efficiency"] = float(endpoint_dist / total_arc)
        feats["total_path_length"] = total_arc

        # Micrographia index: Ratio of bounding box area in first 25% vs last 25% of trajectory
        q1_end = max(2, N // 4)
        q4_start = min(N - 2, 3 * N // 4)
        area_q1 = (np.ptp(traj[:q1_end, 0]) + 1e-4) * (np.ptp(traj[:q1_end, 1]) + 1e-4)
        area_q4 = (np.ptp(traj[q4_start:, 0]) + 1e-4) * (np.ptp(traj[q4_start:, 1]) + 1e-4)
        feats["micrographia_area_ratio"] = float(area_q4 / area_q1)

        # 7. Direction Histogram (8 bins)
        if len(vel) > 0:
            angles = np.arctan2(vel[:, 1], vel[:, 0])  # [-pi, pi]
            hist, _ = np.histogram(angles, bins=8, range=(-np.pi, np.pi), density=True)
            for b in range(8):
                feats[f"dir_hist_bin_{b}"] = float(hist[b])
        else:
            for b in range(8):
                feats[f"dir_hist_bin_{b}"] = 0.0

        return feats
