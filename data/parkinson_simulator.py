"""
Parkinsonian Movement Modeling & Physiological Motion Simulator.
Simulates clinical motor symptoms of Parkinson's Disease on air-writing trajectories:
- 4-7 Hz resting/action tremor with amplitude modulation
- Bradykinesia (movement slowing and non-uniform velocity)
- Micrographia (progressive stroke amplitude shrinkage)
- Rigidity and freezing episodes (hesitations, sudden velocity drops)
- Dysmetria and ataxia (overshoot / undershoot at sharp turning points)
- Tracking sensor noise
"""

import enum
from typing import Optional, Tuple, Union
import numpy as np


class PDSeverity(str, enum.Enum):
    CONTROL = "control"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"


class ParkinsonSimulator:
    """
    Applies biologically plausible Parkinsonian motor symptom degradations
    to clean canonical trajectory sequences.
    """

    def __init__(self, sampling_rate: float = 30.0, rng_seed: Optional[int] = None):
        self.fs = sampling_rate
        self.rng = np.random.default_rng(rng_seed)

    def set_seed(self, seed: int):
        self.rng = np.random.default_rng(seed)

    def apply_parkinson_effects(
        self,
        trajectory: np.ndarray,
        severity: Union[PDSeverity, str] = PDSeverity.MODERATE,
        tremor_freq: Optional[float] = None
    ) -> Tuple[np.ndarray, dict]:
        """
        Apply comprehensive PD motor impairments to a trajectory.
        """
        if isinstance(severity, PDSeverity):
            pass
        elif isinstance(severity, str):
            try:
                severity = PDSeverity(severity.lower())
            except ValueError:
                severity = PDSeverity.MODERATE

        traj = np.copy(trajectory).astype(np.float32)
        N, D = traj.shape
        duration = N / self.fs
        t = np.linspace(0, duration, N, endpoint=False)

        if severity == PDSeverity.CONTROL:
            traj = self._apply_affine_variance(traj, scale_range=(0.9, 1.1), rot_range=(-5, 5))
            traj = self._apply_gaussian_jitter(traj, std=0.005)
            return traj, {"severity": "control", "tremor_freq_hz": 0.0, "micrographia_rate": 0.0}

        # 1. Parameter configurations by severity
        if severity == PDSeverity.MILD:
            tremor_amp = self.rng.uniform(0.015, 0.035)
            f_tremor = tremor_freq or self.rng.uniform(4.5, 6.0)
            micrographia_rate = self.rng.uniform(0.10, 0.22)
            hesitation_prob = 0.20
            overshoot_factor = 0.03
            noise_std = 0.008
        elif severity == PDSeverity.MODERATE:
            tremor_amp = self.rng.uniform(0.035, 0.070)
            f_tremor = tremor_freq or self.rng.uniform(4.0, 6.5)
            micrographia_rate = self.rng.uniform(0.20, 0.38)
            hesitation_prob = 0.45
            overshoot_factor = 0.07
            noise_std = 0.014
        else:  # SEVERE
            tremor_amp = self.rng.uniform(0.070, 0.120)
            f_tremor = tremor_freq or self.rng.uniform(3.8, 7.2)
            micrographia_rate = self.rng.uniform(0.35, 0.55)
            hesitation_prob = 0.70
            overshoot_factor = 0.12
            noise_std = 0.022

        # 2. Apply Micrographia (progressive spatial amplitude shrinkage)
        traj = self._apply_micrographia(traj, shrinkage_rate=micrographia_rate)

        # 3. Apply Bradykinesia & Velocity Hesitation (freezing / pauses)
        traj = self._apply_bradykinesia_and_hesitations(traj, t, hesitation_prob=hesitation_prob)

        # 4. Apply Dysmetria & Ataxia (overshoot at direction turning points)
        traj = self._apply_dysmetria(traj, overshoot_factor=overshoot_factor)

        # 5. Apply Resting/Action Tremor (4-7 Hz oscillatory noise with amplitude envelope)
        traj = self._apply_tremor(traj, t, freq=f_tremor, base_amplitude=tremor_amp)

        # 6. Apply Tracking/Sensor Noise & Sensor Jitter
        traj = self._apply_gaussian_jitter(traj, std=noise_std)

        # 7. Apply random affine transformations (global scale, translation, rotation)
        traj = self._apply_affine_variance(traj, scale_range=(0.85, 1.15), rot_range=(-12, 12))

        metadata = {
            "severity": getattr(severity, "value", str(severity)),
            "tremor_freq_hz": float(f_tremor),
            "tremor_amp": float(tremor_amp),
            "micrographia_rate": float(micrographia_rate),
            "hesitation_prob": float(hesitation_prob),
            "noise_std": float(noise_std)
        }
        return traj, metadata

    def _apply_tremor(
        self, traj: np.ndarray, t: np.ndarray, freq: float, base_amplitude: float
    ) -> np.ndarray:
        N, D = traj.shape
        amp_mod_freq = self.rng.uniform(0.5, 1.5)
        amp_envelope = 1.0 + 0.35 * np.sin(2 * np.pi * amp_mod_freq * t + self.rng.uniform(0, 2*np.pi))

        for d in range(min(D, 2)):
            phase_main = self.rng.uniform(0, 2 * np.pi)
            phase_harm = self.rng.uniform(0, 2 * np.pi)

            tremor_main = np.sin(2 * np.pi * freq * t + phase_main)
            tremor_harm = 0.25 * np.sin(2 * np.pi * (2 * freq) * t + phase_harm)

            component = (tremor_main + tremor_harm) * base_amplitude * amp_envelope
            traj[:, d] += component.astype(np.float32)

        return traj

    def _apply_micrographia(self, traj: np.ndarray, shrinkage_rate: float) -> np.ndarray:
        N, D = traj.shape
        center = np.mean(traj[:, :2], axis=0, keepdims=True)
        scale_factors = np.linspace(1.0, 1.0 - shrinkage_rate, N).reshape(-1, 1)
        traj[:, :2] = center + (traj[:, :2] - center) * scale_factors
        return traj

    def _apply_bradykinesia_and_hesitations(
        self, traj: np.ndarray, t: np.ndarray, hesitation_prob: float
    ) -> np.ndarray:
        N = len(traj)
        if self.rng.random() > hesitation_prob:
            return traj

        num_pauses = self.rng.integers(1, 4)
        for _ in range(num_pauses):
            pause_center = self.rng.integers(int(N * 0.15), int(N * 0.85))
            pause_len = self.rng.integers(2, max(4, int(N * 0.12)))
            start_idx = max(0, pause_center - pause_len // 2)
            end_idx = min(N, pause_center + pause_len // 2)

            if end_idx > start_idx:
                traj[start_idx:end_idx] = traj[start_idx] + self.rng.normal(0, 0.003, traj[start_idx:end_idx].shape)

        return traj

    def _apply_dysmetria(self, traj: np.ndarray, overshoot_factor: float) -> np.ndarray:
        N = len(traj)
        if N < 5:
            return traj

        diffs = np.diff(traj[:, :2], axis=0)
        angles = np.arctan2(diffs[:, 1], diffs[:, 0])
        angle_diffs = np.abs(np.diff(angles))
        angle_diffs = np.where(angle_diffs > np.pi, 2 * np.pi - angle_diffs, angle_diffs)

        turning_points = np.where(angle_diffs > np.pi / 3)[0] + 1
        for idx in turning_points:
            if idx < N - 1:
                momentum = diffs[idx - 1] * overshoot_factor * self.rng.uniform(0.8, 1.5)
                traj[idx, :2] += momentum

        return traj

    def _apply_gaussian_jitter(self, traj: np.ndarray, std: float) -> np.ndarray:
        noise = self.rng.normal(0.0, std, size=traj.shape).astype(np.float32)
        return traj + noise

    def _apply_affine_variance(
        self, traj: np.ndarray, scale_range: Tuple[float, float], rot_range: Tuple[float, float]
    ) -> np.ndarray:
        tx = self.rng.uniform(-0.08, 0.08)
        ty = self.rng.uniform(-0.08, 0.08)
        traj[:, 0] += tx
        traj[:, 1] += ty

        sx = self.rng.uniform(*scale_range)
        sy = self.rng.uniform(*scale_range)
        center = np.mean(traj[:, :2], axis=0, keepdims=True)
        traj[:, 0] = center[0, 0] + (traj[:, 0] - center[0, 0]) * sx
        traj[:, 1] = center[0, 1] + (traj[:, 1] - center[0, 1]) * sy

        angle_deg = self.rng.uniform(*rot_range)
        theta = np.radians(angle_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        R = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float32)

        traj_2d = traj[:, :2] - center
        traj[:, :2] = (traj_2d @ R.T) + center
        return traj
