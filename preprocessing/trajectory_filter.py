"""
Trajectory Filtering & Denoising Module.
Contains tremor-aware and noise-reduction filters:
- Savitzky-Golay filter (preserves intentional geometry and local extrema)
- Butterworth lowpass filter (removes high-frequency tracking sensor noise while preserving 4-7 Hz PD tremor band)
- Kinematic Kalman filter (smooths noisy optical tracking coordinates)
"""

import numpy as np
from scipy import signal


class TrajectoryFilter:
    """
    Filters air-writing trajectories to remove high-frequency tracking jitter
    while preserving intentional writing kinematics and motor characteristics.
    """

    @staticmethod
    def savitzky_golay(
        trajectory: np.ndarray,
        window_length: int = 7,
        polyorder: int = 2
    ) -> np.ndarray:
        """
        Apply Savitzky-Golay polynomial smoothing filter.
        """
        N, D = trajectory.shape
        if N <= window_length:
            window_length = max(3, N if N % 2 == 1 else N - 1)
            polyorder = min(polyorder, window_length - 1)

        if polyorder >= window_length:
            polyorder = window_length - 1

        smoothed = np.zeros_like(trajectory)
        for d in range(D):
            smoothed[:, d] = signal.savgol_filter(trajectory[:, d], window_length, polyorder)
        return smoothed

    @staticmethod
    def butterworth_lowpass(
        trajectory: np.ndarray,
        cutoff_hz: float = 10.0,
        fs: float = 30.0,
        order: int = 3
    ) -> np.ndarray:
        """
        Apply zero-phase Butterworth lowpass filter.
        Cutoff is typically 8-12 Hz to attenuate >15 Hz camera tracking noise
        without erasing 4-7 Hz Parkinsonian tremor dynamics.
        """
        N, D = trajectory.shape
        nyq = 0.5 * fs
        normal_cutoff = min(0.95, cutoff_hz / nyq)

        b, a = signal.butter(order, normal_cutoff, btype="low", analog=False)

        if N <= 3 * max(len(a), len(b)):
            # If trajectory is very short, fallback to lighter smoothing
            return TrajectoryFilter.savitzky_golay(trajectory, window_length=min(5, N if N % 2 == 1 else N - 1))

        smoothed = np.zeros_like(trajectory)
        for d in range(D):
            smoothed[:, d] = signal.filtfilt(b, a, trajectory[:, d])
        return smoothed

    @staticmethod
    def kalman_smooth(
        trajectory: np.ndarray,
        process_noise: float = 1e-3,
        measurement_noise: float = 1e-2
    ) -> np.ndarray:
        """
        Constant-velocity kinematic Kalman filter for 2D/3D trajectory points.
        """
        N, D = trajectory.shape
        dt = 1.0 / 30.0

        smoothed = np.zeros_like(trajectory)
        for d in range(D):
            # State vector: [position, velocity]
            x = np.array([[trajectory[0, d]], [0.0]])
            F = np.array([[1.0, dt], [0.0, 1.0]])
            H = np.array([[1.0, 0.0]])
            Q = np.array([[0.25 * dt**4, 0.5 * dt**3], [0.5 * dt**3, dt**2]]) * process_noise
            R = np.array([[measurement_noise]])
            P = np.eye(2) * 1.0

            for i in range(N):
                # Predict
                x = F @ x
                P = F @ P @ F.T + Q

                # Update
                z = np.array([[trajectory[i, d]]])
                y = z - H @ x
                S = H @ P @ H.T + R
                K = P @ H.T @ np.linalg.inv(S)
                x = x + K @ y
                P = (np.eye(2) - K @ H) @ P

                smoothed[i, d] = x[0, 0]

        return smoothed
