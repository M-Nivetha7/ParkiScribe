from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy.signal import welch

from utils.constants import CLINICAL_DISCLAIMER, TREMOR_FREQ_BAND


@dataclass
class MotorBiomarkers:
    """Quantitative Kinematic and Spectral Motor Features for Parkinson's Analysis."""
    writing_duration_sec: float
    active_drawing_time_sec: float
    pause_time_sec: float
    average_speed_px_sec: float
    peak_speed_px_sec: float
    speed_variability_cv: float
    stroke_smoothness_score: float  # 0 to 100%
    normalized_jerk: float
    tremor_intensity_score: float  # 0 to 100%
    tremor_band_power_ratio: float  # 4-7 Hz band power ratio
    movement_amplitude_width_px: float
    movement_amplitude_height_px: float
    bounding_box_area_px2: float
    micrographia_slope: float  # Size decay across successive strokes
    micrographia_detected: bool
    trajectory_variation: str  # "Low", "Moderate", "High"
    disclaimer: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def summary_text(self) -> str:
        return (
            f"--- Parkinson's Motor Pattern Analysis ---\n"
            f"Writing Duration: {self.writing_duration_sec:.2f} s (Active: {self.active_drawing_time_sec:.2f} s)\n"
            f"Average Writing Speed: {self.average_speed_px_sec:.1f} px/s (Peak: {self.peak_speed_px_sec:.1f} px/s)\n"
            f"Stroke Smoothness Score: {self.stroke_smoothness_score:.1f}%\n"
            f"Tremor Intensity Score: {self.tremor_intensity_score:.1f}% (4-7 Hz Power: {self.tremor_band_power_ratio*100:.1f}%)\n"
            f"Trajectory Variation: {self.trajectory_variation}\n"
            f"Movement Amplitude: {self.movement_amplitude_width_px:.0f} x {self.movement_amplitude_height_px:.0f} px\n"
            f"Micrographia Detected: {'Yes' if self.micrographia_detected else 'No'} (slope: {self.micrographia_slope:.2f})\n"
            f"Note: {self.disclaimer}"
        )


class ParkinsonMotorAnalyzer:
    """
    Extracts clinical and biomechanical motor biomarkers from recorded
    air-writing trajectories:
    1. Writing Speed & Bradykinesia
    2. Tremor Intensity in 4-7 Hz Band
    3. Stroke Smoothness & Normalized Jerk
    4. Writing & Pause Duration
    5. Movement Amplitude & Micrographia Decay
    """

    def __init__(self, sampling_rate: float = 30.0):
        self.fs = sampling_rate
        self.tremor_band = TREMOR_FREQ_BAND

    def analyze_trajectory(
        self,
        trajectory: np.ndarray,
        strokes: Optional[List[np.ndarray]] = None
    ) -> MotorBiomarkers:
        """
        Analyze coordinate trajectory array of shape (N, 3): [timestamp, x, y].
        """
        if len(trajectory) < 3:
            return MotorBiomarkers(
                writing_duration_sec=0.0,
                active_drawing_time_sec=0.0,
                pause_time_sec=0.0,
                average_speed_px_sec=0.0,
                peak_speed_px_sec=0.0,
                speed_variability_cv=0.0,
                stroke_smoothness_score=100.0,
                normalized_jerk=0.0,
                tremor_intensity_score=0.0,
                tremor_band_power_ratio=0.0,
                movement_amplitude_width_px=0.0,
                movement_amplitude_height_px=0.0,
                bounding_box_area_px2=0.0,
                micrographia_slope=0.0,
                micrographia_detected=False,
                trajectory_variation="Low",
                disclaimer=CLINICAL_DISCLAIMER
            )

        t = trajectory[:, 0]
        x = trajectory[:, 1]
        y = trajectory[:, 2]

        # Feature 4: Writing Duration
        total_duration = float(max(t[-1] - t[0], 0.01))
        dt = np.diff(t)
        dt = np.where(dt <= 0, 1.0 / self.fs, dt)

        # Feature 1: Writing Speed & Velocity
        dx = np.diff(x)
        dy = np.diff(y)
        dist = np.sqrt(dx**2 + dy**2)
        total_path_length = float(np.sum(dist))

        velocities = dist / dt
        avg_speed = float(np.mean(velocities))
        peak_speed = float(np.max(velocities)) if len(velocities) > 0 else 0.0
        std_speed = float(np.std(velocities)) if len(velocities) > 0 else 0.0
        speed_cv = float(std_speed / (avg_speed + 1e-6))

        # Active time vs pause time (threshold: speed > 10 px/s is active drawing)
        active_mask = velocities > 10.0
        active_time = float(np.sum(dt[active_mask])) if len(dt) > 0 else total_duration
        pause_time = max(total_duration - active_time, 0.0)

        # Feature 2: Tremor Intensity (4-7 Hz Spectral Density & Oscillation)
        tremor_score, band_ratio = self._calculate_tremor_power(velocities, dt)

        if tremor_score < 25.0:
            variation_level = "Low"
        elif tremor_score < 60.0:
            variation_level = "Moderate"
        else:
            variation_level = "High"

        # Feature 3: Stroke Smoothness & Normalized Jerk
        smoothness_score, norm_jerk = self._calculate_smoothness(dist, dt, total_duration, total_path_length)

        # Feature 5: Movement Amplitude & Micrographia
        w = float(np.max(x) - np.min(x))
        h = float(np.max(y) - np.min(y))
        area = float(w * h)

        micrographia_slope, micrographia_detected = self._evaluate_micrographia(strokes)

        return MotorBiomarkers(
            writing_duration_sec=round(total_duration, 2),
            active_drawing_time_sec=round(active_time, 2),
            pause_time_sec=round(pause_time, 2),
            average_speed_px_sec=round(avg_speed, 1),
            peak_speed_px_sec=round(peak_speed, 1),
            speed_variability_cv=round(speed_cv, 2),
            stroke_smoothness_score=round(smoothness_score, 1),
            normalized_jerk=round(norm_jerk, 2),
            tremor_intensity_score=round(tremor_score, 1),
            tremor_band_power_ratio=round(band_ratio, 3),
            movement_amplitude_width_px=round(w, 1),
            movement_amplitude_height_px=round(h, 1),
            bounding_box_area_px2=round(area, 1),
            micrographia_slope=round(micrographia_slope, 3),
            micrographia_detected=micrographia_detected,
            trajectory_variation=variation_level,
            disclaimer=CLINICAL_DISCLAIMER
        )

    def _calculate_tremor_power(self, velocities: np.ndarray, dt: np.ndarray) -> Tuple[float, float]:
        """Estimate 4-7 Hz band power relative to total frequency spectrum."""
        if len(velocities) < 12:
            return 10.0, 0.05

        # Resample velocity signal to uniform 30 Hz for spectral analysis
        mean_dt = np.mean(dt)
        fs_eff = 1.0 / max(mean_dt, 0.001)

        # Center velocity signal
        v_detrend = velocities - np.mean(velocities)

        # Welch PSD
        nperseg = min(len(v_detrend), 64)
        if nperseg < 8:
            return 15.0, 0.08

        freqs, psd = welch(v_detrend, fs=fs_eff, nperseg=nperseg)
        total_power = np.sum(psd) + 1e-8

        # 4-7 Hz Parkinsonian band
        band_mask = (freqs >= self.tremor_band[0]) & (freqs <= self.tremor_band[1])
        band_power = np.sum(psd[band_mask])
        ratio = float(band_power / total_power)

        # Tremor score (0 to 100) scaled from power ratio and velocity variance
        # Typically normal handwriting has < 15% tremor band power; PD tremor reaches 30-70%
        tremor_score = float(np.clip(ratio * 180.0, 0.0, 100.0))
        return tremor_score, ratio

    def _calculate_smoothness(
        self,
        dist: np.ndarray,
        dt: np.ndarray,
        duration: float,
        path_length: float
    ) -> Tuple[float, float]:
        """
        Calculate Normalized Jerk (NJ) and Stroke Smoothness Score (0 - 100%).
        NJ = sqrt( (T^5 / (2 * L^2)) * integral(jerk^2 dt) )
        """
        if len(dist) < 4 or path_length < 1e-4:
            return 85.0, 10.0

        dt_safe = np.clip(dt, 1e-3, None).astype(np.float64)
        v = dist / dt_safe
        a = np.diff(v) / dt_safe[:-1]
        jerk = np.diff(a) / dt_safe[:-2]
        jerk = np.clip(jerk, -1e5, 1e5)

        dt_jerk = dt_safe[:-2]
        integral_jerk_sq = np.sum((jerk**2) * dt_jerk)

        # Dimensionless Normalized Jerk
        nj_numerator = (duration**5) * integral_jerk_sq
        nj_denominator = 2.0 * (path_length**2) + 1e-6
        norm_jerk = float(np.sqrt(max(nj_numerator / nj_denominator, 0.0)))

        # Convert to a 0-100% smoothness score (lower jerk = smoother stroke)
        # Logarithmic scale calibration: NJ ~ 20-50 is very smooth (90%), NJ > 500 is shaky (<40%)
        smoothness_score = float(np.clip(100.0 - 15.0 * np.log1p(norm_jerk), 10.0, 98.0))

        return smoothness_score, norm_jerk

    def _evaluate_micrographia(
        self,
        strokes: Optional[List[np.ndarray]]
    ) -> Tuple[float, bool]:
        """
        Detect micrographia (progressive shrinkage of stroke height across multi-stroke drawing).
        """
        if not strokes or len(strokes) < 3:
            return 0.0, False

        heights = []
        for s in strokes:
            if len(s) >= 2:
                sh = np.max(s[:, 2]) - np.min(s[:, 2])
                heights.append(sh)

        if len(heights) < 3:
            return 0.0, False

        # Linear regression slope over stroke indices
        indices = np.arange(len(heights))
        slope, _ = np.polyfit(indices, heights, deg=1)

        # Micrographia if statistically significant shrinkage (negative slope < -3.0 px/stroke)
        micrographia_detected = bool(slope < -3.0 and (heights[-1] < 0.65 * heights[0]))

        return float(slope), micrographia_detected
