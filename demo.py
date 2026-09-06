"""
Simulation and Headless Demonstration Script for:
AI-Based Air Writing Recognition and Parkinson's Disease Motor Analysis System.

Simulates handwriting trajectories for Control vs. Parkinsonian conditions (Mild, Moderate, Severe),
renders them onto the virtual AirCanvas, performs image preprocessing, predicts character via CNN,
and calculates quantitative motor biomarkers.
"""

import os
import sys
import argparse
import numpy as np
import cv2

# Project root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from utils.constants import ALL_CLASSES, MODEL_INPUT_SIZE
from data.character_templates import get_canonical_trajectory
from data.parkinson_simulator import ParkinsonSimulator, PDSeverity
from src.drawing_utils import AirCanvas
from src.preprocessing import preprocess_canvas
from src.recognizer import AirWritingRecognizer
from src.motor_analysis import ParkinsonMotorAnalyzer
from main import draw_hud


def run_simulation(
    char: str = "A",
    severity_str: str = "moderate",
    save_output: bool = True,
    output_dir: str = "outputs"
):
    char = char.upper()
    if char not in ALL_CLASSES:
        print(f"[ERROR] Character '{char}' is not in valid classes (0-9, A-Z).")
        return

    try:
        severity = PDSeverity(severity_str.lower())
    except ValueError:
        severity = PDSeverity.MODERATE

    print(f"\n=======================================================")
    print(f"  Simulating Air-Writing for Character: '{char}'")
    print(f"  Condition / Severity: {severity.value.upper()}")
    print(f"=======================================================")

    # 1. Generate canonical trajectory & apply Parkinsonian effects
    simulator = ParkinsonSimulator(sampling_rate=30.0, rng_seed=42)
    base_traj = get_canonical_trajectory(char, num_points=64)
    traj_degraded, meta = simulator.apply_parkinson_effects(base_traj, severity=severity)

    # 2. Simulate virtual AirCanvas drawing
    frame_h, frame_w = 480, 640
    mock_frame = np.full((frame_h, frame_w, 3), 30, dtype=np.uint8)  # dark background

    canvas = AirCanvas()
    canvas.initialize_canvas(mock_frame)

    # Map normalized coordinates [0, 1] to screen pixel coordinates
    margin = 100
    x_coords = traj_degraded[:, 0]
    y_coords = traj_degraded[:, 1]
    scale_x = frame_w - 2 * margin
    scale_y = frame_h - 2 * margin

    screen_x = margin + (x_coords * scale_x).astype(np.int32)
    screen_y = margin + (y_coords * scale_y).astype(np.int32)

    for i in range(len(screen_x)):
        canvas.draw(mock_frame, int(screen_x[i]), int(screen_y[i]), timestamp=float(i) / 30.0)

    # 3. Handwriting Preprocessing (Module 8)
    norm_img, prep_meta = preprocess_canvas(canvas.canvas, target_size=MODEL_INPUT_SIZE)
    print("\n[Module 8: Preprocessing]")
    print(f"  Bounding box: {prep_meta.get('bbox')}")
    print(f"  Preprocessed image shape: {norm_img.shape} (Range: [{norm_img.min():.2f}, {norm_img.max():.2f}])")

    # 4. Deep Learning Character Recognition (Module 9 & 10)
    recognizer = AirWritingRecognizer()
    prediction, _ = recognizer.predict_canvas(canvas.canvas)

    print("\n[Module 9 & 10: AI Character Recognition]")
    if prediction:
        print(f"  Predicted Character : '{prediction['predicted_char']}'")
        print(f"  Confidence          : {prediction['confidence']:.1f}%")
        print("  Top Candidates      :")
        for cand in prediction["top_candidates"]:
            print(f"    - '{cand['char']}': {cand['confidence']:.1f}%")
    else:
        print("  Could not recognize character.")

    # 5. Parkinson's Motor Pattern Analysis (Module 11)
    analyzer = ParkinsonMotorAnalyzer(sampling_rate=30.0)
    biomarkers = analyzer.analyze_trajectory(canvas.get_trajectory_array(), canvas.get_strokes())

    print("\n[Module 11: Parkinson's Motor Pattern Analysis]")
    print(biomarkers.summary_text())

    # 6. Render HUD and save snapshot
    merged = canvas.merge_canvas(mock_frame)
    hud_frame = draw_hud(
        merged,
        gesture_state="draw",
        prediction=prediction,
        biomarkers=biomarkers,
        preprocessed_thumb=prep_meta.get("segmented")
    )

    if save_output:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"sim_{char}_{severity.value}.png")
        cv2.imwrite(out_path, hud_frame)
        print(f"\n[Saved Output Visualization]: {out_path}")

    return {
        "char": char,
        "severity": severity.value,
        "prediction": prediction,
        "biomarkers": biomarkers.to_dict()
    }


def compare_severities(char: str = "A"):
    """Compare motor biomarkers across Control, Mild, Moderate, and Severe PD."""
    print(f"\n=== Comparing Impairment Tiers for Character '{char}' ===")
    severities = ["control", "mild", "moderate", "severe"]
    results = []

    print(f"{'Severity':<10} | {'Pred':<5} | {'Conf':<6} | {'Smoothness':<11} | {'Tremor (4-7Hz)':<15} | {'Variation':<10}")
    print("-" * 75)

    for s in severities:
        res = run_simulation(char=char, severity_str=s, save_output=False)
        pred_c = res["prediction"]["predicted_char"] if res["prediction"] else "?"
        conf = f"{res['prediction']['confidence']:.1f}%" if res["prediction"] else "0%"
        smooth = f"{res['biomarkers']['stroke_smoothness_score']:.1f}%"
        tremor = f"{res['biomarkers']['tremor_intensity_score']:.1f}%"
        var = res["biomarkers"]["trajectory_variation"]
        print(f"{s.upper():<10} | {pred_c:<5} | {conf:<6} | {smooth:<11} | {tremor:<15} | {var:<10}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Air-Writing & Parkinson's Simulation Demo")
    parser.add_argument("--char", type=str, default="A", help="Character to write (0-9, A-Z)")
    parser.add_argument("--severity", type=str, default="moderate", choices=["control", "mild", "moderate", "severe"], help="PD impairment level")
    parser.add_argument("--compare", action="store_true", help="Compare all impairment levels")
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args()

    if args.compare:
        compare_severities(args.char)
    else:
        run_simulation(char=args.char, severity_str=args.severity, save_output=True, output_dir=args.output_dir)
