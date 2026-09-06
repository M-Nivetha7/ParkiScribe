"""
Interactive CLI Demo for Air-Writing Recognition in Parkinson's Disease.
Simulates character air-writing, computes real-time predictions,
visualizes attention & kinematics in terminal, and performs assistive word completion.
"""

import os
import sys
import time
import torch
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from data.character_templates import ALL_CLASSES, NUM_CLASSES
from data.parkinson_simulator import PDSeverity, ParkinsonSimulator
from data.dataset_generator import AirWritingDatasetGenerator
from models.conv1d_bilstm import Conv1DBiLSTMAttention
from models.transformer_model import TrajectoryTransformer
from models.baseline_classifier import BaselineTrajectoryClassifier
from models.ensemble_recognizer import AirWritingRecognizer
from nlp.word_recognizer import AssistiveNLPDecoder


def render_ascii_trajectory(trajectory: np.ndarray, width: int = 40, height: int = 16) -> str:
    """Render a 2D ASCII visualization of the stroke trajectory."""
    grid = [[" " for _ in range(width)] for _ in range(height)]
    pts = np.asarray(trajectory, dtype=np.float32)

    min_xy = np.min(pts[:, :2], axis=0)
    max_xy = np.max(pts[:, :2], axis=0)
    span = np.maximum(max_xy - min_xy, 1e-4)

    for i, (x, y) in enumerate(pts[:, :2]):
        gx = int(((x - min_xy[0]) / span[0]) * (width - 1))
        gy = int(((y - min_xy[1]) / span[1]) * (height - 1))
        gx = max(0, min(width - 1, gx))
        gy = max(0, min(height - 1, gy))

        if i == 0:
            grid[gy][gx] = "S"  # Start
        elif i == len(pts) - 1:
            grid[gy][gx] = "E"  # End
        elif grid[gy][gx] == " ":
            grid[gy][gx] = "•"

    lines = ["+" + "-" * width + "+"]
    for row in grid:
        lines.append("|" + "".join(row) + "|")
    lines.append("+" + "-" * width + "+")
    return "\n".join(lines)


def main():
    print("=" * 70)
    print("  CONTACTLESS AIR-WRITING RECOGNITION FOR PARKINSON'S DISEASE  ")
    print("  Intention-Aware Motor Compensation & Assistive Communication  ")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = os.path.join(BASE_DIR, "saved_models")

    bilstm = Conv1DBiLSTMAttention(in_channels=3, num_classes=NUM_CLASSES)
    b_path = os.path.join(model_dir, "conv1d_bilstm.pt")
    if os.path.exists(b_path):
        ckpt = torch.load(b_path, map_location=device)
        bilstm.load_state_dict(ckpt["model_state_dict"])

    transformer = TrajectoryTransformer(in_channels=3, num_classes=NUM_CLASSES)
    t_path = os.path.join(model_dir, "transformer.pt")
    if os.path.exists(t_path):
        ckpt = torch.load(t_path, map_location=device)
        transformer.load_state_dict(ckpt["model_state_dict"])

    rf_clf = BaselineTrajectoryClassifier(model_type="rf")
    rf_path = os.path.join(model_dir, "baseline_rf.pkl")
    if os.path.exists(rf_path):
        rf_clf.load(rf_path)
    else:
        rf_clf = None

    recognizer = AirWritingRecognizer(
        bilstm_model=bilstm,
        transformer_model=transformer,
        baseline_model=rf_clf,
        device=device
    )

    generator = AirWritingDatasetGenerator(num_points=64, rng_seed=42)
    decoder = AssistiveNLPDecoder()

    demo_words = ["HELP", "WATER", "MEDS", "PAIN"]
    print(f"\nDemonstrating Intention Recognition on Assistive Phrases with Severe Tremor:\n")

    for word in demo_words:
        print(f"\n--- Intended Assistive Word: '{word}' ---")
        char_predictions = []
        posteriors = []

        for char in word:
            traj, _, meta = generator.generate_sample(char, severity=PDSeverity.SEVERE)
            t0 = time.perf_counter()
            result = recognizer.predict(traj, top_k=5)
            latency = (time.perf_counter() - t0) * 1000.0

            pred_char = result["predicted_char"]
            conf = result["confidence"]
            char_predictions.append(pred_char)
            posteriors.append([(c["char"], c["confidence"]) for c in result["candidates"]])

            print(f"Stroke '{char}' [Tremor Freq: {meta.get('tremor_freq_hz', 0.0):.1f} Hz] -> Predicted: '{pred_char}' ({conf*100:.1f}% conf, {latency:.2f} ms)")

        raw_word = "".join(char_predictions)
        decoded_hypotheses = decoder.beam_search_decode(posteriors, beam_width=5)
        nlp_word = decoded_hypotheses[0][0] if decoded_hypotheses else raw_word

        print(f"  Raw Recognized Word: '{raw_word}'")
        print(f"  NLP Decoded Output:   '{nlp_word}' (Beam Candidates: {[w for w, _ in decoded_hypotheses[:3]]})")

    # Sample Character Visualization
    print("\n--- Trajectory Visualization for Character 'A' with Severe Tremor ---")
    traj_a, _, meta_a = generator.generate_sample("A", severity=PDSeverity.SEVERE)
    print(render_ascii_trajectory(traj_a))
    res_a = recognizer.predict(traj_a, top_k=5)
    print(f"Predicted Character: '{res_a['predicted_char']}' with Confidence {res_a['confidence']*100:.1f}%")
    cand_strs = [c['char'] + ": " + f"{c['confidence']*100:.1f}%" for c in res_a['candidates']]
    print(f"Top-5 Candidates: {cand_strs}")
    k = res_a["kinematics"]
    print(f"Tremor 4-7Hz Power Ratio: {k.get('tremor_power_ratio_4_7hz', 0.0):.2f} | Jerk: {k.get('jerk_normalized', 0.0):.1f} | Mean Speed: {k.get('vel_mean_speed', 0.0):.2f}")

    print("\n" + "=" * 70)
    print("  Demo Complete! Run 'python app/web_app.py --port 5050' for the full Web UI.")
    print("=" * 70)


if __name__ == "__main__":
    main()
