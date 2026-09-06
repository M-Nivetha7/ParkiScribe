"""
Interactive Assistive Web Application & REST API for Air-Writing in Parkinson's Disease.
Provides real-time trajectory recognition, tremor simulation, kinematic bio-feedback,
assistive vocabulary decoding, and speech synthesis.
"""

import argparse
import base64
import os
import sys
import json
import cv2
import numpy as np
import torch

from flask import Flask, jsonify, render_template, request

# Add parent directory to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from data.character_templates import ALL_CLASSES, CLASS_TO_IDX, IDX_TO_CLASS, NUM_CLASSES
from data.parkinson_simulator import PDSeverity, ParkinsonSimulator
from data.dataset_generator import AirWritingDatasetGenerator
from models.conv1d_bilstm import Conv1DBiLSTMAttention
from models.transformer_model import TrajectoryTransformer
from models.baseline_classifier import BaselineTrajectoryClassifier
from models.ensemble_recognizer import AirWritingRecognizer
from nlp.word_recognizer import AssistiveNLPDecoder, ASSISTIVE_VOCABULARY
from src.hand_tracker import (
    HandTracker as LiveHandTracker,
    fingers_up as live_fingers_up,
    MEDIAPIPE_AVAILABLE,
)
from utils.constants import LANDMARK_INDEX_TIP
from src.motor_analysis import ParkinsonMotorAnalyzer
from src.recognizer import AirWritingRecognizer as AirWritingCNNRecognizer

app = Flask(__name__, template_folder="templates", static_folder="static")

# Live hand tracker for webcam frames
live_tracker = LiveHandTracker()

# Initialize global components
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bilstm_model = Conv1DBiLSTMAttention(in_channels=3, num_classes=NUM_CLASSES)
transformer_model = TrajectoryTransformer(in_channels=3, num_classes=NUM_CLASSES)
baseline_rf = BaselineTrajectoryClassifier(model_type="rf")

# Load model weights if saved
model_dir = os.path.join(BASE_DIR, "saved_models")
bilstm_path = os.path.join(model_dir, "conv1d_bilstm.pt")
if os.path.exists(bilstm_path):
    ckpt = torch.load(bilstm_path, map_location=device)
    bilstm_model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded {bilstm_path}")

trans_path = os.path.join(model_dir, "transformer.pt")
if os.path.exists(trans_path):
    ckpt = torch.load(trans_path, map_location=device)
    transformer_model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded {trans_path}")

rf_path = os.path.join(model_dir, "baseline_rf.pkl")
if os.path.exists(rf_path):
    baseline_rf.load(rf_path)
    print(f"Loaded {rf_path}")
else:
    baseline_rf = None

recognizer = AirWritingRecognizer(
    bilstm_model=bilstm_model,
    transformer_model=transformer_model,
    baseline_model=baseline_rf,
    device=device
)

nlp_decoder = AssistiveNLPDecoder()
simulator = ParkinsonSimulator(sampling_rate=30.0)
generator = AirWritingDatasetGenerator(num_points=64)
hand_tracker = LiveHandTracker()
motor_analyzer = ParkinsonMotorAnalyzer(sampling_rate=30.0)
cnn_recognizer = AirWritingCNNRecognizer(device=device)



@app.route("/")
def index():
    """Main Assistive UI Dashboard."""
    return render_template("index.html", classes=ALL_CLASSES, vocabulary=ASSISTIVE_VOCABULARY[:30])


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Predict intended character using 2D CNN from canvas image or rendered strokes.
    Accepts JSON: { "image": "data:image/png;base64,...", "trajectory": [...], "top_k": 5 }
    """
    data = request.get_json(force=True)
    top_k = int(data.get("top_k", 5))

    canvas_bgr = None
    image_b64 = data.get("image") or data.get("canvas_image")
    if image_b64:
        try:
            if "," in image_b64:
                image_b64 = image_b64.split(",")[1]
            img_bytes = base64.b64decode(image_b64)
            nparr = np.frombuffer(img_bytes, np.uint8)
            canvas_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception as e:
            canvas_bgr = None

    raw_traj = np.array(data.get("trajectory", []), dtype=np.float32)

    # If canvas not sent directly, render trajectory to virtual canvas
    if canvas_bgr is None and len(raw_traj) >= 2:
        canvas_bgr = np.zeros((480, 480, 3), dtype=np.uint8)
        min_x, max_x = float(np.min(raw_traj[:, 0])), float(np.max(raw_traj[:, 0]))
        min_y, max_y = float(np.min(raw_traj[:, 1])), float(np.max(raw_traj[:, 1]))
        w = max(max_x - min_x, 1e-4)
        h = max(max_y - min_y, 1e-4)
        scale = 360.0 / max(w, h)
        pts = (60 + np.column_stack([(raw_traj[:, 0] - min_x) * scale, (raw_traj[:, 1] - min_y) * scale])).astype(np.int32)
        for i in range(1, len(pts)):
            cv2.line(canvas_bgr, (pts[i-1][0], pts[i-1][1]), (pts[i][0], pts[i][1]), (255, 0, 0), 10)

    # 1. Run 2D CNN recognition (Module 9 & 10)
    cnn_pred, meta = cnn_recognizer.predict_canvas(canvas_bgr, top_k=top_k)

    if cnn_pred:
        pred_result = {
            "predicted_char": cnn_pred["predicted_char"],
            "predicted_word": cnn_pred.get("predicted_word", cnn_pred["predicted_char"]),
            "is_word": cnn_pred.get("is_word", False),
            "raw_characters": cnn_pred.get("raw_characters", cnn_pred["predicted_char"]),
            "characters": cnn_pred.get("characters", []),
            "confidence": float(cnn_pred["confidence"]) / 100.0,
            "candidates": [
                {"char": c["char"], "confidence": float(c["confidence"]) / 100.0}
                for c in cnn_pred["top_candidates"]
            ],
            "filtered_trajectory": raw_traj.tolist() if len(raw_traj) else []
        }

    else:
        # Fallback to trajectory ensemble
        if len(raw_traj) >= 2:
            if raw_traj.ndim == 2 and raw_traj.shape[1] == 2:
                z = np.zeros((len(raw_traj), 1), dtype=np.float32)
                raw_traj = np.hstack([raw_traj, z])
            filtered_traj = recognizer.preprocess_trajectory(raw_traj)
            pred_result = recognizer.predict(raw_traj, top_k=top_k)
            pred_result["filtered_trajectory"] = filtered_traj.tolist()
        else:
            return jsonify({"error": "No valid strokes or image provided"}), 400

    # Calculate Parkinson's motor biomarkers (Module 11)
    if len(raw_traj) >= 3:
        timestamps = np.linspace(0, len(raw_traj) / 30.0, len(raw_traj), dtype=np.float32)
        traj_with_time = np.column_stack([timestamps, raw_traj[:, 0], raw_traj[:, 1]])
        biomarkers = motor_analyzer.analyze_trajectory(traj_with_time)
        pred_result["biomarkers"] = biomarkers.to_dict()
    else:
        pred_result["biomarkers"] = motor_analyzer.analyze_trajectory(np.zeros((3, 3), dtype=np.float32)).to_dict()

    return jsonify(pred_result)


@app.route("/api/simulate", methods=["POST"])
def api_simulate():
    """
    Generate synthetic Parkinsonian trajectory for a selected character and severity.
    Expects JSON: { "char": "A", "severity": "moderate" }
    """
    data = request.get_json(force=True)
    char = str(data.get("char", "A")).upper()
    severity_str = str(data.get("severity", "moderate")).lower()

    try:
        severity = PDSeverity(severity_str)
    except ValueError:
        severity = PDSeverity.MODERATE

    if char not in ALL_CLASSES:
        return jsonify({"error": f"Character {char} not recognized"}), 400

    traj, label, meta = generator.generate_sample(char, severity=severity)
    filtered_traj = recognizer.preprocess_trajectory(traj)

    # Render simulated sample onto canvas and run CNN
    canvas_sim = np.zeros((400, 400, 3), dtype=np.uint8)
    min_x, max_x = float(np.min(traj[:, 0])), float(np.max(traj[:, 0]))
    min_y, max_y = float(np.min(traj[:, 1])), float(np.max(traj[:, 1]))
    scale = 320.0 / max(max_x - min_x, max_y - min_y, 1e-4)
    pts = (40 + np.column_stack([(traj[:, 0] - min_x) * scale, (traj[:, 1] - min_y) * scale])).astype(np.int32)
    for i in range(1, len(pts)):
        cv2.line(canvas_sim, (pts[i-1][0], pts[i-1][1]), (pts[i][0], pts[i][1]), (255, 0, 0), 8)

    cnn_pred, _ = cnn_recognizer.predict_canvas(canvas_sim, top_k=5)
    if cnn_pred:
        pred_result = {
            "predicted_char": cnn_pred["predicted_char"],
            "confidence": float(cnn_pred["confidence"]) / 100.0,
            "candidates": [
                {"char": c["char"], "confidence": float(c["confidence"]) / 100.0}
                for c in cnn_pred["top_candidates"]
            ]
        }
    else:
        pred_result = recognizer.predict(traj, top_k=5)

    # Calculate Parkinson's motor biomarkers (Module 11)
    timestamps = np.linspace(0, len(traj) / 30.0, len(traj), dtype=np.float32)
    traj_with_time = np.column_stack([timestamps, traj[:, 0], traj[:, 1]])
    biomarkers = motor_analyzer.analyze_trajectory(traj_with_time)

    return jsonify({
        "char": char,
        "label": label,
        "metadata": meta,
        "raw_trajectory": traj.tolist(),
        "filtered_trajectory": filtered_traj.tolist(),
        "prediction": pred_result,
        "biomarkers": biomarkers.to_dict()
    })


@app.route("/api/decode_word", methods=["POST"])
def api_decode_word():
    """
    Decode word from character posteriors or query prefix.
    Expects JSON: { "query": "HEL", "posteriors": [...] }
    """
    data = request.get_json(force=True)
    query = str(data.get("query", "")).upper()
    posteriors = data.get("posteriors", None)

    suggestions = []
    if posteriors and len(posteriors) > 0:
        beam_results = nlp_decoder.beam_search_decode(posteriors, beam_width=5)
        suggestions = [{"word": w, "confidence": c} for w, c in beam_results]
    elif query:
        corrections = nlp_decoder.correct_word(query, max_distance=2)
        autocompletes = nlp_decoder.autocomplete_prefix(query, max_results=5)
        seen = set()
        for w, score in corrections:
            if w not in seen:
                suggestions.append({"word": w, "confidence": round(score, 2)})
                seen.add(w)
        for w in autocompletes:
            if w not in seen:
                suggestions.append({"word": w, "confidence": 0.85})
                seen.add(w)

    return jsonify({
        "query": query,
        "suggestions": suggestions[:5]
    })


@app.route("/api/process_frame", methods=["POST"])
def api_process_frame():
    """
    Process single base64 webcam frame for touchless hand tracking.
    Uses index fingertip tracking and gesture recognition matching user's starter code:
      - Only Index up: 'draw'
      - Index + Middle up: 'pen_lift'
      - All down (fist): 'fist'
    """
    data = request.get_json(force=True)
    image_b64 = data.get("image", "")

    if not image_b64:
        return jsonify({"error": "No image provided"}), 400

    if not MEDIAPIPE_AVAILABLE:
        return jsonify({
            "error": "MediaPipe is not installed. Install it with: python3.11 -m pip install mediapipe"
        }), 503

    try:
        if "," in image_b64:
            image_b64 = image_b64.split(",")[1]
        img_bytes = base64.b64decode(image_b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"detected": False, "point": None, "state": "none"}), 200

        live_tracker.find_hands(frame)
        lm_list = live_tracker.get_landmarks(frame)

        if len(lm_list) != 0:
            x1 = lm_list[LANDMARK_INDEX_TIP][1]
            y1 = lm_list[LANDMARK_INDEX_TIP][2]
            h, w = frame.shape[:2]
            norm_x = float(x1) / w
            norm_y = float(y1) / h

            fingers = live_fingers_up(lm_list)

            # Match the standalone air-writing implementation exactly:
            # fist clears, index + middle lifts the pen, index alone draws.
            if fingers == [False, False, False, False]:
                state = "fist"
            elif fingers[0] and fingers[1]:
                state = "pen_lift"
            elif fingers[0] and not fingers[1]:
                state = "draw"
            else:
                state = "idle"

            return jsonify({
                "detected": True,
                "point": [norm_x, norm_y],
                "pixel": [int(x1), int(y1)],
                "state": state,
                "fingers": fingers
            })
        else:
            return jsonify({
                "detected": False,
                "point": None,
                "pixel": None,
                "state": "none",
                "fingers": []
            })

    except Exception as e:
        return jsonify({"error": str(e)}), 500



def run_server():
    parser = argparse.ArgumentParser(description="Start Air-Writing Web Application")
    parser.add_argument("--port", type=int, default=5050)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    print(f"Starting Assistive Air-Writing Server at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    run_server()
