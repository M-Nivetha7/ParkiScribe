"""
Comprehensive Benchmarking & Robustness Evaluation Suite.
Evaluates recognition accuracy across Parkinsonian severity levels,
computes word-level assistive decoding CER/WER, and benchmarks inference latency.
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List
import numpy as np
import torch

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
from eval.metrics import EvaluationMetrics


def run_latency_benchmark(recognizer: AirWritingRecognizer, num_iterations: int = 100) -> Dict[str, float]:
    """Measure inference latency per stroke in milliseconds."""
    sample_traj = np.random.randn(64, 3).astype(np.float32)

    # Warmup
    for _ in range(10):
        recognizer.predict(sample_traj)

    latencies_ms = []
    for _ in range(num_iterations):
        t0 = time.perf_counter()
        recognizer.predict(sample_traj)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    return {
        "mean_latency_ms": float(np.mean(latencies_ms)),
        "median_latency_ms": float(np.median(latencies_ms)),
        "p95_latency_ms": float(np.percentile(latencies_ms, 95)),
        "p99_latency_ms": float(np.percentile(latencies_ms, 99)),
        "fps_throughput": float(1000.0 / np.mean(latencies_ms))
    }


def run_severity_robustness_experiment(
    recognizer: AirWritingRecognizer,
    samples_per_class: int = 25,
    seed: int = 42
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate recognizer performance across Control, Mild, Moderate, and Severe PD.
    """
    simulator = ParkinsonSimulator(rng_seed=seed)
    generator = AirWritingDatasetGenerator(num_points=64, rng_seed=seed)
    results = {}

    for sev in [PDSeverity.CONTROL, PDSeverity.MILD, PDSeverity.MODERATE, PDSeverity.SEVERE]:
        y_true = []
        all_probs = []

        for char in ALL_CLASSES:
            for _ in range(samples_per_class):
                traj, lbl, _ = generator.generate_sample(char, severity=sev)
                probs, _, _ = recognizer.predict_probabilities(traj)
                y_true.append(lbl)
                all_probs.append(probs)

        y_true = np.array(y_true, dtype=np.int64)
        all_probs = np.array(all_probs, dtype=np.float32)

        metrics = EvaluationMetrics.compute_all_metrics(y_true, all_probs)
        results[sev.value] = {
            "top1_accuracy": metrics["top1_accuracy"],
            "top3_accuracy": metrics["top3_accuracy"],
            "top5_accuracy": metrics["top5_accuracy"],
            "macro_f1": metrics["macro_f1"]
        }

    return results


def run_word_decoding_experiment(
    recognizer: AirWritingRecognizer,
    test_words: List[str] = None,
    severity: PDSeverity = PDSeverity.MODERATE,
    seed: int = 42
) -> Dict[str, float]:
    """
    Evaluate word-level communication accuracy with NLP beam search decoder.
    """
    if test_words is None:
        test_words = ["HELP", "WATER", "PAIN", "NURSE", "FOOD", "MEDICINE", "YES", "NO", "DOCTOR", "REST"]

    decoder = AssistiveNLPDecoder()
    generator = AirWritingDatasetGenerator(num_points=64, rng_seed=seed)

    cer_raw_list = []
    cer_nlp_list = []
    wer_raw_list = []
    wer_nlp_list = []

    for word in test_words:
        char_posteriors = []
        raw_predicted_chars = []

        for char in word:
            traj, _, _ = generator.generate_sample(char, severity=severity)
            probs, _, _ = recognizer.predict_probabilities(traj)
            top_k_indices = np.argsort(probs)[::-1][:5]
            char_posteriors.append([(IDX_TO_CLASS[i], float(probs[i])) for i in top_k_indices])
            raw_predicted_chars.append(IDX_TO_CLASS[top_k_indices[0]])

        raw_word = "".join(raw_predicted_chars)
        decoded_hypotheses = decoder.beam_search_decode(char_posteriors, beam_width=5)
        nlp_word = decoded_hypotheses[0][0] if decoded_hypotheses else raw_word

        cer_raw_list.append(EvaluationMetrics.character_error_rate(word, raw_word))
        cer_nlp_list.append(EvaluationMetrics.character_error_rate(word, nlp_word))
        wer_raw_list.append(1.0 if raw_word != word else 0.0)
        wer_nlp_list.append(1.0 if nlp_word != word else 0.0)

    return {
        "raw_cer": float(np.mean(cer_raw_list)),
        "nlp_cer": float(np.mean(cer_nlp_list)),
        "raw_wer": float(np.mean(wer_raw_list)),
        "nlp_wer": float(np.mean(wer_nlp_list)),
        "cer_reduction_pct": float(max(0, (np.mean(cer_raw_list) - np.mean(cer_nlp_list)) / (np.mean(cer_raw_list) + 1e-6) * 100)),
        "wer_reduction_pct": float(max(0, (np.mean(wer_raw_list) - np.mean(wer_nlp_list)) / (np.mean(wer_raw_list) + 1e-6) * 100))
    }


def main():
    parser = argparse.ArgumentParser(description="Air-Writing System Benchmark")
    parser.add_argument("--model_dir", type=str, default="saved_models")
    parser.add_argument("--output_report", type=str, default="eval/benchmark_report.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running comprehensive system benchmarks on {device}...")

    # Load models
    bilstm = Conv1DBiLSTMAttention(in_channels=3, num_classes=NUM_CLASSES)
    bilstm_path = os.path.join(args.model_dir, "conv1d_bilstm.pt")
    if os.path.exists(bilstm_path):
        ckpt = torch.load(bilstm_path, map_location=device)
        bilstm.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded {bilstm_path}")

    transformer = TrajectoryTransformer(in_channels=3, num_classes=NUM_CLASSES)
    t_path = os.path.join(args.model_dir, "transformer.pt")
    if os.path.exists(t_path):
        ckpt = torch.load(t_path, map_location=device)
        transformer.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded {t_path}")

    rf_clf = BaselineTrajectoryClassifier(model_type="rf")
    rf_path = os.path.join(args.model_dir, "baseline_rf.pkl")
    if os.path.exists(rf_path):
        rf_clf.load(rf_path)
        print(f"Loaded {rf_path}")
    else:
        rf_clf = None

    recognizer = AirWritingRecognizer(
        bilstm_model=bilstm,
        transformer_model=transformer,
        baseline_model=rf_clf,
        device=device
    )

    print("\n1. Measuring Inference Latency & FPS Throughput...")
    latency_results = run_latency_benchmark(recognizer, num_iterations=100)
    print(f"  Mean Latency: {latency_results['mean_latency_ms']:.2f} ms")
    print(f"  P95 Latency:  {latency_results['p95_latency_ms']:.2f} ms")
    print(f"  Throughput:   {latency_results['fps_throughput']:.1f} FPS")

    print("\n2. Evaluating Motor Severity Robustness (Control vs Mild vs Moderate vs Severe PD)...")
    severity_results = run_severity_robustness_experiment(recognizer, samples_per_class=20)
    for sev, res in severity_results.items():
        print(f"  [{sev.upper():<8}] Top-1 Acc: {res['top1_accuracy']*100:.2f}% | Top-3 Acc: {res['top3_accuracy']*100:.2f}% | Macro F1: {res['macro_f1']:.4f}")

    print("\n3. Evaluating Word-Level NLP Beam Search & Error Recovery...")
    word_results = run_word_decoding_experiment(recognizer, severity=PDSeverity.MODERATE)
    print(f"  Raw Stroke CER: {word_results['raw_cer']*100:.2f}% -> NLP Decoded CER: {word_results['nlp_cer']*100:.2f}%")
    print(f"  Raw Word WER:   {word_results['raw_wer']*100:.2f}% -> NLP Decoded WER: {word_results['nlp_wer']*100:.2f}% (WER Reduction: {word_results['wer_reduction_pct']:.1f}%)")

    report = {
        "latency": latency_results,
        "severity_robustness": severity_results,
        "word_decoding": word_results
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output_report)), exist_ok=True)
    with open(args.output_report, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved benchmark report to {args.output_report}")


if __name__ == "__main__":
    main()
