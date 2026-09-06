"""
Unified Training & Evaluation Pipeline for Air-Writing Recognition in Parkinson's Disease.
Supports training:
- Multi-Scale Conv1D + BiLSTM + Attention
- Trajectory Transformer
- Random Forest / SVM baselines
- Saving checkpoints and evaluation metrics
"""

import argparse
import json
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from data.character_templates import NUM_CLASSES, ALL_CLASSES
from data.dataset_generator import AirWritingDatasetGenerator
from data.dataset_loader import create_dataloaders
from models.conv1d_bilstm import Conv1DBiLSTMAttention
from models.transformer_model import TrajectoryTransformer
from models.baseline_classifier import BaselineTrajectoryClassifier
from eval.metrics import EvaluationMetrics


def train_deep_model(
    model: nn.Module,
    train_loader,
    val_loader,
    test_loader,
    split_info: dict,
    epochs: int = 25,
    lr: float = 1e-3,
    device: torch.device = torch.device("cpu"),
    save_path: str = "saved_models/model.pt"
) -> dict:
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    best_val_acc = 0.0
    best_weights = None
    train_history = []

    print(f"--- Training {model.__class__.__name__} for {epochs} epochs on {device} ---")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits, _ = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            total_loss += loss.item() * len(y_batch)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y_batch).sum().item()
            total += len(y_batch)

        scheduler.step()
        train_loss = total_loss / total
        train_acc = correct / total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for x_v, y_v in val_loader:
                x_v = x_v.to(device)
                y_v = y_v.to(device)
                l_v, _ = model(x_v)
                v_loss = criterion(l_v, y_v)
                val_loss += v_loss.item() * len(y_v)
                p_v = torch.argmax(l_v, dim=1)
                val_correct += (p_v == y_v).sum().item()
                val_total += len(y_v)

        val_acc = val_correct / val_total
        val_loss /= val_total

        train_history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f}, Acc: {train_acc*100:.1f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.1f}% (Best: {best_val_acc*100:.1f}%)")

    # Load best weights
    if best_weights is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})

    # Test Evaluation
    model.eval()
    all_probs = []
    all_targets = []
    with torch.no_grad():
        for x_t, y_t in test_loader:
            x_t = x_t.to(device)
            l_t, _ = model(x_t)
            probs = F.softmax(l_t, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(y_t.numpy())

    all_probs = np.vstack(all_probs)
    all_targets = np.concatenate(all_targets)

    test_metrics = EvaluationMetrics.compute_all_metrics(
        all_targets, all_probs, severities=split_info.get("test_severities")
    )
    print(f"\n--- Test Results for {model.__class__.__name__} ---")
    print(f"Accuracy (Top-1): {test_metrics['top1_accuracy']*100:.2f}%")
    print(f"Top-3 Accuracy:   {test_metrics['top3_accuracy']*100:.2f}%")
    print(f"Top-5 Accuracy:   {test_metrics['top5_accuracy']*100:.2f}%")
    print(f"Macro F1-Score:   {test_metrics['macro_f1']:.4f}")

    if "severity_breakdown" in test_metrics:
        print("\nPerformance Breakdown by Parkinsonian Impairment Severity:")
        for sev, s_metrics in test_metrics["severity_breakdown"].items():
            print(f"  [{sev.upper():<8}] Top-1: {s_metrics['top1_accuracy']*100:.1f}% | Top-3: {s_metrics['top3_accuracy']*100:.1f}% (N={s_metrics['count']})")

    # Save model
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_class": model.__class__.__name__,
        "metrics": test_metrics,
        "history": train_history
    }, save_path)
    print(f"Saved checkpoint to {save_path}\n")

    return test_metrics


def main():
    parser = argparse.ArgumentParser(description="Train Air-Writing Recognition Models for Parkinson's Disease")
    parser.add_argument("--model", type=str, default="all", choices=["conv1d_bilstm", "transformer", "baseline_rf", "baseline_svm", "all"])
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--samples_per_class", type=int, default=45)
    parser.add_argument("--save_dir", type=str, default="saved_models")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Generating synthetic multi-severity Parkinsonian air-writing dataset ({args.samples_per_class} samples/class)...")
    generator = AirWritingDatasetGenerator(num_points=64, rng_seed=args.seed)
    dataset = generator.generate_dataset(samples_per_class=args.samples_per_class)
    print(f"Total dataset samples: {len(dataset['labels'])} across {NUM_CLASSES} classes.")

    train_loader, val_loader, test_loader, split_info = create_dataloaders(
        dataset, batch_size=args.batch_size, random_state=args.seed
    )
    print(f"Train: {split_info['num_train']}, Val: {split_info['num_val']}, Test: {split_info['num_test']}")

    results = {}

    # 1. Conv1D-BiLSTM
    if args.model in ["conv1d_bilstm", "all"]:
        bilstm = Conv1DBiLSTMAttention(in_channels=3, num_classes=NUM_CLASSES)
        m_path = os.path.join(args.save_dir, "conv1d_bilstm.pt")
        res = train_deep_model(bilstm, train_loader, val_loader, test_loader, split_info, epochs=args.epochs, lr=args.lr, device=device, save_path=m_path)
        results["conv1d_bilstm"] = res

    # 2. Transformer
    if args.model in ["transformer", "all"]:
        transformer = TrajectoryTransformer(in_channels=3, num_classes=NUM_CLASSES)
        t_path = os.path.join(args.save_dir, "transformer.pt")
        res = train_deep_model(transformer, train_loader, val_loader, test_loader, split_info, epochs=args.epochs, lr=args.lr, device=device, save_path=t_path)
        results["transformer"] = res

    # 3. Random Forest Baseline
    if args.model in ["baseline_rf", "all"]:
        print("--- Training Random Forest Baseline on Kinematic/Spectral Features ---")
        rf_clf = BaselineTrajectoryClassifier(model_type="rf", random_state=args.seed)
        # Reconstruct train & test arrays
        X_train_full = []
        y_train_full = []
        for x_b, y_b in train_loader:
            X_train_full.append(x_b.transpose(1, 2).numpy())
            y_train_full.append(y_b.numpy())
        X_train_full = np.concatenate(X_train_full)
        y_train_full = np.concatenate(y_train_full)

        rf_clf.fit(X_train_full, y_train_full)
        rf_path = os.path.join(args.save_dir, "baseline_rf.pkl")
        rf_clf.save(rf_path)

        # Test evaluation
        X_test = split_info["test_trajectories"]
        y_test = split_info["test_labels"]
        rf_probs = rf_clf.predict_proba(X_test)
        rf_metrics = EvaluationMetrics.compute_all_metrics(y_test, rf_probs, severities=split_info.get("test_severities"))
        print(f"Random Forest Top-1 Accuracy: {rf_metrics['top1_accuracy']*100:.2f}% | Top-3: {rf_metrics['top3_accuracy']*100:.2f}%\n")
        results["baseline_rf"] = rf_metrics

    # Save summary report
    summary_path = os.path.join(args.save_dir, "training_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved complete training summary to {summary_path}")


if __name__ == "__main__":
    main()
