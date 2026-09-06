"""
Training Script for 2D AirWritingCNN using Tremor-Augmented Synthetic Data (Modules 9 & 10).
Renders canonical and Parkinsonian trajectories onto virtual canvases,
applies image preprocessing, and trains a robust 2D CNN classifier for 36 classes.
"""

import os
import sys
import argparse
from typing import Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import cv2

# Project root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from utils.constants import ALL_CLASSES, CLASS_TO_IDX, NUM_CLASSES, MODEL_INPUT_SIZE
from data.character_templates import get_canonical_trajectory
from data.parkinson_simulator import ParkinsonSimulator, PDSeverity
from src.model import AirWritingCNN, augment_tremor_image
from src.preprocessing import preprocess_canvas


def render_trajectory_to_canvas(
    trajectory: np.ndarray,
    canvas_size: int = 240,
    brush_thickness: int = 6
) -> np.ndarray:
    """Render normalized (N, 3) trajectory onto a black BGR canvas."""
    canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)

    # Normalize to [margin, canvas_size - margin]
    margin = 25
    x = trajectory[:, 0]
    y = trajectory[:, 1]

    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)

    w = max(x_max - x_min, 1e-4)
    h = max(y_max - y_min, 1e-4)
    scale = (canvas_size - 2 * margin) / max(w, h)

    px = margin + ((x - x_min) * scale).astype(np.int32)
    py = margin + ((y - y_min) * scale).astype(np.int32)

    for i in range(1, len(px)):
        cv2.line(canvas, (px[i - 1], py[i - 1]), (px[i], py[i]), (255, 0, 0), brush_thickness)

    return canvas


def generate_cnn_dataset(samples_per_class: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate dataset of 2D preprocessed images across 36 classes
    spanning Control, Mild, Moderate, and Severe Parkinsonian tremors.
    """
    simulator = ParkinsonSimulator(sampling_rate=30.0, rng_seed=123)
    severities = [PDSeverity.CONTROL, PDSeverity.MILD, PDSeverity.MODERATE, PDSeverity.SEVERE]

    images = []
    labels = []

    print(f"Generating synthetic training dataset: {samples_per_class} samples/class x {NUM_CLASSES} classes...")

    for char_idx, char in enumerate(ALL_CLASSES):
        base_traj = get_canonical_trajectory(char, num_points=64)

        for s in range(samples_per_class):
            sev = severities[s % len(severities)]
            deg_traj, _ = simulator.apply_parkinson_effects(base_traj, severity=sev)

            canvas = render_trajectory_to_canvas(deg_traj)
            norm_img, meta = preprocess_canvas(canvas, target_size=MODEL_INPUT_SIZE)

            if not meta.get("has_content", False):
                norm_img = cv2.resize(canvas[:, :, 0], MODEL_INPUT_SIZE).astype(np.float32) / 255.0

            # Apply tremor image augmentation (Module 10)
            if sev != PDSeverity.CONTROL:
                norm_img = augment_tremor_image(norm_img, tremor_intensity=0.4)

            images.append(norm_img)
            labels.append(char_idx)

    X = np.array(images, dtype=np.float32)[:, np.newaxis, :, :]  # (N, 1, H, W)
    y = np.array(labels, dtype=np.int64)

    # Shuffle
    perm = np.random.permutation(len(y))
    return X[perm], y[perm]


def train_cnn(epochs: int = 15, batch_size: int = 32, lr: float = 1e-3, save_path: str = "saved_models/airwriting_cnn.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training AirWritingCNN on device: {device}")

    X, y = generate_cnn_dataset(samples_per_class=35)
    total_samples = len(y)
    split = int(0.85 * total_samples)

    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = AirWritingCNN(num_classes=NUM_CLASSES, input_channels=1).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, correct, total = 0.0, 0, 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

        scheduler.step()
        train_acc = correct / total

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == targets).sum().item()
                val_total += targets.size(0)

        val_acc = val_correct / val_total
        print(f"Epoch {epoch:2d}/{epochs} | Train Loss: {train_loss/total:.4f} Acc: {train_acc*100:.1f}% | Val Loss: {val_loss/val_total:.4f} Acc: {val_acc*100:.1f}%")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            torch.save({"model_state_dict": model.state_dict(), "val_acc": val_acc}, save_path)

    print(f"\nTraining Complete! Best Val Accuracy: {best_val_acc*100:.2f}%. Saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--save_path", type=str, default="saved_models/airwriting_cnn.pt")
    args = parser.parse_args()

    train_cnn(epochs=args.epochs, save_path=os.path.join(BASE_DIR, args.save_path))
