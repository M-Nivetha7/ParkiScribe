"""
Enhanced CNN Training Script for Air Writing Recognition (Modules 9 & 10).
Trains AirWritingCNN using a rich multimodal dataset combining:
1. OpenCV vector stroke typography (Hershey fonts: Simplex, Duplex, Complex, Triplex)
   with random line thickness, scale, aspect-ratio scaling, shear, and rotation.
2. Canonical continuous trajectories from character_templates.py.
3. Parkinsonian tremor augmentations (4-7 Hz sinusoidal ripples, elastic warping,
   Gaussian jitter, stroke breaks, and micrographia).
"""

import os
import sys
import argparse
from typing import Tuple, List, Optional
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from utils.constants import ALL_CLASSES, CLASS_TO_IDX, NUM_CLASSES, MODEL_INPUT_SIZE
from data.character_templates import get_canonical_trajectory
from data.parkinson_simulator import ParkinsonSimulator, PDSeverity
from src.model import AirWritingCNN, augment_tremor_image
from src.preprocessing import preprocess_canvas


def render_font_sample(
    char: str,
    font_face: int,
    thickness: int = 6,
    scale: float = 3.5,
    rotation_deg: float = 0.0,
    shear: float = 0.0,
    aspect_ratio_scale: float = 1.0,
    canvas_size: int = 240
) -> np.ndarray:
    """Render character using OpenCV Hershey vector fonts with affine distortions."""
    canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)

    # Get text size to center
    (tw, th), baseline = cv2.getTextSize(char, font_face, scale, thickness)
    tx = max(10, (canvas_size - tw) // 2)
    ty = max(th + 10, (canvas_size + th) // 2)

    cv2.putText(canvas, char, (tx, ty), font_face, scale, (255, 0, 0), thickness, cv2.LINE_AA)

    # Apply aspect ratio stretching / compressing
    if abs(aspect_ratio_scale - 1.0) > 0.05:
        M_scale = np.array([
            [aspect_ratio_scale, 0, (1.0 - aspect_ratio_scale) * (canvas_size / 2.0)],
            [0, 1.0, 0]
        ], dtype=np.float32)
        canvas = cv2.warpAffine(canvas, M_scale, (canvas_size, canvas_size))

    # Apply rotation and shear
    if abs(rotation_deg) > 0.5 or abs(shear) > 0.01:
        M = cv2.getRotationMatrix2D((canvas_size // 2, canvas_size // 2), rotation_deg, 1.0)
        M[0, 1] += shear  # horizontal shear
        canvas = cv2.warpAffine(canvas, M, (canvas_size, canvas_size))

    return canvas


def render_trajectory_sample(
    trajectory: np.ndarray,
    canvas_size: int = 240,
    brush_thickness: int = 6
) -> np.ndarray:
    """Render 2D/3D trajectory onto canvas with line interpolation."""
    canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
    margin = 30
    x, y = trajectory[:, 0], trajectory[:, 1]
    w = max(np.max(x) - np.min(x), 1e-4)
    h = max(np.max(y) - np.min(y), 1e-4)
    scale = (canvas_size - 2 * margin) / max(w, h)

    px = margin + ((x - np.min(x)) * scale).astype(np.int32)
    py = margin + ((y - np.min(y)) * scale).astype(np.int32)

    for i in range(1, len(px)):
        cv2.line(canvas, (px[i - 1], py[i - 1]), (px[i], py[i]), (255, 0, 0), brush_thickness)

    return canvas


def generate_comprehensive_dataset(samples_per_class: int = 140) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates balanced, diverse training set for all 36 classes (0-9, A-Z):
    - Multi-font stroke renderings
    - Multi-severity Parkinsonian tremors
    - Variable brush widths, rotations, and anisotropic aspect-ratio scaling
    - Both uppercase and lowercase handwritten variants
    """
    print(f"Generating comprehensive dataset: {samples_per_class} samples/class x {NUM_CLASSES} classes = {samples_per_class * NUM_CLASSES} samples...")

    fonts = [
        cv2.FONT_HERSHEY_SIMPLEX,
        cv2.FONT_HERSHEY_DUPLEX,
        cv2.FONT_HERSHEY_COMPLEX,
        cv2.FONT_HERSHEY_TRIPLEX,
        cv2.FONT_HERSHEY_PLAIN
    ]

    simulator = ParkinsonSimulator(sampling_rate=30.0, rng_seed=42)
    severities = [PDSeverity.CONTROL, PDSeverity.MILD, PDSeverity.MODERATE, PDSeverity.SEVERE]

    images = []
    labels = []
    rng = np.random.default_rng(42)

    for char_idx, char in enumerate(ALL_CLASSES):
        base_traj = get_canonical_trajectory(char, num_points=64)

        for s in range(samples_per_class):
            sev = severities[s % len(severities)]
            source_type = s % 3  # 0: font, 1: trajectory, 2: augmented font

            # 35% chance to render lowercase for letter classes so both forms are recognized
            draw_char = char.lower() if (char.isalpha() and rng.random() < 0.35) else char
            aspect_scale = float(rng.uniform(0.50, 1.65))

            if source_type == 0:
                # Clean or mildly jittered font
                f = fonts[rng.integers(0, len(fonts))]
                thick = int(rng.integers(3, 13))
                scale = float(rng.uniform(2.8, 4.2)) if f != cv2.FONT_HERSHEY_PLAIN else float(rng.uniform(4.0, 7.0))
                rot = float(rng.uniform(-16.0, 16.0))
                shear = float(rng.uniform(-0.15, 0.15))
                canvas = render_font_sample(draw_char, f, thickness=thick, scale=scale, rotation_deg=rot, shear=shear, aspect_ratio_scale=aspect_scale)

            elif source_type == 1:
                # Continuous trajectory with Parkinson simulator
                deg_traj, _ = simulator.apply_parkinson_effects(base_traj, severity=sev)
                thick = int(rng.integers(4, 11))
                canvas = render_trajectory_sample(deg_traj, brush_thickness=thick)

            else:
                # Font with simulated hand tremor
                f = fonts[rng.integers(0, len(fonts))]
                thick = int(rng.integers(4, 12))
                scale = float(rng.uniform(3.0, 4.0)) if f != cv2.FONT_HERSHEY_PLAIN else float(rng.uniform(4.5, 6.5))
                rot = float(rng.uniform(-14.0, 14.0))
                canvas = render_font_sample(draw_char, f, thickness=thick, scale=scale, rotation_deg=rot, aspect_ratio_scale=aspect_scale)


            # Preprocess canvas to 64x64 normalized float tensor
            norm_img, meta = preprocess_canvas(canvas, target_size=MODEL_INPUT_SIZE)
            if not meta.get("has_content", False):
                norm_img = cv2.resize(canvas[:, :, 0], MODEL_INPUT_SIZE).astype(np.float32) / 255.0

            # Apply tremor augmentation for non-control
            if sev != PDSeverity.CONTROL:
                intensity = 0.3 if sev == PDSeverity.MILD else (0.6 if sev == PDSeverity.MODERATE else 0.85)
                norm_img = augment_tremor_image(norm_img, tremor_intensity=intensity, rng=rng)

            images.append(norm_img)
            labels.append(char_idx)

    X = np.array(images, dtype=np.float32)[:, np.newaxis, :, :]  # (N, 1, 64, 64)
    y = np.array(labels, dtype=np.int64)

    # Random shuffle
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


def train_enhanced_cnn(
    epochs: int = 18,
    batch_size: int = 64,
    lr: float = 1e-3,
    save_path: str = "saved_models/airwriting_cnn.pt"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Enhanced AirWritingCNN on device: {device}")

    X, y = generate_comprehensive_dataset(samples_per_class=120)
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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

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
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--save_path", type=str, default="saved_models/airwriting_cnn.pt")
    args = parser.parse_args()

    train_enhanced_cnn(epochs=args.epochs, save_path=os.path.join(BASE_DIR, args.save_path))
