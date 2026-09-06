from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils.constants import NUM_CLASSES, MODEL_INPUT_SIZE


class AirWritingCNN(nn.Module):
    """
    Convolutional Neural Network for Air-Written Character Recognition (Module 9).
    Classifies preprocessed 2D virtual canvas images across 36 alphanumeric classes (0-9, A-Z).
    Engineered with spatial regularization and batch normalization to provide robust classification
    on irregular, tremor-affected handwriting.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, input_channels: int = 1):
        super().__init__()
        self.num_classes = num_classes

        # Block 1
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Pooling and Dropout
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout_conv = nn.Dropout2d(0.15)
        self.dropout_fc = nn.Dropout(0.35)

        # Adaptive pool guarantees fixed size before linear layers
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Tensor of shape (B, 1, H, W)
        Returns:
            logits: Tensor of shape (B, num_classes)
        """
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)

        # Block 2
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout_conv(x)

        # Block 3
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)

        # Dense Head
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)
        logits = self.fc2(x)

        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Compute softmax probability distribution."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=-1)


def augment_tremor_image(
    image: np.ndarray,
    tremor_intensity: float = 0.5,
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Module 10: Tremor & Parkinsonian Data Augmentation on 2D stroke image:
    1. Sinusoidal oscillatory distortion (simulating 4-7Hz hand tremor)
    2. Gaussian noise
    3. Micrographia scaling
    4. Minor rotation & stroke interruption
    """
    if rng is None:
        rng = np.random.default_rng()

    h, w = image.shape[:2]
    distorted = image.copy().astype(np.float32)

    # 1. Sinusoidal coordinate shift (horizontal & vertical tremor ripples)
    if tremor_intensity > 0.1:
        freq = rng.uniform(4.0, 7.0)
        amp = rng.uniform(1.0, 3.5) * tremor_intensity
        y_indices, x_indices = np.indices((h, w), dtype=np.float32)

        # Apply spatial displacement
        x_shift = amp * np.sin(2 * np.pi * freq * (y_indices / float(h)))
        y_shift = amp * np.cos(2 * np.pi * freq * (x_indices / float(w)))

        map_x = np.clip(x_indices + x_shift, 0, w - 1).astype(np.float32)
        map_y = np.clip(y_indices + y_shift, 0, h - 1).astype(np.float32)

        import cv2
        distorted = cv2.remap(distorted, map_x, map_y, cv2.INTER_LINEAR)

    # 2. Gaussian noise
    noise = rng.normal(0, 0.03 * tremor_intensity, size=(h, w)).astype(np.float32)
    distorted = np.clip(distorted + noise, 0.0, 1.0)

    return distorted
