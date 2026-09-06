"""
Trajectory Transformer Architecture.
Uses multi-head self-attention to model long-range stroke dependencies
and compensate for non-uniform writing speed in Parkinsonian air-writing.
"""

import math
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from data.character_templates import ALL_CLASSES, IDX_TO_CLASS, NUM_CLASSES


class PositionalEncoding(nn.Module):
    """Sinusoidal Positional Encoding."""

    def __init__(self, d_model: int, max_len: int = 256):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, d_model)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class TrajectoryTransformer(nn.Module):
    """
    Transformer Encoder for Air-Writing Intention Recognition.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = NUM_CLASSES,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 128,
        dropout: float = 0.2
    ):
        super().__init__()
        self.in_channels = in_channels
        self.d_model = d_model

        self.input_projection = nn.Linear(in_channels, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Tensor of shape (B, C, L) e.g. (B, 3, 64)
        Returns:
            logits: (B, num_classes)
            pooled: (B, d_model)
        """
        # Convert (B, C, L) -> (B, L, C)
        x_seq = x.transpose(1, 2)
        proj = self.input_projection(x_seq) * math.sqrt(self.d_model)
        proj = self.pos_encoder(proj)

        encoded = self.transformer_encoder(proj)  # (B, L, d_model)

        # Global average pooling + max pooling
        avg_pool = torch.mean(encoded, dim=1)
        logits = self.classifier(avg_pool)
        return logits, avg_pool

    def predict_single(
        self,
        trajectory: np.ndarray,
        device: torch.device = torch.device("cpu"),
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        self.eval()
        traj = np.asarray(trajectory, dtype=np.float32)
        if traj.shape[1] == 2:
            z = np.zeros((traj.shape[0], 1), dtype=np.float32)
            traj = np.hstack([traj, z])

        x_tensor = torch.from_numpy(traj).transpose(0, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            logits, _ = self(x_tensor)
            probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        top_indices = np.argsort(probs)[::-1][:top_k]
        return [(IDX_TO_CLASS[idx], float(probs[idx])) for idx in top_indices]
