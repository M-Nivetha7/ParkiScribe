"""
Deep Spatiotemporal Model: Multi-Scale 1D-CNN + Bidirectional LSTM + Temporal Attention.
Designed for intention recognition from tremor-affected and irregular air-writing trajectories.
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from data.character_templates import ALL_CLASSES, IDX_TO_CLASS, NUM_CLASSES


class TemporalAttention(nn.Module):
    """
    Learns temporal importance weights over trajectory sequence time steps,
    downweighting tremor bursts and hesitations while emphasizing intentional strokes.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, hidden_dim)
        Returns:
            context: Tensor of shape (batch_size, hidden_dim)
            weights: Tensor of shape (batch_size, seq_len, 1)
        """
        scores = self.projection(x)  # (B, L, 1)
        weights = F.softmax(scores, dim=1)  # (B, L, 1)
        context = torch.sum(x * weights, dim=1)  # (B, hidden_dim)
        return context, weights


class Conv1DBiLSTMAttention(nn.Module):
    """
    Hybrid Multi-Scale 1D-CNN + BiLSTM + Attention Network for Air-Writing Trajectories.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = NUM_CLASSES,
        conv_channels: int = 64,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # 1. Multi-scale 1D Conv Blocks (kernels 3, 5, 7)
        self.conv3 = nn.Conv1d(in_channels, conv_channels // 3, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(in_channels, conv_channels // 3, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(in_channels, conv_channels - 2 * (conv_channels // 3), kernel_size=7, padding=3)

        self.bn_conv = nn.BatchNorm1d(conv_channels)
        self.gelu = nn.GELU()
        self.drop_conv = nn.Dropout(dropout)

        # Second Conv refinement layer
        self.conv2 = nn.Conv1d(conv_channels, conv_channels, kernel_size=3, padding=1)
        self.bn_conv2 = nn.BatchNorm1d(conv_channels)

        # 2. Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0
        )

        bi_hidden = lstm_hidden * 2

        # 3. Temporal Attention Pooling
        self.attention = TemporalAttention(bi_hidden)

        # 4. Dense Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(bi_hidden, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor of shape (B, C, L) e.g. (B, 3, 64)
        Returns:
            logits: (B, num_classes)
            attn_weights: (B, L, 1)
        """
        # Multi-scale conv
        c3 = self.conv3(x)
        c5 = self.conv5(x)
        c7 = self.conv7(x)
        feat = torch.cat([c3, c5, c7], dim=1)  # (B, conv_channels, L)
        feat = self.drop_conv(self.gelu(self.bn_conv(feat)))

        feat = self.drop_conv(self.gelu(self.bn_conv2(self.conv2(feat))))

        # Reshape for LSTM: (B, L, conv_channels)
        feat_seq = feat.transpose(1, 2)

        lstm_out, _ = self.lstm(feat_seq)  # (B, L, 2 * lstm_hidden)

        context, attn_weights = self.attention(lstm_out)  # (B, 2 * lstm_hidden)

        logits = self.classifier(context)  # (B, num_classes)
        return logits, attn_weights

    def predict_single(
        self,
        trajectory: np.ndarray,
        device: torch.device = torch.device("cpu"),
        top_k: int = 5
    ) -> Tuple[List[Tuple[str, float]], np.ndarray]:
        """
        Predict single trajectory of shape (N, 3) or (N, 2).
        Returns top_k predictions and attention weight array.
        """
        self.eval()
        traj = np.asarray(trajectory, dtype=np.float32)
        if traj.shape[1] == 2:
            z = np.zeros((traj.shape[0], 1), dtype=np.float32)
            traj = np.hstack([traj, z])

        # Convert to tensor (1, 3, L)
        x_tensor = torch.from_numpy(traj).transpose(0, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            logits, attn = self(x_tensor)
            probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            attn_arr = attn.squeeze().cpu().numpy()

        top_indices = np.argsort(probs)[::-1][:top_k]
        top_preds = [(IDX_TO_CLASS[idx], float(probs[idx])) for idx in top_indices]
        return top_preds, attn_arr
