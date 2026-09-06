"""
PyTorch Dataset and DataLoader pipelines for air-writing trajectories.
Includes stratified dataset partitioning and real-time trajectory augmentation.
"""

from typing import Dict, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class TrajectoryAugmenter:
    """Data augmentation transforms for air-writing trajectories."""

    def __init__(self, rng_seed: Optional[int] = None):
        self.rng = np.random.default_rng(rng_seed)

    def __call__(self, traj: np.ndarray) -> np.ndarray:
        t = np.copy(traj)

        # 1. Random small rotation (-8 to +8 degrees)
        if self.rng.random() > 0.4:
            theta = np.radians(self.rng.uniform(-8.0, 8.0))
            c, s = np.cos(theta), np.sin(theta)
            R = np.array([[c, -s], [s, c]], dtype=np.float32)
            center = np.mean(t[:, :2], axis=0, keepdims=True)
            t[:, :2] = (t[:, :2] - center) @ R.T + center

        # 2. Random scaling (0.9 to 1.1)
        if self.rng.random() > 0.4:
            scale = self.rng.uniform(0.9, 1.1)
            center = np.mean(t[:, :2], axis=0, keepdims=True)
            t[:, :2] = center + (t[:, :2] - center) * scale

        # 3. Random small translation
        if self.rng.random() > 0.4:
            t[:, 0] += self.rng.uniform(-0.03, 0.03)
            t[:, 1] += self.rng.uniform(-0.03, 0.03)

        # 4. Small additive noise
        if self.rng.random() > 0.5:
            t += self.rng.normal(0.0, 0.004, size=t.shape).astype(np.float32)

        return t


class AirWritingDataset(Dataset):
    """PyTorch Dataset for air-writing trajectories."""

    def __init__(
        self,
        trajectories: np.ndarray,
        labels: np.ndarray,
        severities: Optional[np.ndarray] = None,
        augment: bool = False,
        rng_seed: Optional[int] = None
    ):
        self.trajectories = trajectories.astype(np.float32)
        self.labels = labels.astype(np.int64)
        self.severities = severities
        self.augment = augment
        self.augmenter = TrajectoryAugmenter(rng_seed=rng_seed) if augment else None

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        traj = self.trajectories[idx]
        if self.augment and self.augmenter is not None:
            traj = self.augmenter(traj)

        # Output shape: (C, L) where C=3 (x, y, z) and L=sequence_length (e.g. 64)
        traj_tensor = torch.from_numpy(traj).transpose(0, 1)  # (3, 64)
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)
        return traj_tensor, label_tensor


def create_dataloaders(
    dataset_dict: Dict[str, np.ndarray],
    batch_size: int = 32,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, any]]:
    """
    Split dataset into stratified Train/Val/Test loaders.
    """
    X = dataset_dict["trajectories"]
    y = dataset_dict["labels"]
    sev = dataset_dict.get("severities", np.array(["unknown"] * len(y)))

    # First split: train vs (val + test)
    val_test_ratio = val_ratio + test_ratio
    X_train, X_temp, y_train, y_temp, sev_train, sev_temp = train_test_split(
        X, y, sev, test_size=val_test_ratio, stratify=y, random_state=random_state
    )

    # Second split: val vs test
    val_rel_ratio = val_ratio / val_test_ratio
    X_val, X_test, y_val, y_test, sev_val, sev_test = train_test_split(
        X_temp, y_temp, sev_temp, test_size=(1.0 - val_rel_ratio), stratify=y_temp, random_state=random_state
    )

    train_ds = AirWritingDataset(X_train, y_train, sev_train, augment=True, rng_seed=random_state)
    val_ds = AirWritingDataset(X_val, y_val, sev_val, augment=False)
    test_ds = AirWritingDataset(X_test, y_test, sev_test, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    split_info = {
        "num_train": len(X_train),
        "num_val": len(X_val),
        "num_test": len(X_test),
        "test_severities": sev_test,
        "test_labels": y_test,
        "test_trajectories": X_test
    }
    return train_loader, val_loader, test_loader, split_info
