"""
Classical Machine Learning Baseline Classifiers.
Supports Random Forest, SVM (RBF kernel), and Gradient Boosting on kinematic/spectral features.
"""

import os
import pickle
from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

from preprocessing.feature_extractor import TrajectoryFeatureExtractor
from data.character_templates import ALL_CLASSES, CLASS_TO_IDX, IDX_TO_CLASS


class BaselineTrajectoryClassifier:
    """
    Classical ML model trained on engineered kinematic, geometric, and spectral features.
    """

    def __init__(self, model_type: str = "rf", random_state: int = 42):
        """
        Args:
            model_type: "rf" (Random Forest), "svm" (Support Vector Machine), or "gbdt" (Gradient Boosting)
        """
        self.model_type = model_type
        self.random_state = random_state
        self.feature_extractor = TrajectoryFeatureExtractor()

        if model_type == "rf":
            base_clf = RandomForestClassifier(
                n_estimators=150,
                max_depth=16,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1
            )
        elif model_type == "svm":
            base_clf = SVC(
                C=10.0,
                kernel="rbf",
                gamma="scale",
                probability=True,
                class_weight="balanced",
                random_state=random_state
            )
        elif model_type == "gbdt":
            base_clf = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=random_state
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}. Choose rf, svm, or gbdt.")

        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", base_clf)
        ])

    def _extract_matrix(self, trajectories: np.ndarray) -> np.ndarray:
        features = [self.feature_extractor.extract_features(t) for t in trajectories]
        return np.array(features, dtype=np.float32)

    def fit(self, X_trajectories: np.ndarray, y_labels: np.ndarray):
        """Train the baseline classifier."""
        X_feats = self._extract_matrix(X_trajectories)
        self.pipeline.fit(X_feats, y_labels)

    def predict(self, X_trajectories: np.ndarray) -> np.ndarray:
        """Predict class indices for trajectories."""
        X_feats = self._extract_matrix(X_trajectories)
        return self.pipeline.predict(X_feats)

    def predict_proba(self, X_trajectories: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        X_feats = self._extract_matrix(X_trajectories)
        return self.pipeline.predict_proba(X_feats)

    def predict_single(self, trajectory: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Predict single trajectory and return top_k (character, confidence) tuples.
        """
        feat = self.feature_extractor.extract_features(trajectory).reshape(1, -1)
        probs = self.pipeline.predict_proba(feat)[0]
        top_indices = np.argsort(probs)[::-1][:top_k]
        return [(IDX_TO_CLASS[idx], float(probs[idx])) for idx in top_indices]

    def evaluate(self, X_trajectories: np.ndarray, y_labels: np.ndarray) -> Dict[str, any]:
        """Evaluate on test set."""
        preds = self.predict(X_trajectories)
        acc = accuracy_score(y_labels, preds)
        report = classification_report(y_labels, preds, target_names=ALL_CLASSES, output_dict=True, zero_division=0)
        return {"accuracy": float(acc), "report": report}

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self.pipeline, f)
        print(f"Saved {self.model_type} baseline model to {filepath}")

    def load(self, filepath: str):
        with open(filepath, "rb") as f:
            self.pipeline = pickle.load(f)
        print(f"Loaded {self.model_type} baseline model from {filepath}")
