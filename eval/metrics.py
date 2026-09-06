"""
Evaluation Metrics & Clinical Assistive Performance Indicators.
Computes Accuracy, Top-k Accuracy, Macro/Weighted F1, Confusion Matrix,
Character Error Rate (CER), Word Error Rate (WER), and Latency benchmarks.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from data.character_templates import ALL_CLASSES


class EvaluationMetrics:
    """
    Computes standard ML and assistive communication metrics.
    """

    @staticmethod
    def top_k_accuracy(y_true: np.ndarray, y_probs: np.ndarray, k: int = 3) -> float:
        """Compute Top-k accuracy."""
        top_k_preds = np.argsort(y_probs, axis=1)[:, -k:]
        correct = np.any(top_k_preds == y_true[:, None], axis=1)
        return float(np.mean(correct))

    @staticmethod
    def compute_all_metrics(
        y_true: np.ndarray,
        y_probs: np.ndarray,
        severities: Optional[np.ndarray] = None
    ) -> Dict[str, any]:
        """
        Compute comprehensive classification metrics.
        """
        y_pred = np.argmax(y_probs, axis=1)
        acc = accuracy_score(y_true, y_pred)
        top1 = acc
        top3 = EvaluationMetrics.top_k_accuracy(y_true, y_probs, k=3)
        top5 = EvaluationMetrics.top_k_accuracy(y_true, y_probs, k=5)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )

        metrics = {
            "accuracy": float(acc),
            "top1_accuracy": float(top1),
            "top3_accuracy": float(top3),
            "top5_accuracy": float(top5),
            "macro_precision": float(precision),
            "macro_recall": float(recall),
            "macro_f1": float(f1),
            "weighted_f1": float(weighted_f1)
        }

        # Severity breakdown if available
        if severities is not None:
            sev_breakdown = {}
            for sev in np.unique(severities):
                mask = (severities == sev)
                if np.sum(mask) > 0:
                    sev_acc = accuracy_score(y_true[mask], y_pred[mask])
                    sev_top3 = EvaluationMetrics.top_k_accuracy(y_true[mask], y_probs[mask], k=3)
                    sev_breakdown[str(sev)] = {
                        "count": int(np.sum(mask)),
                        "top1_accuracy": float(sev_acc),
                        "top3_accuracy": float(sev_top3)
                    }
            metrics["severity_breakdown"] = sev_breakdown

        return metrics

    @staticmethod
    def character_error_rate(reference: str, hypothesis: str) -> float:
        """Compute Character Error Rate (CER = Levenshtein / len(ref))."""
        from nlp.word_recognizer import AssistiveNLPDecoder
        dist = AssistiveNLPDecoder.levenshtein_distance(reference, hypothesis)
        return float(dist / max(len(reference), 1))

    @staticmethod
    def word_error_rate(ref_words: List[str], hyp_words: List[str]) -> float:
        """Compute Word Error Rate (WER)."""
        if len(ref_words) == 0:
            return float(len(hyp_words))

        dp = np.zeros((len(ref_words) + 1, len(hyp_words) + 1), dtype=np.int32)
        for i in range(len(ref_words) + 1):
            dp[i, 0] = i
        for j in range(len(hyp_words) + 1):
            dp[0, j] = j

        for i in range(1, len(ref_words) + 1):
            for j in range(1, len(hyp_words) + 1):
                if ref_words[i-1] == hyp_words[j-1]:
                    dp[i, j] = dp[i-1, j-1]
                else:
                    dp[i, j] = 1 + min(dp[i-1, j], dp[i, j-1], dp[i-1, j-1])

        return float(dp[len(ref_words), len(hyp_words)] / len(ref_words))
