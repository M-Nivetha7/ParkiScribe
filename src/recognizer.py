import os
import sys
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
import torch

from utils.constants import ALL_CLASSES, CLASS_TO_IDX, IDX_TO_CLASS, NUM_CLASSES, MODEL_INPUT_SIZE
from src.model import AirWritingCNN
from src.preprocessing import preprocess_canvas
from nlp.word_recognizer import AssistiveNLPDecoder


class AirWritingRecognizer:
    """
    Character and Word Recognition Engine for Air-Writing.
    Combines 2D CNN recognition on preprocessed canvas images with
    multi-character word segmentation and beam search language modeling.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[torch.device] = None
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.cnn_model = AirWritingCNN(num_classes=NUM_CLASSES, input_channels=1).to(self.device)
        self.nlp_decoder = AssistiveNLPDecoder()
        self.is_trained = False

        if model_path is None:
            default_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "saved_models", "airwriting_cnn.pt")
            if os.path.exists(default_path):
                model_path = default_path

        if model_path and os.path.exists(model_path):
            self.load_weights(model_path)

    def load_weights(self, path: str) -> None:
        """Load trained model weights."""
        checkpoint = torch.load(path, map_location=self.device)
        if "model_state_dict" in checkpoint:
            self.cnn_model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.cnn_model.load_state_dict(checkpoint)
        self.cnn_model.eval()
        self.is_trained = True

    def save_weights(self, path: str) -> None:
        """Save model weights."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({"model_state_dict": self.cnn_model.state_dict()}, path)

    def predict_image(
        self,
        img_normalized: np.ndarray,
        top_k: int = 3
    ) -> Dict[str, Any]:
        """
        Predict character from normalized 2D image array of shape (H, W).
        """
        self.cnn_model.eval()
        tensor = torch.from_numpy(img_normalized).unsqueeze(0).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            logits = self.cnn_model(tensor)
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        top_indices = np.argsort(probs)[::-1][:top_k]
        top_candidates = [(IDX_TO_CLASS[idx], float(probs[idx])) for idx in top_indices]

        top_char, top_conf = top_candidates[0]

        return {
            "predicted_char": top_char,
            "confidence": round(top_conf * 100.0, 2),
            "top_candidates": [
                {"char": c, "confidence": round(p * 100.0, 2)}
                for c, p in top_candidates
            ],
            "all_probabilities": probs.tolist()
        }

    def predict_canvas(
        self,
        canvas_bgr: np.ndarray,
        top_k: int = 3
    ) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        """
        Full end-to-end inference from raw canvas:
        Preprocess -> Multi-character Segmentation -> CNN Predict -> Beam Search Word Decoding.
        Supports both isolated characters and full multi-character words.
        """
        norm_img, meta = preprocess_canvas(canvas_bgr)
        if not meta.get("has_content", False):
            return None, meta

        seg_chars = meta.get("segmented_characters", [])

        # If multiple characters are detected on the canvas (User wrote a word!)
        if len(seg_chars) > 1:
            char_preds = []
            char_posteriors = []
            char_details = []

            for char_norm, char_box in seg_chars:
                p = self.predict_image(char_norm, top_k=max(3, top_k))
                char_preds.append(p["predicted_char"])
                char_posteriors.append([(c["char"], c["confidence"] / 100.0) for c in p["top_candidates"]])
                char_details.append({
                    "char": p["predicted_char"],
                    "confidence": p["confidence"],
                    "candidates": p["top_candidates"],
                    "bbox": char_box
                })

            raw_word = "".join(char_preds)
            beam_res = self.nlp_decoder.beam_search_decode(char_posteriors, beam_width=6)

            # Determine best word: vocabulary match or raw decoded string
            if beam_res:
                best_word = beam_res[0][0]
                top_cands = [{"char": w, "confidence": round(prob * 100.0, 2)} for w, prob in beam_res[:top_k]]
            else:
                best_word = raw_word
                top_cands = [{"char": raw_word, "confidence": 95.0}]

            avg_conf = round(float(np.mean([c["confidence"] for c in char_details])), 2)

            pred = {
                "is_word": True,
                "predicted_char": best_word,  # Backward compatible
                "predicted_word": best_word,
                "raw_characters": raw_word,
                "confidence": avg_conf,
                "characters": char_details,
                "top_candidates": top_cands
            }
            return pred, meta

        # Single character inference
        pred = self.predict_image(norm_img, top_k=top_k)
        pred["is_word"] = False
        pred["predicted_word"] = pred["predicted_char"]
        return pred, meta

