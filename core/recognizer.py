"""Simple sign recognizer wrapper for Nocturne.

Provides a lightweight interface to load a scikit-learn / joblib model
that predicts solfege labels from MediaPipe hand landmarks.
"""
from pathlib import Path
import numpy as np

try:
    import joblib
except Exception:
    joblib = None


class SignRecognizer:
    def __init__(self, model_path):
        self.model_path = Path(model_path)
        self.model = None
        if self.model_path.exists() and joblib is not None:
            try:
                self.model = joblib.load(self.model_path)
            except Exception:
                self.model = None

    def is_loaded(self):
        return self.model is not None

    def landmarks_to_features(self, hand_landmarks):
        """Convert MediaPipe hand_landmarks to a normalized feature vector.

        hand_landmarks: iterable of 21 landmarks with .x and .y
        returns: 1D numpy array
        """
        pts = np.array([[lm.x, lm.y] for lm in hand_landmarks])  # (21,2)
        # Normalize: translate so wrist (0) at origin
        wrist = pts[0].copy()
        pts -= wrist
        # Scale by max distance to keep invariant to size
        scale = np.max(np.linalg.norm(pts, axis=1)) + 1e-8
        pts /= scale
        return pts.reshape(-1)

    def predict(self, hand_landmarks):
        if not self.is_loaded():
            return None
        X = self.landmarks_to_features(hand_landmarks)[None, :]
        try:
            return self.model.predict(X)[0]
        except Exception:
            return None
