# models/calibration.py
# Manual isotonic calibration for pre-fitted XGBoost models.
# Replaces CalibratedClassifierCV(cv='prefit') which was removed in sklearn 1.8.
import numpy as np
import joblib


class PreFitIsotonicCalibrator:
    """
    Wraps a pre-fitted multi-class classifier and applies per-class
    isotonic regression calibration on a held-out set.

    sklearn's CalibratedClassifierCV(cv='prefit') was removed in 1.8.
    This class replicates its behaviour while staying library-version-agnostic.
    """

    def __init__(self, base_estimator):
        self.base_estimator = base_estimator
        self.calibrators_   = None
        self.classes_       = base_estimator.classes_

    def fit(self, X, y):
        from sklearn.isotonic import IsotonicRegression
        probs = self.base_estimator.predict_proba(X)
        n_classes = probs.shape[1]
        self.calibrators_ = []
        for c in range(n_classes):
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(probs[:, c], (y == c).astype(int))
            self.calibrators_.append(iso)
        return self

    def predict_proba(self, X):
        probs = self.base_estimator.predict_proba(X)
        cal = np.zeros_like(probs)
        for c, iso in enumerate(self.calibrators_):
            cal[:, c] = iso.predict(probs[:, c])
        # Prevent isotonic clipping from collapsing any class to exactly 0 or 1
        # before renormalization — without this, prob_flat=0.98+ isotonic-clips to
        # exactly 1.0, zeroing the other classes and making the model always output FLAT.
        cal = np.clip(cal, 1e-4, 1 - 1e-4)
        row_sums = cal.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        return cal / row_sums

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
