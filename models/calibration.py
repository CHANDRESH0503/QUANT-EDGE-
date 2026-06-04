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


class PreFitSigmoidCalibrator:
    """
    Per-class Platt (sigmoid) calibration for a pre-fitted multi-class
    classifier.

    Isotonic regression needs a few hundred samples to be reliable; on a
    small calibration set (e.g. the last 15-20% of ~1,300 weekly positional
    samples, or a thin per-bank tail) it over-fits and collapses — a class
    with one noisy region gets mapped to a near-step function, which is how
    the positional/intraday models ended up pinned to one class. Platt
    scaling fits a single logistic per class (one parameter pair), so it
    stays smooth and well-behaved on small n.

    Use via `make_calibrator(base, n_cal)` which picks this for small n.
    """

    def __init__(self, base_estimator):
        self.base_estimator = base_estimator
        self.calibrators_   = None
        self.classes_       = base_estimator.classes_

    def fit(self, X, y):
        from sklearn.linear_model import LogisticRegression
        probs = self.base_estimator.predict_proba(X)
        n_classes = probs.shape[1]
        y = np.asarray(y)
        self.calibrators_ = []
        for c in range(n_classes):
            target = (y == c).astype(int)
            # A class absent (or omnipresent) in the calibration tail can't
            # train a logistic — fall back to its empirical base rate.
            if target.min() == target.max():
                self.calibrators_.append(("const", float(target.mean())))
                continue
            try:
                lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
                lr.fit(probs[:, c].reshape(-1, 1), target)
                self.calibrators_.append(("lr", lr))
            except Exception:
                self.calibrators_.append(("const", float(target.mean())))
        return self

    def predict_proba(self, X):
        probs = self.base_estimator.predict_proba(X)
        cal = np.zeros_like(probs)
        for c, (kind, est) in enumerate(self.calibrators_):
            if kind == "const":
                cal[:, c] = est
            else:
                cal[:, c] = est.predict_proba(probs[:, c].reshape(-1, 1))[:, 1]
        cal = np.clip(cal, 1e-4, 1 - 1e-4)
        row_sums = cal.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        return cal / row_sums

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


# Isotonic needs a few hundred samples to be trustworthy; below this we use
# Platt scaling instead (smooth, 1-parameter-per-class — robust on small n).
ISOTONIC_MIN_SAMPLES = 150


def make_calibrator(base_estimator, n_cal_samples: int):
    """
    Pick the right calibrator for the calibration-set size.

    Large set → isotonic (flexible, non-parametric).
    Small set → Platt/sigmoid (stable, avoids the small-n collapse that
                pinned the positional/intraday models to a single class).
    """
    if n_cal_samples >= ISOTONIC_MIN_SAMPLES:
        return PreFitIsotonicCalibrator(base_estimator)
    return PreFitSigmoidCalibrator(base_estimator)
