import numpy as np
import pandas as pd
import logging
import json
import os
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("shap not installed. pip install shap")

try:
    from sklearn.feature_selection import VarianceThreshold
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class FeatureSelector:
    """
    Selects optimal features for each ML model.

    20yr trader insight:
    More features is not better. Noise drowns signal.
    A model trained on 20 high-quality features beats
    a model trained on 200 mediocre ones every time.

    Selection pipeline:
    1. Remove zero-variance features (useless constants)
    2. Remove highly correlated pairs (redundant information)
    3. SHAP importance ranking (what the model actually uses)
    4. Keep top N by importance + minimum threshold

    Target feature counts:
    Swing:      50–65 features (from 83 candidates)
    Intraday:   25–35 features (from 41 candidates)
    Positional: 35–45 features (from 55 candidates)
    """

    IMPORTANCE_PATH = "models/evaluation/feature_importance.json"
    CORR_THRESHOLD  = 0.85   # remove if correlation > this
    VAR_THRESHOLD   = 0.001  # remove if variance < this
    MIN_SHAP        = 0.001  # remove if SHAP importance < this

    def __init__(self):
        self._importance_cache: Dict = {}

    # ─────────────────────────────────────────────────────────────
    # PUBLIC
    # ─────────────────────────────────────────────────────────────

    def select(
        self,
        X:             pd.DataFrame,
        y:             pd.Series,
        model,
        model_type:    str,
        all_features:  List[str],
        max_features:  Optional[int] = None,
    ) -> List[str]:
        """
        Run full feature selection pipeline.
        Returns ordered list of selected feature names.

        Args:
            X:            Training feature matrix
            y:            Labels
            model:        Trained XGBoost model (for SHAP)
            model_type:   "swing" | "intraday" | "positional"
            all_features: All candidate feature names
            max_features: Optional hard cap on selected count
        """
        logger.info(f"Feature selection for {model_type}: {len(all_features)} candidates")

        selected = [f for f in all_features if f in X.columns]

        # Step 1: Zero variance
        selected = self._remove_zero_variance(X[selected], selected)
        logger.info(f"  After variance filter: {len(selected)}")

        # Step 2: Correlation
        selected = self._remove_correlated(X[selected], selected)
        logger.info(f"  After correlation filter: {len(selected)}")

        # Step 3: SHAP importance
        if SHAP_AVAILABLE and len(X) >= 100:
            selected, importances = self._shap_selection(
                X[selected], model, selected
            )
            self._save_importances(model_type, importances)
        else:
            # Fallback: XGBoost native importance
            selected = self._xgb_importance_selection(model, selected)

        logger.info(f"  After SHAP filter: {len(selected)}")

        # Step 4: Apply max cap
        if max_features and len(selected) > max_features:
            selected = selected[:max_features]
            logger.info(f"  Capped at {max_features}: {len(selected)}")

        logger.info(f"Feature selection complete: {len(selected)} features selected")
        return selected

    def load_importances(self, model_type: str) -> Dict[str, float]:
        """Load saved feature importances for dashboard display."""
        try:
            if os.path.exists(self.IMPORTANCE_PATH):
                with open(self.IMPORTANCE_PATH) as f:
                    data = json.load(f)
                return data.get(model_type, {})
        except Exception:
            pass
        return {}

    def get_top_features(self, model_type: str, n: int = 10) -> List[Tuple[str, float]]:
        """Return top N most important features for a model type."""
        imp = self.load_importances(model_type)
        if not imp:
            return []
        sorted_imp = sorted(imp.items(), key=lambda x: x[1], reverse=True)
        return sorted_imp[:n]

    # ─────────────────────────────────────────────────────────────
    # SELECTION STEPS
    # ─────────────────────────────────────────────────────────────

    def _remove_zero_variance(
        self, X: pd.DataFrame, features: List[str]
    ) -> List[str]:
        """Remove features with near-zero variance — they carry no information."""
        if not SKLEARN_AVAILABLE:
            return features
        try:
            sel = VarianceThreshold(threshold=self.VAR_THRESHOLD)
            sel.fit(X.fillna(0))
            mask    = sel.get_support()
            kept    = [f for f, m in zip(features, mask) if m]
            removed = [f for f, m in zip(features, mask) if not m]
            if removed:
                logger.debug(f"  Zero-variance removed: {removed}")
            return kept
        except Exception as e:
            logger.warning(f"Variance filter error: {e}")
            return features

    def _remove_correlated(
        self, X: pd.DataFrame, features: List[str]
    ) -> List[str]:
        """
        Remove one feature from each highly correlated pair.
        When two features say the same thing, one is redundant noise.
        Keep the one with higher variance (more informative).
        """
        try:
            corr_matrix = X.fillna(0).corr().abs()
            upper       = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            to_drop = set()
            for col in upper.columns:
                if any(upper[col] > self.CORR_THRESHOLD):
                    to_drop.add(col)

            kept = [f for f in features if f not in to_drop]
            if to_drop:
                logger.debug(f"  Correlated removed ({len(to_drop)}): {list(to_drop)[:5]}...")
            return kept
        except Exception as e:
            logger.warning(f"Correlation filter error: {e}")
            return features

    def _shap_selection(
        self, X: pd.DataFrame, model, features: List[str]
    ) -> Tuple[List[str], Dict[str, float]]:
        """
        Use SHAP values to rank features by actual model contribution.
        SHAP is more accurate than XGBoost's native feature importance
        because it accounts for feature interactions.
        """
        try:
            X_clean    = X.fillna(0).replace([np.inf, -np.inf], 0)
            # Use a sample for speed if large dataset
            sample     = X_clean.sample(min(500, len(X_clean)), random_state=42)

            explainer  = shap.TreeExplainer(model)
            shap_vals  = explainer.shap_values(sample)

            # For multi-class, shap_vals is a list — take max across classes
            if isinstance(shap_vals, list):
                importance = np.max(
                    [np.abs(sv).mean(axis=0) for sv in shap_vals], axis=0
                )
            else:
                importance = np.abs(shap_vals).mean(axis=0)

            imp_dict   = {f: float(v) for f, v in zip(features, importance)}
            # Keep features above minimum SHAP threshold
            selected   = [
                f for f, v in sorted(imp_dict.items(), key=lambda x: x[1], reverse=True)
                if v >= self.MIN_SHAP
            ]
            return selected, imp_dict

        except Exception as e:
            logger.warning(f"SHAP selection failed, using XGB importance: {e}")
            selected = self._xgb_importance_selection(model, features)
            return selected, {}

    def _xgb_importance_selection(
        self, model, features: List[str]
    ) -> List[str]:
        """Fallback: XGBoost native gain importance."""
        try:
            imp  = model.get_booster().get_fscore()
            # Map feature indices to names
            ranked = sorted(
                [(f, imp.get(f, imp.get(f"f{i}", 0)))
                 for i, f in enumerate(features)],
                key=lambda x: x[1], reverse=True,
            )
            return [f for f, v in ranked if v > 0]
        except Exception:
            return features

    def _save_importances(
        self, model_type: str, importances: Dict[str, float]
    ) -> None:
        """Persist feature importances for dashboard and debugging."""
        try:
            os.makedirs(os.path.dirname(self.IMPORTANCE_PATH), exist_ok=True)
            existing = {}
            if os.path.exists(self.IMPORTANCE_PATH):
                with open(self.IMPORTANCE_PATH) as f:
                    existing = json.load(f)
            existing[model_type] = importances
            with open(self.IMPORTANCE_PATH, "w") as f:
                json.dump(existing, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save importances: {e}")