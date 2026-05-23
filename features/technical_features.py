# features/technical_features.py
# Selects and normalises technical features for ML model input
# Pulls from processing/technical.py output
# Returns the exact 23 technical features selected by senior trader

import pandas as pd
import numpy as np
import logging
from typing import Dict

logger = logging.getLogger(__name__)


class TechnicalFeatures:
    """
    Selects the 23 most predictive technical features from the
    full technical processor output. Applies winsorisation and
    normalisation where needed for stable ML input.

    Features selected based on 20yr backtesting:
    - Momentum: RSI, StochRSI, MACD histogram + acceleration
    - Trend: EMA spread + rate of change, ADX, Supertrend
    - Volatility: BB%B, BB squeeze, ATR ratio, HV ratio
    - Volume: OBV ROC, MFI, CMF, volume ratio
    - Price action: returns 1/5/20d, close position in range,
                    52W range position, gap
    """

    # All feature names this class produces for ML
    FEATURE_NAMES = [
        "rsi_14", "stoch_rsi", "rsi_divergence",
        "macd_hist", "macd_hist_roc",
        "ema_spread", "ema_spread_roc", "adx", "supertrend",
        "bb_pct_b", "bb_squeeze", "atr_ratio", "hv_ratio",
        "obv_roc", "mfi", "cmf", "volume_ratio",
        "returns_1d", "returns_5d", "returns_20d",
        "close_pct_range", "gap_pct", "range_pct_52w",
    ]

    # Winsorise limits — clip extreme values to prevent ML instability
    CLIP_BOUNDS = {
        "rsi_14":        (0, 100),
        "stoch_rsi":     (0, 1),
        "macd_hist_roc": (-5, 5),
        "ema_spread":    (-0.15, 0.15),
        "ema_spread_roc":(-2, 2),
        "adx":           (0, 100),
        "bb_pct_b":      (-0.5, 1.5),
        "atr_ratio":     (0, 0.1),
        "hv_ratio":      (0, 5),
        "obv_roc":       (-3, 3),
        "volume_ratio":  (0, 10),
        "returns_1d":    (-0.15, 0.15),
        "returns_5d":    (-0.25, 0.25),
        "returns_20d":   (-0.40, 0.40),
        "gap_pct":       (-0.10, 0.10),
    }

    def extract(self, tech_df: pd.DataFrame) -> Dict:
        """
        Extract and clean technical features from TechnicalProcessor output.

        Args:
            tech_df: DataFrame from TechnicalProcessor.build_features()

        Returns:
            Dict of {feature_name: float} ready for ML input
        """
        if tech_df.empty:
            return self._zeros()

        # Take the last row (most recent bar)
        row = tech_df.iloc[-1]
        features = {}

        for feat in self.FEATURE_NAMES:
            val = float(row.get(feat, 0))

            # Replace NaN/inf
            if not np.isfinite(val):
                val = 0.0

            # Winsorise
            bounds = self.CLIP_BOUNDS.get(feat)
            if bounds:
                val = max(bounds[0], min(bounds[1], val))

            features[feat] = round(val, 6)

        return features

    def extract_sequence(self, tech_df: pd.DataFrame,
                          lookback: int = 20) -> np.ndarray:
        """
        Extract a 2D sequence for LSTM-based models.
        Shape: (lookback, n_features)
        """
        if len(tech_df) < lookback:
            return np.zeros((lookback, len(self.FEATURE_NAMES)))

        result = []
        for i in range(lookback, 0, -1):
            row = tech_df.iloc[-i]
            row_feats = []
            for feat in self.FEATURE_NAMES:
                val = float(row.get(feat, 0))
                if not np.isfinite(val):
                    val = 0.0
                row_feats.append(val)
            result.append(row_feats)

        return np.array(result)

    def _zeros(self) -> Dict:
        return {f: 0.0 for f in self.FEATURE_NAMES}