# processing/__init__.py
# Processing layer — converts raw data into ML-ready features
# Every processor takes raw data → returns clean numerical features
# All processors are stateless — they read from DB and return dicts

from .technical      import TechnicalProcessor
from .finbert_sentiment import FinBERTSentiment
from .llm_analyzer   import LLMAnalyzer
from .fundamental    import FundamentalProcessor
from .macro_analyzer import MacroAnalyzer
from .pattern_detector import PatternDetector
from .anomaly_detector import AnomalyDetector
from .earnings_predictor import EarningsPredictor
from .order_flow     import OrderFlowAnalyzer
from .support_resistance import SupportResistanceEngine
from .intermarket    import IntermarketAnalyzer
from .regime_detector import RegimeDetector

__all__ = [
    "TechnicalProcessor", "FinBERTSentiment", "LLMAnalyzer",
    "FundamentalProcessor", "MacroAnalyzer", "PatternDetector",
    "AnomalyDetector", "EarningsPredictor", "OrderFlowAnalyzer",
    "SupportResistanceEngine", "IntermarketAnalyzer", "RegimeDetector",
]