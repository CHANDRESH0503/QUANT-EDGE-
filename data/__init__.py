# data/__init__.py
# Data layer — all external data collection modules

from .price_fetcher import PriceFetcher
from .news_fetcher import NewsFetcher
from .bse_fetcher import BSEFetcher
from .options_fetcher import OptionsFetcher
from .fii_fetcher import FIIFetcher
from .global_fetcher import GlobalFetcher
from .social_fetcher import SocialFetcher
from .insider_fetcher import InsiderFetcher
from .alternative_fetcher import AlternativeFetcher

__all__ = [
    "PriceFetcher",
    "NewsFetcher",
    "BSEFetcher",
    "OptionsFetcher",
    "FIIFetcher",
    "GlobalFetcher",
    "SocialFetcher",
    "InsiderFetcher",
    "AlternativeFetcher",
]