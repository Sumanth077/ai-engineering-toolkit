"""Data models for the intraday trading analyzer."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional


@dataclass
class MACDData:
    value: float
    signal: float
    histogram: float


@dataclass
class BollingerBands:
    upper: float
    middle: float
    lower: float


@dataclass
class TimeframeIndicators:
    ema9: float
    ema21: float
    rsi14: float
    macd: MACDData
    atr14: float
    ema50: Optional[float] = None
    bbands: Optional[BollingerBands] = None
    volume_sma20: Optional[float] = None


@dataclass
class PriceSummary:
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: float
    avg_volume: Optional[int] = None
    volatility_percentile: Optional[float] = None


@dataclass
class SupportResistance:
    supports: list[float] = field(default_factory=list)
    resistances: list[float] = field(default_factory=list)


@dataclass
class NewsSentiment:
    impact: str  # High, Medium, Low, None
    sentiment: str  # Bullish, Bearish, Neutral
    headlines: list[str] = field(default_factory=list)


@dataclass
class AnalyzerInput:
    ticker: str
    current_price: float
    indicators: dict[str, TimeframeIndicators]  # keys: "1m", "15m", "1h"
    price_summary: PriceSummary
    support_resistance: SupportResistance
    news_sentiment: NewsSentiment

    def validate(self) -> list[str]:
        """Return a list of validation errors, empty if valid."""
        errors = []
        if self.current_price <= 0:
            errors.append("current_price must be positive")
        required_tfs = {"1m", "15m", "1h"}
        missing = required_tfs - set(self.indicators.keys())
        if missing:
            errors.append(f"Missing timeframe indicators: {missing}")
        for tf_key, ind in self.indicators.items():
            if ind.atr14 < 0:
                errors.append(f"{tf_key} ATR must be non-negative")
            if not (0 <= ind.rsi14 <= 100):
                errors.append(f"{tf_key} RSI must be 0-100")
        if self.news_sentiment.impact not in ("High", "Medium", "Low", "None"):
            errors.append(f"Invalid news impact: {self.news_sentiment.impact}")
        if self.news_sentiment.sentiment not in ("Bullish", "Bearish", "Neutral"):
            errors.append(f"Invalid news sentiment: {self.news_sentiment.sentiment}")
        return errors


@dataclass
class SignalMeta:
    hourly_trend: str  # Bullish, Bearish, Neutral
    fifteen_min_setup: str
    one_min_timing: str
    news_impact: str  # High, Medium, Low, None
    news_sentiment_conflict: bool = False


@dataclass
class TradeSignal:
    ticker: str
    timestamp: str
    action: str  # Buy, Sell, Hold
    confidence: int
    reasons: list[str]
    invalidations: list[str]
    meta: SignalMeta
    entry: Optional[float] = None
    stop: Optional[float] = None
    target1: Optional[float] = None
    target2: Optional[float] = None
    risk_reward: Optional[float] = None
    position_size_pct: Optional[float] = None

    def to_dict(self) -> dict:
        """Convert to a plain dict matching the output JSON schema."""
        result = {
            "ticker": self.ticker,
            "timestamp": self.timestamp,
            "action": self.action,
            "confidence": self.confidence,
            "entry": self.entry,
            "stop": self.stop,
            "target1": self.target1,
            "target2": self.target2,
            "risk_reward": self.risk_reward,
            "position_size_pct": self.position_size_pct,
            "reasons": self.reasons,
            "invalidations": self.invalidations,
            "meta": {
                "hourly_trend": self.meta.hourly_trend,
                "fifteen_min_setup": self.meta.fifteen_min_setup,
                "one_min_timing": self.meta.one_min_timing,
                "news_impact": self.meta.news_impact,
                "news_sentiment_conflict": self.meta.news_sentiment_conflict,
            },
        }
        return result


def parse_input(data: dict) -> AnalyzerInput:
    """Parse a raw dict (matching input schema) into an AnalyzerInput."""
    indicators = {}
    for tf_key in ("1m", "15m", "1h"):
        raw = data["indicators"][tf_key]
        macd = MACDData(**raw["macd"])
        bbands = None
        if raw.get("bbands"):
            bbands = BollingerBands(**raw["bbands"])
        indicators[tf_key] = TimeframeIndicators(
            ema9=raw["ema9"],
            ema21=raw["ema21"],
            rsi14=raw["rsi14"],
            macd=macd,
            atr14=raw["atr14"],
            ema50=raw.get("ema50"),
            bbands=bbands,
            volume_sma20=raw.get("volume_sma20"),
        )

    price_summary = PriceSummary(
        open=data["price_summary"]["open"],
        high=data["price_summary"]["high"],
        low=data["price_summary"]["low"],
        close=data["price_summary"]["close"],
        volume=data["price_summary"]["volume"],
        vwap=data["price_summary"]["vwap"],
        avg_volume=data["price_summary"].get("avg_volume"),
        volatility_percentile=data["price_summary"].get("volatility_percentile"),
    )

    sr = SupportResistance(
        supports=data["support_resistance"]["supports"],
        resistances=data["support_resistance"]["resistances"],
    )

    news = NewsSentiment(
        impact=data["news_sentiment"]["impact"],
        sentiment=data["news_sentiment"]["sentiment"],
        headlines=data["news_sentiment"].get("headlines", []),
    )

    return AnalyzerInput(
        ticker=data["ticker"],
        current_price=data["current_price"],
        indicators=indicators,
        price_summary=price_summary,
        support_resistance=sr,
        news_sentiment=news,
    )
