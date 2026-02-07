"""Tests for the intraday trading analyzer."""

from __future__ import annotations

import json
import os

import pytest

from intraday_analyzer.analyzer import IntradayAnalyzer
from intraday_analyzer.models import (
    AnalyzerInput,
    BollingerBands,
    MACDData,
    NewsSentiment,
    PriceSummary,
    SignalMeta,
    SupportResistance,
    TimeframeIndicators,
    TradeSignal,
    parse_input,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_indicators(
    ema9: float,
    ema21: float,
    rsi14: float,
    macd_hist: float,
    atr14: float,
    ema50: float | None = None,
    macd_value: float = 0.0,
    macd_signal: float = 0.0,
) -> TimeframeIndicators:
    """Shorthand to build TimeframeIndicators."""
    return TimeframeIndicators(
        ema9=ema9,
        ema21=ema21,
        rsi14=rsi14,
        macd=MACDData(
            value=macd_value or macd_hist,
            signal=macd_signal,
            histogram=macd_hist,
        ),
        atr14=atr14,
        ema50=ema50,
    )


def _bullish_input() -> AnalyzerInput:
    """Create a strongly bullish scenario."""
    return AnalyzerInput(
        ticker="AAPL",
        current_price=185.00,
        indicators={
            "1h": _make_indicators(
                ema9=185.50, ema21=184.00, rsi14=62.0, macd_hist=0.45,
                atr14=1.20, ema50=182.00, macd_value=0.50, macd_signal=0.05,
            ),
            "15m": _make_indicators(
                ema9=185.20, ema21=184.80, rsi14=58.0, macd_hist=0.15,
                atr14=0.60, macd_value=0.20, macd_signal=0.05,
            ),
            "1m": _make_indicators(
                ema9=185.05, ema21=184.95, rsi14=42.0, macd_hist=0.02,
                atr14=0.10, macd_value=0.03, macd_signal=0.01,
            ),
        },
        price_summary=PriceSummary(
            open=183.50, high=185.80, low=183.20, close=185.00,
            volume=52_000_000, vwap=184.50, avg_volume=40_000_000,
            volatility_percentile=55.0,
        ),
        support_resistance=SupportResistance(
            supports=[183.00, 181.50, 180.00],
            resistances=[186.50, 188.00, 190.00],
        ),
        news_sentiment=NewsSentiment(
            impact="Low", sentiment="Neutral", headlines=["AAPL steady trading day"],
        ),
    )


def _bearish_input() -> AnalyzerInput:
    """Create a strongly bearish scenario."""
    return AnalyzerInput(
        ticker="TSLA",
        current_price=220.00,
        indicators={
            "1h": _make_indicators(
                ema9=219.00, ema21=221.50, rsi14=38.0, macd_hist=-0.80,
                atr14=2.50, ema50=223.00, macd_value=-0.90, macd_signal=-0.10,
            ),
            "15m": _make_indicators(
                ema9=219.50, ema21=220.80, rsi14=42.0, macd_hist=-0.30,
                atr14=1.20, macd_value=-0.35, macd_signal=-0.05,
            ),
            "1m": _make_indicators(
                ema9=219.90, ema21=220.10, rsi14=58.0, macd_hist=-0.05,
                atr14=0.25, macd_value=-0.06, macd_signal=-0.01,
            ),
        },
        price_summary=PriceSummary(
            open=222.00, high=222.50, low=219.00, close=220.00,
            volume=65_000_000, vwap=220.80, avg_volume=50_000_000,
            volatility_percentile=70.0,
        ),
        support_resistance=SupportResistance(
            supports=[218.00, 215.00, 212.00],
            resistances=[222.50, 225.00, 228.00],
        ),
        news_sentiment=NewsSentiment(
            impact="Low", sentiment="Neutral", headlines=["TSLA normal trading"],
        ),
    )


def _neutral_input() -> AnalyzerInput:
    """Create a neutral / choppy scenario."""
    return AnalyzerInput(
        ticker="SPY",
        current_price=450.00,
        indicators={
            "1h": _make_indicators(
                ema9=450.00, ema21=450.00, rsi14=50.0, macd_hist=0.00,
                atr14=2.00,
            ),
            "15m": _make_indicators(
                ema9=450.05, ema21=449.95, rsi14=50.0, macd_hist=0.00,
                atr14=0.80,
            ),
            "1m": _make_indicators(
                ema9=450.00, ema21=450.00, rsi14=50.0, macd_hist=0.00,
                atr14=0.15,
            ),
        },
        price_summary=PriceSummary(
            open=449.50, high=450.50, low=449.00, close=450.00,
            volume=30_000_000, vwap=450.00,
        ),
        support_resistance=SupportResistance(
            supports=[449.00, 448.00],
            resistances=[451.00, 452.00],
        ),
        news_sentiment=NewsSentiment(
            impact="None", sentiment="Neutral", headlines=[],
        ),
    )


@pytest.fixture
def analyzer():
    return IntradayAnalyzer()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestHourlyTrend:
    """Rule 1: Trend assessment on the 1h timeframe."""

    def test_bullish_trend(self, analyzer):
        inp = _bullish_input()
        signal = analyzer.analyze(inp)
        assert signal.meta.hourly_trend == "Bullish"

    def test_bearish_trend(self, analyzer):
        inp = _bearish_input()
        signal = analyzer.analyze(inp)
        assert signal.meta.hourly_trend == "Bearish"

    def test_neutral_trend_results_in_hold(self, analyzer):
        inp = _neutral_input()
        signal = analyzer.analyze(inp)
        assert signal.meta.hourly_trend == "Neutral"
        assert signal.action == "Hold"
        assert any("neutral" in r.lower() or "unclear" in r.lower() for r in signal.reasons)


class TestFifteenMinuteConfirmation:
    """Rule 2: 15m setup confirmation."""

    def test_bullish_confirmed(self, analyzer):
        inp = _bullish_input()
        signal = analyzer.analyze(inp)
        # 15m should confirm bullish
        assert "15m" in signal.meta.fifteen_min_setup.lower() or "ema" in signal.meta.fifteen_min_setup.lower()

    def test_bearish_not_confirmed_on_15m_gives_hold(self, analyzer):
        inp = _bearish_input()
        # Make 15m indicators bullish (contradicting the hourly bearish trend)
        inp.indicators["15m"] = _make_indicators(
            ema9=221.50, ema21=220.00, rsi14=65.0, macd_hist=0.30,
            atr14=1.20, macd_value=0.35, macd_signal=0.05,
        )
        signal = analyzer.analyze(inp)
        assert signal.action == "Hold"
        assert any("15m" in r.lower() or "not confirmed" in r.lower() for r in signal.reasons)


class TestConfidenceAndRiskReward:
    """Rule 3: Confidence >= 60 and R:R >= 1.5."""

    def test_bullish_signal_meets_thresholds(self, analyzer):
        inp = _bullish_input()
        signal = analyzer.analyze(inp)
        if signal.action == "Buy":
            assert signal.confidence >= 60
            assert signal.risk_reward is not None
            assert signal.risk_reward >= 1.5

    def test_low_confidence_gives_hold(self, analyzer):
        inp = _bullish_input()
        # Weaken the signal: bad 15m and 1m
        inp.indicators["15m"] = _make_indicators(
            ema9=184.80, ema21=185.20, rsi14=48.0, macd_hist=-0.01,
            atr14=0.60, macd_value=-0.01, macd_signal=0.0,
        )
        signal = analyzer.analyze(inp)
        # Should be Hold since 15m doesn't confirm
        assert signal.action == "Hold"


class TestNewsImpact:
    """Rule 4: High-impact contradictory news defaults to Hold."""

    def test_high_impact_bearish_news_on_bullish_trend(self, analyzer):
        inp = _bullish_input()
        inp.news_sentiment = NewsSentiment(
            impact="High",
            sentiment="Bearish",
            headlines=["AAPL faces major lawsuit", "Analyst downgrades AAPL"],
        )
        signal = analyzer.analyze(inp)
        # Should be Hold since technicals are not extremely strong (< 85)
        assert signal.action == "Hold"
        assert signal.meta.news_sentiment_conflict is True

    def test_high_impact_bullish_news_on_bearish_trend(self, analyzer):
        inp = _bearish_input()
        inp.news_sentiment = NewsSentiment(
            impact="High",
            sentiment="Bullish",
            headlines=["TSLA beats earnings", "Major EV breakthrough"],
        )
        signal = analyzer.analyze(inp)
        assert signal.action == "Hold"
        assert signal.meta.news_sentiment_conflict is True

    def test_low_impact_news_does_not_block(self, analyzer):
        inp = _bullish_input()
        inp.news_sentiment = NewsSentiment(
            impact="Low", sentiment="Bearish", headlines=["Minor concern"],
        )
        signal = analyzer.analyze(inp)
        # Low impact should not block the trade
        assert signal.action in ("Buy", "Hold")  # depends on other factors


class TestEntryAndStopRules:
    """Rule 5: Entry proximity and stop distance."""

    def test_entry_close_to_current_price(self, analyzer):
        inp = _bullish_input()
        signal = analyzer.analyze(inp)
        if signal.action == "Buy":
            distance_pct = abs(signal.entry - inp.current_price) / inp.current_price * 100
            assert distance_pct <= 0.5

    def test_stop_respects_atr_minimum(self, analyzer):
        inp = _bullish_input()
        signal = analyzer.analyze(inp)
        if signal.action == "Buy":
            atr15 = inp.indicators["15m"].atr14
            min_stop = 0.5 * atr15
            actual_stop_distance = abs(signal.entry - signal.stop)
            assert actual_stop_distance >= min_stop - 0.01  # small float tolerance

    def test_sell_stop_respects_atr_minimum(self, analyzer):
        inp = _bearish_input()
        signal = analyzer.analyze(inp)
        if signal.action == "Sell":
            atr15 = inp.indicators["15m"].atr14
            min_stop = 0.5 * atr15
            actual_stop_distance = abs(signal.stop - signal.entry)
            assert actual_stop_distance >= min_stop - 0.01


class TestInsufficientData:
    """Rule 6: Incomplete data returns Hold."""

    def test_missing_timeframe(self, analyzer):
        inp = _bullish_input()
        del inp.indicators["1m"]
        signal = analyzer.analyze(inp)
        assert signal.action == "Hold"
        assert any("insufficient" in r.lower() for r in signal.reasons)

    def test_negative_atr(self, analyzer):
        inp = _bullish_input()
        inp.indicators["15m"].atr14 = -1.0
        signal = analyzer.analyze(inp)
        assert signal.action == "Hold"

    def test_invalid_rsi(self, analyzer):
        inp = _bullish_input()
        inp.indicators["1h"].rsi14 = 150.0
        signal = analyzer.analyze(inp)
        assert signal.action == "Hold"


class TestOutputSchema:
    """Verify the output conforms to the JSON schema."""

    def test_hold_output_schema(self, analyzer):
        inp = _neutral_input()
        signal = analyzer.analyze(inp)
        d = signal.to_dict()
        assert d["action"] == "Hold"
        assert isinstance(d["confidence"], int)
        assert isinstance(d["reasons"], list)
        assert len(d["reasons"]) >= 1
        assert isinstance(d["invalidations"], list)
        assert isinstance(d["meta"], dict)
        assert d["meta"]["hourly_trend"] in ("Bullish", "Bearish", "Neutral")
        assert d["meta"]["news_impact"] in ("High", "Medium", "Low", "None")
        # Hold signals should have null entry/stop/target
        assert d["entry"] is None
        assert d["stop"] is None
        assert d["target1"] is None

    def test_buy_output_schema(self, analyzer):
        inp = _bullish_input()
        signal = analyzer.analyze(inp)
        d = signal.to_dict()
        if d["action"] == "Buy":
            assert d["entry"] is not None
            assert d["stop"] is not None
            assert d["target1"] is not None
            assert d["risk_reward"] is not None
            assert d["risk_reward"] >= 1.5
            assert d["confidence"] >= 60

    def test_json_output_is_valid_json(self, analyzer):
        inp = _bullish_input()
        json_str = analyzer.analyze_json(inp)
        parsed = json.loads(json_str)
        assert "ticker" in parsed
        assert "action" in parsed
        assert "meta" in parsed

    def test_schema_validation_with_jsonschema(self, analyzer):
        """Validate output against the actual JSON schema file."""
        try:
            import jsonschema
        except ImportError:
            pytest.skip("jsonschema not installed")

        schema_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "schemas",
            "trade_signal_schema.json",
        )
        with open(schema_path) as f:
            schema = json.load(f)

        # Test with bullish input
        inp = _bullish_input()
        signal = analyzer.analyze(inp)
        d = signal.to_dict()
        jsonschema.validate(instance=d, schema=schema)

        # Test with neutral (hold) input
        inp = _neutral_input()
        signal = analyzer.analyze(inp)
        d = signal.to_dict()
        jsonschema.validate(instance=d, schema=schema)


class TestParseInput:
    """Test the parse_input function for JSON -> dataclass conversion."""

    def test_roundtrip(self):
        raw = {
            "ticker": "MSFT",
            "current_price": 420.00,
            "indicators": {
                "1m": {
                    "ema9": 420.10, "ema21": 419.90, "rsi14": 55.0,
                    "macd": {"value": 0.05, "signal": 0.02, "histogram": 0.03},
                    "atr14": 0.20,
                },
                "15m": {
                    "ema9": 420.50, "ema21": 419.50, "rsi14": 58.0,
                    "macd": {"value": 0.15, "signal": 0.05, "histogram": 0.10},
                    "atr14": 0.80,
                },
                "1h": {
                    "ema9": 421.00, "ema21": 419.00, "rsi14": 60.0,
                    "macd": {"value": 0.30, "signal": 0.10, "histogram": 0.20},
                    "atr14": 1.50, "ema50": 417.00,
                },
            },
            "price_summary": {
                "open": 418.00, "high": 421.00, "low": 417.50,
                "close": 420.00, "volume": 25_000_000, "vwap": 419.50,
            },
            "support_resistance": {
                "supports": [418.00, 416.00],
                "resistances": [422.00, 425.00],
            },
            "news_sentiment": {
                "impact": "Low", "sentiment": "Neutral", "headlines": [],
            },
        }
        inp = parse_input(raw)
        assert inp.ticker == "MSFT"
        assert inp.current_price == 420.00
        assert inp.indicators["1h"].ema50 == 417.00
        assert inp.indicators["15m"].macd.histogram == 0.10
        assert inp.news_sentiment.impact == "Low"


class TestBearishSignal:
    """Test that a strong bearish scenario generates a Sell signal."""

    def test_sell_signal(self, analyzer):
        inp = _bearish_input()
        signal = analyzer.analyze(inp)
        assert signal.action == "Sell"
        assert signal.confidence >= 60
        assert signal.risk_reward >= 1.5
        assert signal.entry is not None
        assert signal.stop is not None
        assert signal.stop > signal.entry  # stop above entry for short
        assert signal.target1 < signal.entry  # target below entry for short

    def test_sell_invalidations_populated(self, analyzer):
        inp = _bearish_input()
        signal = analyzer.analyze(inp)
        if signal.action == "Sell":
            assert len(signal.invalidations) >= 2
            assert any("stop" in inv.lower() for inv in signal.invalidations)


class TestBullishSignal:
    """Test that a strong bullish scenario generates a Buy signal."""

    def test_buy_signal(self, analyzer):
        inp = _bullish_input()
        signal = analyzer.analyze(inp)
        assert signal.action == "Buy"
        assert signal.confidence >= 60
        assert signal.risk_reward >= 1.5
        assert signal.entry is not None
        assert signal.stop is not None
        assert signal.stop < signal.entry  # stop below entry for long
        assert signal.target1 > signal.entry  # target above entry for long

    def test_buy_invalidations_populated(self, analyzer):
        inp = _bullish_input()
        signal = analyzer.analyze(inp)
        if signal.action == "Buy":
            assert len(signal.invalidations) >= 2
            assert any("stop" in inv.lower() for inv in signal.invalidations)


class TestPositionSizing:
    """Test position size calculation."""

    def test_position_size_capped_at_5(self, analyzer):
        inp = _bullish_input()
        signal = analyzer.analyze(inp)
        if signal.position_size_pct is not None:
            assert signal.position_size_pct <= 5.0

    def test_hold_has_no_position_size(self, analyzer):
        inp = _neutral_input()
        signal = analyzer.analyze(inp)
        assert signal.position_size_pct is None
