"""Core intraday trading analyzer engine.

Implements a multi-timeframe analysis approach:
  - 1h: Determine the primary trend direction
  - 15m: Confirm the setup aligns with the trend
  - 1m: Fine-tune entry timing

Trading rules enforced:
  1. Prefer trading in the direction of the 1h trend; Hold if unclear.
  2. Use 15m for setup confirmation and 1m for timing.
  3. Only Buy/Sell if confidence >= 60 and R:R >= 1.5.
  4. High-impact news that contradicts the trade direction defaults to Hold
     unless technicals are extremely strong (with explanation).
  5. Entry must be within ~0.5% of current price for liquid large caps.
     Stop must not be tighter than 0.5 * ATR(14) of the 15m timeframe.
  6. Incomplete or inconsistent data returns Hold with reason.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Optional

from intraday_analyzer.models import (
    AnalyzerInput,
    SignalMeta,
    TradeSignal,
)


# Thresholds
_CONFIDENCE_THRESHOLD = 60
_MIN_RISK_REWARD = 1.5
_ENTRY_TOLERANCE_PCT = 0.5  # percent
_MIN_STOP_ATR_FACTOR = 0.5
_RSI_OVERBOUGHT = 70
_RSI_OVERSOLD = 30
_RSI_EXTREME_OVERBOUGHT = 80
_RSI_EXTREME_OVERSOLD = 20


class IntradayAnalyzer:
    """Multi-timeframe intraday trading analyzer."""

    def __init__(self, schema_dir: Optional[str] = None):
        self._schema_dir = schema_dir or os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "schemas"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, inp: AnalyzerInput) -> TradeSignal:
        """Produce a trade signal from the given analyzer input."""
        # Rule 6: validate input data completeness
        errors = inp.validate()
        if errors:
            return self._hold_signal(
                inp.ticker,
                reasons=["Insufficient data: " + "; ".join(errors)],
                meta=self._default_meta(inp),
            )

        hourly_trend = self._assess_hourly_trend(inp)
        fifteen_setup = self._assess_15m_setup(inp, hourly_trend)
        one_min_timing = self._assess_1m_timing(inp, hourly_trend)
        news_conflict = self._check_news_conflict(inp, hourly_trend)

        meta = SignalMeta(
            hourly_trend=hourly_trend,
            fifteen_min_setup=fifteen_setup["summary"],
            one_min_timing=one_min_timing["summary"],
            news_impact=inp.news_sentiment.impact,
            news_sentiment_conflict=news_conflict,
        )

        # Rule 1: If hourly trend is unclear, be conservative
        if hourly_trend == "Neutral":
            return self._hold_signal(
                inp.ticker,
                reasons=["1h trend is neutral/unclear — staying flat per rule 1"],
                meta=meta,
            )

        # Rule 2: Need 15m confirmation
        if not fifteen_setup["confirmed"]:
            return self._hold_signal(
                inp.ticker,
                reasons=[
                    f"1h trend is {hourly_trend} but 15m setup not confirmed",
                    fifteen_setup["summary"],
                ],
                meta=meta,
            )

        # Rule 4: High-impact news contradicting direction
        if news_conflict and inp.news_sentiment.impact == "High":
            technicals_strength = self._technicals_strength_score(inp, hourly_trend)
            if technicals_strength < 85:
                return self._hold_signal(
                    inp.ticker,
                    reasons=[
                        f"High-impact news sentiment ({inp.news_sentiment.sentiment}) "
                        f"contradicts {hourly_trend} trend",
                        "Technicals not strong enough to override "
                        f"(strength={technicals_strength}/100)",
                    ],
                    meta=meta,
                )
            # Technicals extremely strong — proceed but note it
            override_reason = (
                f"Overriding high-impact {inp.news_sentiment.sentiment} news: "
                f"technical strength {technicals_strength}/100"
            )
        else:
            override_reason = None

        # Determine direction
        direction = "Buy" if hourly_trend == "Bullish" else "Sell"

        # Compute entry, stop, targets
        entry, stop, target1, target2 = self._compute_levels(inp, direction)

        # Rule 5: entry proximity check
        entry_distance_pct = abs(entry - inp.current_price) / inp.current_price * 100
        if entry_distance_pct > _ENTRY_TOLERANCE_PCT:
            return self._hold_signal(
                inp.ticker,
                reasons=[
                    f"Computed entry {entry:.2f} is {entry_distance_pct:.2f}% from "
                    f"current price {inp.current_price:.2f} (max {_ENTRY_TOLERANCE_PCT}%)",
                ],
                meta=meta,
            )

        # Rule 5: stop not tighter than 0.5 * ATR14 of 15m
        atr15 = inp.indicators["15m"].atr14
        min_stop_distance = _MIN_STOP_ATR_FACTOR * atr15
        actual_stop_distance = abs(entry - stop)
        if actual_stop_distance < min_stop_distance:
            # Widen the stop to minimum
            if direction == "Buy":
                stop = entry - min_stop_distance
            else:
                stop = entry + min_stop_distance
            actual_stop_distance = min_stop_distance

        # Rule 3: risk/reward check
        risk = abs(entry - stop)
        reward = abs(target1 - entry)
        risk_reward = round(reward / risk, 2) if risk > 0 else 0.0

        # Build confidence score
        confidence = self._compute_confidence(
            inp, hourly_trend, fifteen_setup, one_min_timing, risk_reward
        )

        # Rule 3: thresholds
        if confidence < _CONFIDENCE_THRESHOLD:
            return self._hold_signal(
                inp.ticker,
                reasons=[
                    f"Confidence {confidence} < {_CONFIDENCE_THRESHOLD} threshold",
                    f"Direction was {direction} but signal not strong enough",
                ],
                meta=meta,
            )

        if risk_reward < _MIN_RISK_REWARD:
            return self._hold_signal(
                inp.ticker,
                reasons=[
                    f"Risk/Reward {risk_reward} < {_MIN_RISK_REWARD} minimum",
                    f"Entry={entry:.2f}, Stop={stop:.2f}, Target1={target1:.2f}",
                ],
                meta=meta,
            )

        # Build reasons
        reasons = self._build_reasons(
            inp, direction, hourly_trend, fifteen_setup, one_min_timing
        )
        if override_reason:
            reasons.append(override_reason)

        # Build invalidations
        invalidations = self._build_invalidations(
            inp, direction, hourly_trend, stop, entry
        )

        # Position sizing (simplified: scale with confidence, cap at 5%)
        position_size_pct = round(min(5.0, confidence / 20.0), 1)

        return TradeSignal(
            ticker=inp.ticker,
            timestamp=datetime.now(timezone.utc).isoformat(),
            action=direction,
            confidence=confidence,
            entry=round(entry, 2),
            stop=round(stop, 2),
            target1=round(target1, 2),
            target2=round(target2, 2) if target2 is not None else None,
            risk_reward=risk_reward,
            position_size_pct=position_size_pct,
            reasons=reasons,
            invalidations=invalidations,
            meta=meta,
        )

    def analyze_json(self, inp: AnalyzerInput) -> str:
        """Produce a strict JSON string (no markdown wrapping)."""
        signal = self.analyze(inp)
        return json.dumps(signal.to_dict(), indent=2)

    # ------------------------------------------------------------------
    # Trend / Setup / Timing assessments
    # ------------------------------------------------------------------

    def _assess_hourly_trend(self, inp: AnalyzerInput) -> str:
        """Determine the 1h trend: Bullish, Bearish, or Neutral."""
        h = inp.indicators["1h"]
        signals = []

        # EMA alignment
        if h.ema9 > h.ema21:
            signals.append(1)
        elif h.ema9 < h.ema21:
            signals.append(-1)
        else:
            signals.append(0)

        # EMA50 if available
        if h.ema50 is not None:
            if h.ema9 > h.ema50 and h.ema21 > h.ema50:
                signals.append(1)
            elif h.ema9 < h.ema50 and h.ema21 < h.ema50:
                signals.append(-1)
            else:
                signals.append(0)

        # MACD
        if h.macd.histogram > 0 and h.macd.value > h.macd.signal:
            signals.append(1)
        elif h.macd.histogram < 0 and h.macd.value < h.macd.signal:
            signals.append(-1)
        else:
            signals.append(0)

        # RSI positioning
        if h.rsi14 > 50:
            signals.append(1)
        elif h.rsi14 < 50:
            signals.append(-1)
        else:
            signals.append(0)

        # Price vs VWAP
        if inp.current_price > inp.price_summary.vwap:
            signals.append(1)
        elif inp.current_price < inp.price_summary.vwap:
            signals.append(-1)
        else:
            signals.append(0)

        score = sum(signals)
        total = len(signals)

        # Require a meaningful majority
        if score >= 2:
            return "Bullish"
        elif score <= -2:
            return "Bearish"
        return "Neutral"

    def _assess_15m_setup(
        self, inp: AnalyzerInput, hourly_trend: str
    ) -> dict:
        """Check if the 15m timeframe confirms the hourly trend direction."""
        m15 = inp.indicators["15m"]
        confirms = []
        details = []

        if hourly_trend == "Bullish":
            if m15.ema9 > m15.ema21:
                confirms.append(True)
                details.append("15m EMA9 > EMA21")
            else:
                confirms.append(False)
                details.append("15m EMA9 < EMA21 (bearish cross)")

            if m15.macd.histogram > 0:
                confirms.append(True)
                details.append("15m MACD histogram positive")
            else:
                confirms.append(False)
                details.append("15m MACD histogram negative")

            if m15.rsi14 > 40 and m15.rsi14 < _RSI_OVERBOUGHT:
                confirms.append(True)
                details.append(f"15m RSI {m15.rsi14:.1f} in bullish range")
            else:
                confirms.append(False)
                details.append(f"15m RSI {m15.rsi14:.1f} outside bullish range")

        elif hourly_trend == "Bearish":
            if m15.ema9 < m15.ema21:
                confirms.append(True)
                details.append("15m EMA9 < EMA21")
            else:
                confirms.append(False)
                details.append("15m EMA9 > EMA21 (bullish cross)")

            if m15.macd.histogram < 0:
                confirms.append(True)
                details.append("15m MACD histogram negative")
            else:
                confirms.append(False)
                details.append("15m MACD histogram positive")

            if m15.rsi14 < 60 and m15.rsi14 > _RSI_OVERSOLD:
                confirms.append(True)
                details.append(f"15m RSI {m15.rsi14:.1f} in bearish range")
            else:
                confirms.append(False)
                details.append(f"15m RSI {m15.rsi14:.1f} outside bearish range")

        # Need at least 2/3 confirmations
        confirmed = sum(confirms) >= 2
        summary = "; ".join(details)
        return {"confirmed": confirmed, "summary": summary, "score": sum(confirms)}

    def _assess_1m_timing(
        self, inp: AnalyzerInput, hourly_trend: str
    ) -> dict:
        """Check the 1m chart for entry timing signals."""
        m1 = inp.indicators["1m"]
        signals = []
        details = []

        if hourly_trend == "Bullish":
            # Look for pullback-then-go on 1m
            if m1.rsi14 < 45:
                signals.append(True)
                details.append(f"1m RSI {m1.rsi14:.1f} shows pullback (good entry)")
            elif m1.rsi14 > _RSI_OVERBOUGHT:
                signals.append(False)
                details.append(f"1m RSI {m1.rsi14:.1f} overbought (wait for pullback)")
            else:
                signals.append(True)
                details.append(f"1m RSI {m1.rsi14:.1f} acceptable")

            if m1.macd.histogram > 0:
                signals.append(True)
                details.append("1m MACD histogram turning positive")
            else:
                signals.append(False)
                details.append("1m MACD histogram still negative")

        elif hourly_trend == "Bearish":
            if m1.rsi14 > 55:
                signals.append(True)
                details.append(f"1m RSI {m1.rsi14:.1f} shows bounce (good short entry)")
            elif m1.rsi14 < _RSI_OVERSOLD:
                signals.append(False)
                details.append(f"1m RSI {m1.rsi14:.1f} oversold (wait for bounce)")
            else:
                signals.append(True)
                details.append(f"1m RSI {m1.rsi14:.1f} acceptable")

            if m1.macd.histogram < 0:
                signals.append(True)
                details.append("1m MACD histogram negative")
            else:
                signals.append(False)
                details.append("1m MACD histogram still positive")

        favorable = sum(signals)
        summary = "; ".join(details)
        return {"favorable": favorable >= 1, "summary": summary, "score": favorable}

    # ------------------------------------------------------------------
    # News conflict detection
    # ------------------------------------------------------------------

    def _check_news_conflict(self, inp: AnalyzerInput, hourly_trend: str) -> bool:
        """Return True if news sentiment contradicts the hourly trend."""
        sentiment = inp.news_sentiment.sentiment
        if hourly_trend == "Bullish" and sentiment == "Bearish":
            return True
        if hourly_trend == "Bearish" and sentiment == "Bullish":
            return True
        return False

    # ------------------------------------------------------------------
    # Level computations
    # ------------------------------------------------------------------

    def _compute_levels(
        self, inp: AnalyzerInput, direction: str
    ) -> tuple[float, float, float, Optional[float]]:
        """Compute entry, stop, target1, target2 based on direction and S/R."""
        atr15 = inp.indicators["15m"].atr14
        price = inp.current_price
        supports = sorted(inp.support_resistance.supports, reverse=True)
        resistances = sorted(inp.support_resistance.resistances)

        if direction == "Buy":
            entry = price
            # Stop at nearest support below entry, at least 0.5 * ATR
            stop = self._find_nearest_below(supports, entry)
            if stop is None or (entry - stop) < _MIN_STOP_ATR_FACTOR * atr15:
                stop = entry - max(1.0 * atr15, _MIN_STOP_ATR_FACTOR * atr15)

            risk = entry - stop
            # Target1: nearest resistance above, or 1.5R
            t1 = self._find_nearest_above(resistances, entry)
            if t1 is None or (t1 - entry) < _MIN_RISK_REWARD * risk:
                t1 = entry + _MIN_RISK_REWARD * risk

            # Target2: next resistance above target1, or 2.5R
            t2_candidates = [r for r in resistances if r > t1]
            t2 = t2_candidates[0] if t2_candidates else entry + 2.5 * risk

        else:  # Sell
            entry = price
            # Stop at nearest resistance above entry
            stop = self._find_nearest_above(resistances, entry)
            if stop is None or (stop - entry) < _MIN_STOP_ATR_FACTOR * atr15:
                stop = entry + max(1.0 * atr15, _MIN_STOP_ATR_FACTOR * atr15)

            risk = stop - entry
            # Target1: nearest support below, or 1.5R
            t1 = self._find_nearest_below(supports, entry)
            if t1 is None or (entry - t1) < _MIN_RISK_REWARD * risk:
                t1 = entry - _MIN_RISK_REWARD * risk

            # Target2
            t2_candidates = [s for s in supports if s < t1]
            t2 = t2_candidates[0] if t2_candidates else entry - 2.5 * risk

        return entry, stop, t1, t2

    @staticmethod
    def _find_nearest_below(levels: list[float], price: float) -> Optional[float]:
        """Find the nearest level strictly below price."""
        candidates = [lv for lv in levels if lv < price]
        return max(candidates) if candidates else None

    @staticmethod
    def _find_nearest_above(levels: list[float], price: float) -> Optional[float]:
        """Find the nearest level strictly above price."""
        candidates = [lv for lv in levels if lv > price]
        return min(candidates) if candidates else None

    # ------------------------------------------------------------------
    # Confidence scoring
    # ------------------------------------------------------------------

    def _compute_confidence(
        self,
        inp: AnalyzerInput,
        hourly_trend: str,
        fifteen_setup: dict,
        one_min_timing: dict,
        risk_reward: float,
    ) -> int:
        """Compute a confidence score 0-100."""
        score = 0

        # Hourly trend clarity (max 25)
        h = inp.indicators["1h"]
        if hourly_trend == "Bullish":
            ema_spread = (h.ema9 - h.ema21) / h.ema21 * 100 if h.ema21 else 0
        else:
            ema_spread = (h.ema21 - h.ema9) / h.ema21 * 100 if h.ema21 else 0
        score += min(15, int(ema_spread * 10))
        if h.macd.histogram != 0:
            score += 5
        if (hourly_trend == "Bullish" and h.rsi14 > 55) or (
            hourly_trend == "Bearish" and h.rsi14 < 45
        ):
            score += 5

        # 15m setup (max 25)
        score += fifteen_setup["score"] * 8  # up to 24

        # 1m timing (max 20)
        score += one_min_timing["score"] * 10

        # R:R bonus (max 15)
        if risk_reward >= 3.0:
            score += 15
        elif risk_reward >= 2.0:
            score += 10
        elif risk_reward >= 1.5:
            score += 5

        # Volume confirmation (max 10)
        ps = inp.price_summary
        if ps.avg_volume and ps.volume > ps.avg_volume:
            vol_ratio = ps.volume / ps.avg_volume
            score += min(10, int(vol_ratio * 5))

        # VWAP alignment (max 5)
        if hourly_trend == "Bullish" and inp.current_price > ps.vwap:
            score += 5
        elif hourly_trend == "Bearish" and inp.current_price < ps.vwap:
            score += 5

        return min(100, max(0, score))

    def _technicals_strength_score(
        self, inp: AnalyzerInput, hourly_trend: str
    ) -> int:
        """Score how strong technicals are (0-100) to decide if we override news."""
        score = 0
        h = inp.indicators["1h"]
        m15 = inp.indicators["15m"]

        # Strong EMA alignment across timeframes
        if hourly_trend == "Bullish":
            if h.ema9 > h.ema21:
                score += 20
            if h.ema50 and h.ema9 > h.ema50:
                score += 10
            if m15.ema9 > m15.ema21:
                score += 15
            if h.rsi14 > 60:
                score += 10
            if h.rsi14 > _RSI_OVERBOUGHT:
                score += 5  # extreme
            if h.macd.histogram > 0:
                score += 10
            if m15.macd.histogram > 0:
                score += 10
        else:
            if h.ema9 < h.ema21:
                score += 20
            if h.ema50 and h.ema9 < h.ema50:
                score += 10
            if m15.ema9 < m15.ema21:
                score += 15
            if h.rsi14 < 40:
                score += 10
            if h.rsi14 < _RSI_OVERSOLD:
                score += 5
            if h.macd.histogram < 0:
                score += 10
            if m15.macd.histogram < 0:
                score += 10

        # Volume surge
        ps = inp.price_summary
        if ps.avg_volume and ps.volume > 1.5 * ps.avg_volume:
            score += 10

        # Price beyond all S/R
        if hourly_trend == "Bullish" and inp.support_resistance.resistances:
            if inp.current_price > max(inp.support_resistance.resistances):
                score += 10
        elif hourly_trend == "Bearish" and inp.support_resistance.supports:
            if inp.current_price < min(inp.support_resistance.supports):
                score += 10

        return min(100, score)

    # ------------------------------------------------------------------
    # Reason / invalidation builders
    # ------------------------------------------------------------------

    def _build_reasons(
        self,
        inp: AnalyzerInput,
        direction: str,
        hourly_trend: str,
        fifteen_setup: dict,
        one_min_timing: dict,
    ) -> list[str]:
        """Build human-readable reasons for the trade."""
        h = inp.indicators["1h"]
        reasons = [
            f"1h trend {hourly_trend}: EMA9={h.ema9:.2f} vs EMA21={h.ema21:.2f}, "
            f"RSI={h.rsi14:.1f}, MACD hist={h.macd.histogram:.4f}",
            f"15m confirms: {fifteen_setup['summary']}",
            f"1m timing: {one_min_timing['summary']}",
        ]

        ps = inp.price_summary
        if direction == "Buy" and inp.current_price > ps.vwap:
            reasons.append(f"Price above VWAP ({ps.vwap:.2f})")
        elif direction == "Sell" and inp.current_price < ps.vwap:
            reasons.append(f"Price below VWAP ({ps.vwap:.2f})")

        if ps.avg_volume and ps.volume > ps.avg_volume:
            reasons.append(
                f"Volume {ps.volume:,} above average {ps.avg_volume:,} "
                f"({ps.volume / ps.avg_volume:.1f}x)"
            )

        return reasons

    def _build_invalidations(
        self,
        inp: AnalyzerInput,
        direction: str,
        hourly_trend: str,
        stop: float,
        entry: float,
    ) -> list[str]:
        """Build invalidation conditions."""
        h = inp.indicators["1h"]
        invalidations = []

        if direction == "Buy":
            invalidations.append(f"Price closes below stop {stop:.2f}")
            invalidations.append(
                f"1h EMA9 crosses below EMA21 (currently {h.ema9:.2f} vs {h.ema21:.2f})"
            )
            invalidations.append("1h MACD histogram turns negative")
            if inp.support_resistance.supports:
                nearest_support = max(
                    s for s in inp.support_resistance.supports if s < entry
                ) if any(s < entry for s in inp.support_resistance.supports) else None
                if nearest_support:
                    invalidations.append(
                        f"Break below key support at {nearest_support:.2f}"
                    )
        else:
            invalidations.append(f"Price closes above stop {stop:.2f}")
            invalidations.append(
                f"1h EMA9 crosses above EMA21 (currently {h.ema9:.2f} vs {h.ema21:.2f})"
            )
            invalidations.append("1h MACD histogram turns positive")
            if inp.support_resistance.resistances:
                nearest_resistance = min(
                    r for r in inp.support_resistance.resistances if r > entry
                ) if any(r > entry for r in inp.support_resistance.resistances) else None
                if nearest_resistance:
                    invalidations.append(
                        f"Break above key resistance at {nearest_resistance:.2f}"
                    )

        invalidations.append("Unexpected high-impact news release")
        return invalidations

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _hold_signal(
        self,
        ticker: str,
        reasons: list[str],
        meta: SignalMeta,
    ) -> TradeSignal:
        """Produce a Hold signal."""
        return TradeSignal(
            ticker=ticker,
            timestamp=datetime.now(timezone.utc).isoformat(),
            action="Hold",
            confidence=0,
            reasons=reasons,
            invalidations=[],
            meta=meta,
        )

    def _default_meta(self, inp: AnalyzerInput) -> SignalMeta:
        return SignalMeta(
            hourly_trend="Neutral",
            fifteen_min_setup="N/A",
            one_min_timing="N/A",
            news_impact=inp.news_sentiment.impact,
            news_sentiment_conflict=False,
        )
