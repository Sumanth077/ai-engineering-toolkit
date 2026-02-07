# Intraday Trading Analyzer

A rule-based, multi-timeframe intraday trading signal engine. Accepts computed technical indicators across 1m/15m/1h timeframes, support/resistance levels, and news sentiment, then outputs a strict JSON trade signal (Buy / Sell / Hold) with entry, stop, targets, and reasoning.

## Trading Rules

| # | Rule | Detail |
|---|------|--------|
| 1 | **Trade with the 1h trend** | If the hourly trend is unclear, output Hold. |
| 2 | **15m confirms, 1m times** | 15m setup must align with 1h direction; 1m refines entry. |
| 3 | **Confidence >= 60, R:R >= 1.5** | Both thresholds must be met or the signal is Hold. |
| 4 | **High-impact news override** | Contradictory high-impact news defaults to Hold unless technicals score >= 85/100. |
| 5 | **Realistic levels** | Entry within ~0.5% of current price; stop >= 0.5 * ATR(14) on the 15m chart. |
| 6 | **Data quality gate** | Incomplete or inconsistent input returns Hold with "Insufficient data". |

## Quick Start

```bash
# Install (Python >= 3.10)
pip install -e ".[dev,ui]"

# Launch the web UI
streamlit run app.py

# Or run via CLI (reads JSON from stdin)
cat examples/bullish_aapl.json | intraday-analyzer

# Or pass a file path
intraday-analyzer examples/bullish_aapl.json

# Run tests
pytest
```

### Web UI

The Streamlit dashboard provides three input modes:

- **Interactive Form** -- Fill in ticker, indicators, S/R levels, and news via form fields and tabs
- **JSON Editor** -- Paste or edit raw JSON directly
- **Load Example** -- Pick a bundled example file (AAPL bullish, TSLA bearish)

The UI displays the signal action (color-coded), confidence, R:R, trade levels, multi-timeframe summary, reasons, invalidations, and the raw JSON output.

## Project Structure

```
app.py                            # Streamlit web UI
intraday_analyzer/
    __init__.py        # Package exports
    models.py          # Dataclasses for input/output + parse_input()
    analyzer.py        # Core IntradayAnalyzer engine
    cli.py             # CLI entry point
schemas/
    analyzer_input_schema.json    # JSON Schema for input
    trade_signal_schema.json      # JSON Schema for output
examples/
    bullish_aapl.json             # Sample bullish input
    bearish_tsla.json             # Sample bearish input
tests/
    test_analyzer.py              # 27 unit tests
```

## Input Format

Provide a JSON object with:

| Field | Description |
|-------|-------------|
| `ticker` | Instrument symbol |
| `current_price` | Latest price |
| `indicators` | Object with keys `1m`, `15m`, `1h`, each containing EMA9, EMA21, RSI14, MACD, ATR14 (plus optional EMA50, Bollinger Bands, volume SMA) |
| `price_summary` | OHLCV + VWAP (optional: avg_volume, volatility_percentile) |
| `support_resistance` | `supports[]` and `resistances[]` price levels |
| `news_sentiment` | `impact` (High/Medium/Low/None), `sentiment` (Bullish/Bearish/Neutral), `headlines[]` |

See `schemas/analyzer_input_schema.json` for the full specification.

## Output Format

```json
{
  "ticker": "AAPL",
  "timestamp": "2026-02-07T12:30:00+00:00",
  "action": "Buy",
  "confidence": 72,
  "entry": 185.00,
  "stop": 183.00,
  "target1": 188.00,
  "target2": 190.00,
  "risk_reward": 1.50,
  "position_size_pct": 3.6,
  "reasons": [
    "1h trend Bullish: EMA9=185.50 vs EMA21=184.00, RSI=62.0",
    "15m confirms: EMA9 > EMA21; MACD histogram positive",
    "Price above VWAP (184.50)"
  ],
  "invalidations": [
    "Price closes below stop 183.00",
    "1h EMA9 crosses below EMA21",
    "Unexpected high-impact news release"
  ],
  "meta": {
    "hourly_trend": "Bullish",
    "fifteen_min_setup": "15m EMA9 > EMA21; 15m MACD histogram positive; 15m RSI 58.0 in bullish range",
    "one_min_timing": "1m RSI 42.0 shows pullback (good entry); 1m MACD histogram turning positive",
    "news_impact": "Low",
    "news_sentiment_conflict": false
  }
}
```

See `schemas/trade_signal_schema.json` for the full specification.

## Python API

```python
from intraday_analyzer import IntradayAnalyzer
from intraday_analyzer.models import parse_input

# From a dict matching the input schema
data = { ... }
inp = parse_input(data)

analyzer = IntradayAnalyzer()
signal = analyzer.analyze(inp)       # returns TradeSignal dataclass
json_str = analyzer.analyze_json(inp)  # returns strict JSON string
```

## How It Works

1. **Hourly trend scoring** -- EMA9/21/50 alignment, MACD histogram direction, RSI position relative to 50, price vs VWAP. Requires a clear majority (score >= 2 or <= -2) to declare Bullish/Bearish.

2. **15-minute confirmation** -- Checks EMA cross direction, MACD histogram sign, and RSI range alignment with the hourly trend. Requires 2 of 3 to confirm.

3. **1-minute timing** -- Looks for pullback entries (RSI dip on bullish, RSI bounce on bearish) and MACD histogram turn.

4. **Level computation** -- Entry at current price, stop at nearest structural S/R (floored at 0.5 * ATR), targets at subsequent S/R levels or risk multiples.

5. **Confidence scoring** -- Composite of trend clarity (25 pts), 15m setup quality (24 pts), 1m timing (20 pts), R:R bonus (15 pts), volume (10 pts), VWAP alignment (5 pts). Capped at 100.

6. **News filter** -- High-impact contradictory sentiment blocks the trade unless the technical strength score exceeds 85.

## License

MIT
