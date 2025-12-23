# Watchlist Comparison Page - Technical Documentation

This document provides a comprehensive overview of how the Watchlist Comparison page calculates projections, the Conviction Engine scoring system, and all underlying assumptions.

---

## Table of Contents

1. [Overview](#overview)
2. [Multi-Watchlist System](#multi-watchlist-system)
3. [Comparison Table Calculations](#comparison-table-calculations)
4. [Conviction Engine](#conviction-engine)
5. [Time-Alignment Analysis](#time-alignment-analysis)
6. [Risk Profiling](#risk-profiling)
7. [Optimism Bias Detection](#optimism-bias-detection)
8. [Price/Sales Fallback for Negative EPS](#pricesales-fallback-for-negative-eps)
9. [Assumptions & Limitations](#assumptions--limitations)

---

## Overview

The Watchlist Comparison page allows you to compare multiple stocks side-by-side across Bear, Base, and Bull scenarios. It includes:

1. **Conviction Engine Summary** — Weighted analysis with ratings and risk profiles
2. **5-Year Price Targets Table** — Detailed projections for each scenario
3. **Warnings & Flags** — Alerts for potential issues with projections

---

## Multi-Watchlist System

### Configuration

Watchlists are defined in `watchlists.py`:

```python
WATCHLISTS = {
    "watchlist_0": ["ADBE", "AMD", "AMZN", ...],
    "watchlist_1": ["PLTR", "NVDA", "HOOD", "TTWO"],
    # Add more watchlists as needed
}

DEFAULT_WATCHLIST = "watchlist_0"
```

### Adding New Watchlists

To add a new watchlist, edit `watchlists.py`:

```python
WATCHLISTS = {
    "watchlist_0": [...],
    "watchlist_1": [...],
    "my_new_watchlist": ["AAPL", "GOOGL", "MSFT"],  # Add here
}
```

### Session Persistence

The selected watchlist is stored in `st.session_state.selected_watchlist` and persists across page navigation within the same session.

---

## Comparison Table Calculations

### Scenario Defaults

For each stock, three scenarios are calculated using these default assumptions:

| Scenario | Revenue Growth | Net Margin | P/E Adjustment |
|----------|---------------|------------|----------------|
| **Bear** | Current - 5% | Current - 3% | P/E - 2 |
| **Base** | Current | Current | P/E (as-is) |
| **Bull** | Current + 5% | Current + 3% | P/E + 2 |

### 5-Year Projection Formula

For each scenario, Year 5 projections are calculated:

```
Year 5 Revenue = Current Revenue × (1 + Growth Rate)^5
Year 5 Net Income = Year 5 Revenue × Net Margin
Year 5 EPS = Year 5 Net Income / Shares Outstanding
Year 5 Price = Year 5 EPS × P/E Ratio
```

### CAGR Calculation

```
CAGR = ((Year 5 Price / Current Price)^(1/5) - 1) × 100
```

Each scenario produces two CAGR values:
- **CAGR Low** — Using P/E Low estimate
- **CAGR High** — Using P/E High estimate

---

## Conviction Engine

The Conviction Engine transforms raw projections into actionable ratings by weighting scenarios and analyzing consistency.

### Weighted CAGR Calculation

The expected CAGR is calculated using probability-weighted scenario outcomes:

```
Expected CAGR = (Bear Weight × Bear CAGR) + (Base Weight × Base CAGR) + (Bull Weight × Bull CAGR)
```

**Default Weights:**
| Scenario | Probability Weight |
|----------|-------------------|
| Bear | 25% |
| Base | 50% |
| Bull | 25% |

**CAGR for each scenario uses the midpoint:**
```
Scenario CAGR = (CAGR Low + CAGR High) / 2
```

### Expected 5-Year Price

```
Expected 5Y Price = Current Price × (1 + Expected CAGR / 100)^5
```

### Conviction Rating

The conviction rating compares the expected CAGR to a **hurdle rate** (default: 10%, representing index fund benchmark returns):

| Rating | Condition | Interpretation |
|--------|-----------|----------------|
| 🟢 **Strong Buy** | CAGR ≥ Hurdle + 10% | Significantly beats market |
| 🟡 **Buy** | CAGR ≥ Hurdle | Beats market benchmark |
| 🟠 **Hold** | CAGR ≥ Hurdle - 5% | Near market returns |
| 🔴 **Avoid** | CAGR ≥ 0% | Positive but below benchmark |
| ⛔ **Strong Avoid** | CAGR < 0% | Expected capital loss |

### Hurdle Rate Adjustment

Users can adjust the hurdle rate in the UI. Common values:
- **10%** — S&P 500 historical average
- **7%** — Conservative long-term expectation
- **15%** — Aggressive growth target

---

## Time-Alignment Analysis

Time-alignment checks whether short-term price action aligns with long-term projections. This helps identify potential value traps or accumulation opportunities.

### 1-Year Return Calculation

```python
hist = stock.history(period="1y")
one_year_return = ((current_price / year_ago_price) - 1) × 100
```

### Alignment Categories

| Status | Condition | Interpretation |
|--------|-----------|----------------|
| ✅ **Aligned** | 1Y return and 5Y CAGR both positive OR both negative | Short-term confirms long-term thesis |
| 💎 **Accumulation Zone** | 1Y return negative, 5Y CAGR positive | Short-term pain, long-term gain opportunity |
| ⚠️ **Value Trap** | 1Y return positive, 5Y CAGR negative | Short-term rally, long-term decline expected |
| ⚠️ **Bearish Alignment** | Both negative | Consistent bearish outlook |
| ➖ **Mixed** | Unable to determine | Insufficient data |

### Logic Implementation

```python
if one_year_return is not None and expected_cagr is not None:
    if one_year_return < 0 and expected_cagr > hurdle_rate:
        alignment = "💎 Accumulation Zone"
    elif one_year_return > 0 and expected_cagr < 0:
        alignment = "⚠️ Value Trap"
    elif one_year_return < 0 and expected_cagr < 0:
        alignment = "⚠️ Bearish"
    elif one_year_return > 0 and expected_cagr > 0:
        alignment = "✅ Aligned"
    else:
        alignment = "➖ Mixed"
```

---

## Risk Profiling

Risk is measured by the **variance** between Bull and Bear case CAGRs. Higher variance indicates more uncertainty in outcomes.

### Variance Calculation

```
Variance = Bull CAGR High - Bear CAGR Low
```

This represents the full range of potential outcomes.

### Risk Categories

| Risk Profile | Variance Range | Interpretation |
|--------------|---------------|----------------|
| 🏦 **Predictable** | < 10% | Stable, mature business |
| 📊 **Medium Volatility** | 10% - 20% | Normal growth stock |
| 📈 **High Volatility** | 20% - 30% | Growth stock with uncertainty |
| 🎰 **Speculative** | > 30% | High risk/reward profile |

### Example

If a stock has:
- Bear CAGR Low: -5%
- Bull CAGR High: +45%

```
Variance = 45% - (-5%) = 50%
Risk Profile = 🎰 Speculative
```

---

## Optimism Bias Detection

The system flags projections that may be unrealistically optimistic based on mathematical constraints.

### Large-Cap Growth Limits

For companies with market cap > $100B, extremely high CAGRs become mathematically improbable due to the law of large numbers.

**Warning Triggers:**

| Market Cap | CAGR Threshold | Warning |
|------------|---------------|---------|
| > $500B | > 25% | "Mega-cap growth >25% CAGR is historically rare" |
| > $100B | > 35% | "Large-cap growth >35% CAGR is mathematically challenging" |

### Implementation

```python
if market_cap and market_cap > 500e9 and expected_cagr > 25:
    warnings.append("⚠️ Optimism Bias: Mega-cap (>$500B) achieving >25% CAGR is historically rare")
elif market_cap and market_cap > 100e9 and expected_cagr > 35:
    warnings.append("⚠️ Optimism Bias: Large-cap (>$100B) achieving >35% CAGR is mathematically challenging")
```

### Why This Matters

A $500B company growing at 25% CAGR would reach $1.5T in 5 years. Only a handful of companies have ever achieved this. The warning reminds users to scrutinize aggressive assumptions for large companies.

---

## Price/Sales Fallback for Negative EPS

For companies with negative earnings (e.g., TTWO, early-stage growth companies), P/E-based valuation is not applicable. The system uses **Price/Sales (P/S) ratio** as a fallback.

### When P/S Fallback Activates

```python
if pe_low is None or pe_high is None:
    if market_cap and revenue and revenue > 0:
        ps_ratio = market_cap / revenue
        use_ps_ratio = True
```

### P/S-Based Price Projection

```python
revenue_per_share = projected_revenue / shares_outstanding
price_low = revenue_per_share × (ps_ratio × 0.7)   # P/S compression
price_high = revenue_per_share × (ps_ratio × 1.3)  # P/S expansion
```

### Assumptions

| Case | P/S Multiple | Rationale |
|------|-------------|-----------|
| Bear | Current P/S × 0.7 | Multiple compression (market pessimism) |
| Bull | Current P/S × 1.3 | Multiple expansion (market optimism) |

### Limitations of P/S Valuation

1. **No Profitability Signal** — P/S doesn't account for whether the company can become profitable
2. **Industry Variance** — Appropriate P/S ratios vary widely by industry
3. **Revenue Quality** — Doesn't distinguish between high-margin and low-margin revenue

---

## Complete Conviction Score Calculation

Here's the full algorithm for calculating a conviction score:

```python
def calculate_conviction_score(stock_data, hurdle_rate=10.0):
    # 1. Extract CAGR bounds from each scenario
    bear_cagr = (bear_cagr_low + bear_cagr_high) / 2
    base_cagr = (base_cagr_low + base_cagr_high) / 2
    bull_cagr = (bull_cagr_low + bull_cagr_high) / 2
    
    # 2. Calculate weighted expected CAGR
    expected_cagr = (0.25 × bear_cagr) + (0.50 × base_cagr) + (0.25 × bull_cagr)
    
    # 3. Calculate expected 5-year price
    expected_5y_price = current_price × (1 + expected_cagr/100)^5
    
    # 4. Determine conviction rating
    if expected_cagr >= hurdle_rate + 10:
        rating = "🟢 Strong Buy"
    elif expected_cagr >= hurdle_rate:
        rating = "🟡 Buy"
    elif expected_cagr >= hurdle_rate - 5:
        rating = "🟠 Hold"
    elif expected_cagr >= 0:
        rating = "🔴 Avoid"
    else:
        rating = "⛔ Strong Avoid"
    
    # 5. Calculate risk profile
    variance = bull_cagr_high - bear_cagr_low
    if variance < 10:
        risk = "🏦 Predictable"
    elif variance < 20:
        risk = "📊 Medium Vol"
    elif variance < 30:
        risk = "📈 High Vol"
    else:
        risk = "🎰 Speculative"
    
    # 6. Check time alignment
    if one_year_return < 0 and expected_cagr > hurdle_rate:
        alignment = "💎 Accumulation Zone"
    # ... (other conditions)
    
    # 7. Check for optimism bias
    warnings = []
    if market_cap > 500e9 and expected_cagr > 25:
        warnings.append("Optimism Bias: Mega-cap growth warning")
    
    return {
        'expected_cagr': expected_cagr,
        'expected_value_5y': expected_5y_price,
        'conviction_rating': rating,
        'risk_profile': risk,
        'time_alignment': alignment,
        'warnings': warnings
    }
```

---

## Interpreting the Results

### Ideal Stock Profile

A stock with strong conviction characteristics would show:

- 🟢 **Strong Buy** conviction (CAGR significantly above hurdle)
- 🏦 or 📊 **Low-Medium Risk** (predictable outcomes)
- ✅ or 💎 **Aligned or Accumulation** (time alignment confirms thesis)
- **No warnings** (realistic growth expectations)

### Red Flags to Watch

1. **🎰 Speculative + 🟢 Strong Buy** — High conviction but high uncertainty; proceed with caution
2. **⚠️ Value Trap** — Short-term gains may not persist
3. **Optimism Bias warnings** — Projections may be unrealistic
4. **N/A values** — Insufficient data for analysis

### Portfolio Construction Ideas

| Strategy | Filter Criteria |
|----------|----------------|
| **Conservative** | 🟡+ conviction, 🏦📊 risk, ✅ aligned |
| **Growth** | 🟢 conviction, any risk, 💎 accumulation preferred |
| **Contrarian** | 🟡+ conviction, 💎 accumulation zone |
| **Avoid** | 🔴⛔ conviction, ⚠️ value trap |

---

## Assumptions & Limitations

### Key Assumptions

1. **Scenario Probabilities** — 25/50/25 weighting assumes base case is most likely
2. **Constant Growth** — Revenue grows at constant rate (no acceleration/deceleration)
3. **Margin Stability** — Net margins remain stable over 5 years
4. **P/E Stability** — Market assigns similar multiples in the future
5. **No Black Swans** — Model doesn't account for extreme events

### Limitations

1. **Default Assumptions** — Scenarios use mechanical defaults; manual adjustment recommended
2. **Backward-Looking** — Growth rates based on historical performance
3. **No Qualitative Factors** — Doesn't consider management, competition, moat
4. **Data Gaps** — Some stocks may have incomplete data from Yahoo Finance
5. **P/S Fallback Limitations** — Less reliable for unprofitable companies

### Best Practices

1. **Adjust Scenarios** — Review and modify default growth/margin assumptions
2. **Cross-Reference** — Use alongside fundamental research
3. **Focus on Relative Ranking** — Compare stocks within the same watchlist
4. **Monitor Warnings** — Take optimism bias flags seriously
5. **Rebalance Regularly** — Re-run analysis as new data becomes available

---

## Glossary

| Term | Definition |
|------|------------|
| **CAGR** | Compound Annual Growth Rate — annualized return |
| **Conviction** | Confidence level in investment thesis |
| **Hurdle Rate** | Minimum acceptable return (benchmark) |
| **P/E Ratio** | Price-to-Earnings — valuation multiple |
| **P/S Ratio** | Price-to-Sales — revenue-based valuation |
| **Time Alignment** | Consistency between short and long-term trends |
| **Variance** | Spread between best and worst case outcomes |
| **Accumulation Zone** | Short-term weakness + long-term strength |
| **Value Trap** | Appears cheap but fundamentally declining |

---

## Formula Reference

### Quick Reference Card

```
Expected CAGR = 0.25×Bear + 0.50×Base + 0.25×Bull

Expected 5Y Price = Current × (1 + CAGR/100)^5

Variance = Bull CAGR High - Bear CAGR Low

1Y Return = (Current Price / Price 1 Year Ago - 1) × 100

P/S Ratio = Market Cap / Revenue

P/S Price = (Revenue / Shares) × P/S × Adjustment Factor
```

---

*This documentation reflects the current implementation as of December 2024. The Conviction Engine algorithm may be refined in future versions.*
