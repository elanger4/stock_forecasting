# Monte Carlo Simulation - Technical Documentation

This document provides a comprehensive overview of how the Monte Carlo simulation works, the assumptions being made, and how to interpret the results.

---

## Table of Contents

1. [Overview](#overview)
2. [What is Monte Carlo Simulation?](#what-is-monte-carlo-simulation)
3. [Input Parameters](#input-parameters)
4. [The Simulation Algorithm](#the-simulation-algorithm)
5. [Distribution Types](#distribution-types)
6. [Output Metrics](#output-metrics)
7. [Visualizations](#visualizations)
8. [Interpreting Results](#interpreting-results)
9. [Assumptions & Limitations](#assumptions--limitations)

---

## Overview

The Monte Carlo simulation page allows you to run thousands of probabilistic scenarios to understand the **range of possible outcomes** for a stock investment. Unlike the Stock Analysis page which shows discrete Bear/Base/Bull scenarios, Monte Carlo simulation shows the full probability distribution of outcomes.

---

## What is Monte Carlo Simulation?

Monte Carlo simulation is a computational technique that uses **random sampling** to model uncertainty. Named after the famous casino, it "rolls the dice" thousands of times with slightly different assumptions each time.

### Why Use It?

1. **Quantify Uncertainty** — Instead of a single price target, see the full range of possibilities
2. **Probability Estimates** — Know the actual odds of doubling your money or losing 20%
3. **Risk Assessment** — Understand worst-case scenarios with Value at Risk (VaR)
4. **Confidence Intervals** — Get 80% or 90% probability ranges

### Simple Example

If you expect 15% revenue growth but acknowledge it could vary by ±5%:
- Traditional analysis: "Price will be $150"
- Monte Carlo: "Price has 80% chance of being between $120-$180, with median $148"

---

## Input Parameters

### Simulation Settings

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| **Number of Simulations** | 1,000 | 100-50,000 | More simulations = more accurate but slower |
| **Projection Years** | 5 | 1-10 | How far into the future to project |
| **Distribution Type** | Normal | 4 options | How random values are generated |

### Variable Parameters

Each variable has a **mean** (expected value) and **standard deviation** (uncertainty):

#### Revenue Growth
| Parameter | Default Source | Description |
|-----------|---------------|-------------|
| Mean | Current YoY growth from yfinance | Expected annual growth rate |
| Std Dev | Historical growth volatility | How much growth might vary year-to-year |

#### Net Margin
| Parameter | Default Source | Description |
|-----------|---------------|-------------|
| Mean | Current net margin | Expected profitability |
| Std Dev | Historical margin volatility | How much margin might fluctuate |

#### P/E Multiple
| Parameter | Default Source | Description |
|-----------|---------------|-------------|
| Mean | Trailing P/E (capped at 50) | Expected valuation multiple |
| Std Dev | 20% of mean (max 10) | How much the market might re-rate the stock |

### Advanced Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| **Growth-Margin Correlation** | 0.3 | Positive = high growth tends to come with higher margins |

---

## The Simulation Algorithm

### Step-by-Step Process

For each of the N simulations:

```
1. Start with current revenue, margin, EPS, and price

2. For each year (1 to num_years):
   a. Generate random growth rate from distribution
   b. Generate random margin from distribution
   c. Generate random P/E from distribution
   d. Apply growth-margin correlation adjustment
   
   e. Calculate:
      - New Revenue = Previous Revenue × (1 + Growth%)
      - Net Income = Revenue × Margin%
      - EPS = Net Income / Shares Outstanding
      - Price = EPS × P/E (or P/S fallback if EPS < 0)
   
   f. Store the price for this year

3. Record final year price and calculate CAGR
```

### Price Calculation

For **positive EPS**:
```
Price = EPS × P/E Multiple
```

For **negative EPS** (P/S fallback):
```
Price = (Revenue / Shares) × P/S Ratio × Growth Adjustment
```

### CAGR Calculation

```
CAGR = ((Final Price / Current Price)^(1/Years) - 1) × 100
```

---

## Distribution Types

### Normal Distribution (Default)

```
Value = random.normal(mean, std_dev)
```

- **Shape**: Symmetric bell curve
- **Best for**: Variables that can go above or below the mean equally
- **Pros**: Simple, intuitive
- **Cons**: Can produce negative values for growth

### Log-Normal Distribution

```
Value = random.lognormal(adjusted_mean, adjusted_std)
```

- **Shape**: Right-skewed, always positive
- **Best for**: Growth rates, stock prices (can't go negative)
- **Pros**: Prevents impossible negative prices
- **Cons**: More complex to parameterize

### Triangular Distribution

```
Value = random.triangular(mean - 2×std, mean, mean + 2×std)
```

- **Shape**: Triangle with peak at the mode
- **Best for**: When you have min/mode/max estimates
- **Pros**: Intuitive, bounded range
- **Cons**: Sharp cutoffs at min/max

### Uniform Distribution

```
Value = random.uniform(mean - 2×std, mean + 2×std)
```

- **Shape**: Flat, equal probability across range
- **Best for**: When all outcomes in a range are equally likely
- **Pros**: Simple, bounded
- **Cons**: Unrealistic for most financial variables

### Which to Choose?

| Situation | Recommended |
|-----------|-------------|
| General use | Normal |
| High-growth stocks | Log-Normal |
| Scenario planning | Triangular |
| Maximum uncertainty | Uniform |

---

## Output Metrics

### Key Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Expected Price (Mean)** | Average of all final prices | Central tendency, affected by outliers |
| **Median Price** | Middle value when sorted | More robust than mean for skewed distributions |
| **Expected CAGR** | Average of all CAGRs | Annualized expected return |
| **Median CAGR** | Middle CAGR value | More robust return estimate |

### Probability Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Probability of Profit** | % of simulations > current price | Chance of any positive return |
| **Probability of 2x** | % of simulations > 2× current price | Chance of doubling |
| **Probability of -20%** | % of simulations < 0.8× current price | Chance of significant loss |
| **Probability of -50%** | % of simulations < 0.5× current price | Chance of severe loss |

### Confidence Intervals

| Percentile | Meaning |
|------------|---------|
| **10th Percentile** | 90% of outcomes are above this (bearish bound) |
| **25th Percentile** | 75% of outcomes are above this |
| **75th Percentile** | 25% of outcomes are above this |
| **90th Percentile** | 10% of outcomes are above this (bullish bound) |

**80% Confidence Interval** = Range between 10th and 90th percentiles

### Risk Metrics (Value at Risk)

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **VaR 95%** | 5th percentile price | 95% of the time, price will be above this |
| **VaR 99%** | 1st percentile price | 99% of the time, price will be above this |
| **Max Expected Loss (95%)** | (VaR 95% / Current - 1) × 100 | Worst-case loss 95% of the time |

---

## Visualizations

### 1. Price Distribution Histogram

Shows the frequency distribution of final simulated prices.

**Key markers:**
- **Red dashed line**: Current price
- **Green dashed line**: Median simulated price
- **Orange dotted line**: 10th percentile (bearish)
- **Blue dotted line**: 90th percentile (bullish)

**How to read:**
- Taller bars = more likely outcomes
- Width of distribution = uncertainty
- Skewness shows asymmetric risk/reward

### 2. Fan Chart (Price Paths Over Time)

Shows how the range of possible prices evolves year by year.

**Components:**
- **Light blue band**: 10-90% probability range
- **Darker blue band**: 25-75% probability range
- **Blue line**: Median path
- **Red dashed line**: Current price (flat reference)

**How to read:**
- Widening fan = increasing uncertainty over time
- If median stays above current price = generally bullish
- Overlap with current price line = probability of loss

### 3. Cumulative Probability Curve

Shows the probability of exceeding any given price target.

**How to read:**
- Y-axis: Probability of price being HIGHER than X-axis value
- Find your target price on X-axis, read probability on Y-axis
- 50% line intersection = median price
- Steeper curve = more certainty; flatter = more uncertainty

---

## Interpreting Results

### Bullish Signals

- Probability of Profit > 70%
- Median CAGR > 10% (beats market)
- 10th percentile still shows positive return
- VaR 95% > current price (even worst case is profitable)

### Bearish Signals

- Probability of Profit < 50%
- Probability of -20% > 30%
- Wide distribution (high uncertainty)
- Median CAGR < 5%

### High Conviction Opportunities

Look for stocks where:
1. **High probability of profit** (>75%)
2. **Asymmetric upside** (90th percentile much higher than 10th percentile is low)
3. **Reasonable worst case** (VaR 95% not catastrophic)
4. **Median beats hurdle rate** (>10% CAGR)

### Red Flags

1. **Probability of -50% > 10%** — Significant risk of severe loss
2. **Mean >> Median** — Skewed by unrealistic outliers
3. **Very wide confidence interval** — Too much uncertainty
4. **VaR 99% near zero** — Risk of total loss

---

## Assumptions & Limitations

### Key Assumptions

1. **Independent Annual Returns**
   - Each year's growth/margin is drawn independently
   - Reality: There's often momentum or mean reversion

2. **Constant Distribution Parameters**
   - Mean and std dev don't change over time
   - Reality: Volatility clusters, growth rates decay

3. **No Black Swan Events**
   - Distributions don't capture extreme tail events
   - Reality: Pandemics, bankruptcies, breakthroughs happen

4. **P/E Stability**
   - Market assigns similar multiples in the future
   - Reality: Sentiment shifts can dramatically re-rate stocks

5. **No Dividends or Buybacks**
   - Model focuses on price appreciation only
   - Reality: Total return includes dividends

### Limitations

1. **Garbage In, Garbage Out**
   - Results are only as good as your input assumptions
   - Always sanity-check defaults against your research

2. **Historical Volatility ≠ Future Volatility**
   - Past variance may not predict future variance
   - Young companies have less reliable historical data

3. **Correlation Simplification**
   - Only models growth-margin correlation
   - Ignores macro factors, sector correlations

4. **No Fundamental Constraints**
   - Model doesn't prevent impossible scenarios
   - E.g., 50% margins for a retailer

5. **Computational Randomness**
   - Different random seeds = slightly different results
   - Use more simulations for stability

### Best Practices

1. **Adjust Defaults** — Don't blindly trust auto-calculated parameters
2. **Run Multiple Times** — Verify results are stable
3. **Compare Distributions** — Try Normal vs Log-Normal to see sensitivity
4. **Focus on Probabilities** — More useful than point estimates
5. **Use Alongside Fundamentals** — Monte Carlo complements, doesn't replace, research

---

## Formula Reference

### Quick Reference Card

```
# Revenue Projection
Revenue[t] = Revenue[t-1] × (1 + Growth[t] / 100)

# Net Income
Net Income = Revenue × (Margin / 100)

# EPS
EPS = Net Income / Shares Outstanding

# Price (positive EPS)
Price = EPS × P/E

# Price (negative EPS, P/S fallback)
Price = (Revenue / Shares) × P/S Ratio

# CAGR
CAGR = ((Final Price / Current Price)^(1/Years) - 1) × 100

# Probability of Profit
P(Profit) = Count(Final Price > Current Price) / N × 100

# VaR 95%
VaR 95% = 5th Percentile of Final Prices

# 80% Confidence Interval
CI 80% = [10th Percentile, 90th Percentile]
```

---

## Glossary

| Term | Definition |
|------|------------|
| **CAGR** | Compound Annual Growth Rate |
| **Confidence Interval** | Range containing a specified probability of outcomes |
| **Distribution** | Mathematical function describing probability of values |
| **Mean** | Average value |
| **Median** | Middle value when sorted |
| **Monte Carlo** | Simulation technique using random sampling |
| **Percentile** | Value below which a percentage of data falls |
| **Standard Deviation** | Measure of spread/uncertainty |
| **VaR** | Value at Risk — worst-case loss at a confidence level |
| **Volatility** | Degree of price variation over time |

---

*This documentation reflects the current implementation as of December 2024. The simulation algorithm may be refined in future versions.*
