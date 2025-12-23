# Stock Analysis Page - Technical Documentation

This document provides a comprehensive overview of how the Stock Analysis page calculates projections, valuations, and metrics. Understanding these calculations will help you interpret the results and recognize the assumptions being made.

---

## Table of Contents

1. [Overview](#overview)
2. [Data Sources](#data-sources)
3. [Key Metrics & Definitions](#key-metrics--definitions)
4. [Scenario-Based Projections](#scenario-based-projections)
5. [Share Price Calculations](#share-price-calculations)
6. [CAGR Calculations](#cagr-calculations)
7. [Historical Data](#historical-data)
8. [Valuation Ratios](#valuation-ratios)
9. [Assumptions & Limitations](#assumptions--limitations)

---

## Overview

The Stock Analysis page is a **5-Year Forward-Estimate Model** that projects a company's financial performance under three scenarios:

- **🐻 Bear Case** — Conservative/pessimistic assumptions
- **📊 Base Case** — Moderate/expected assumptions  
- **🐂 Bull Case** — Optimistic assumptions

Each scenario uses different revenue growth rates and net income margins to project future earnings and share prices.

---

## Data Sources

All financial data is fetched from **Yahoo Finance** via the `yfinance` Python library:

| Data Point | Source |
|------------|--------|
| Current Price | `info['currentPrice']` or `info['regularMarketPrice']` |
| Revenue | `income_stmt['Total Revenue']` or `info['totalRevenue']` |
| Net Income | `income_stmt['Net Income']` or `info['netIncomeToCommon']` |
| EPS | `income_stmt['Basic EPS']` or `income_stmt['Diluted EPS']` |
| Shares Outstanding | `info['sharesOutstanding']` |
| 52-Week Low/High | `info['fiftyTwoWeekLow']`, `info['fiftyTwoWeekHigh']` |
| Revenue Growth | `info['revenueGrowth']` (YoY) |
| Historical P/E | `info['trailingPE']` |
| Market Cap | `info['marketCap']` |

### Historical Data (3 Years)

When "Show Historical Data" is enabled, the system fetches up to 3 years of historical financials from the income statement and calculates year-over-year metrics.

---

## Key Metrics & Definitions

### Revenue
Total sales/income before any expenses. This is the "top line" of the income statement.

### Revenue Growth %
Year-over-year percentage increase in revenue:

```
Rev Growth = ((Current Year Revenue / Prior Year Revenue) - 1) × 100
```

### Net Income
Profit after all expenses, taxes, and costs. This is the "bottom line."

### Net Income Growth %
Year-over-year percentage increase in net income:

```
NI Growth = ((Current Year Net Income / Prior Year Net Income) - 1) × 100
```

**Note:** For companies transitioning from loss to profit (or vice versa), this calculation may produce extreme values or be shown as "—".

### Net Income Margin %
Profitability ratio showing what percentage of revenue becomes profit:

```
Net Margin = (Net Income / Revenue) × 100
```

### EPS (Earnings Per Share)
Net income divided by shares outstanding:

```
EPS = Net Income / Shares Outstanding
```

**ADR Note:** For American Depositary Receipts (ADRs), the system uses the reported EPS directly rather than calculating from net income, as share counts may not align with the ADR ratio.

### P/E Ratio (Price-to-Earnings)
How much investors pay per dollar of earnings:

```
P/E = Share Price / EPS
```

---

## Scenario-Based Projections

### Default Scenario Settings

When you load a stock, the system calculates default values based on current financials:

| Scenario | Revenue Growth | Net Margin | P/E Range |
|----------|---------------|------------|-----------|
| **Bear** | Current Growth - 5% | Current Margin - 3% | P/E Low - 2, P/E High - 2 |
| **Base** | Current Growth | Current Margin | P/E Low, P/E High |
| **Bull** | Current Growth + 5% | Current Margin + 3% | P/E Low + 2, P/E High + 2 |

### P/E Low & P/E High Estimation

The system estimates P/E bounds from analyst EPS estimates:

```python
if eps_low > 0 and eps_high > 0:
    pe_high = current_price / eps_low   # Higher P/E when EPS is lower
    pe_low = current_price / eps_high   # Lower P/E when EPS is higher
elif historical_pe > 0:
    pe_low = historical_pe × 0.8
    pe_high = historical_pe × 1.2
```

### Revenue Projection Formula

For each future year (Year 1 through Year 5):

```
Projected Revenue = Current Revenue × (1 + Growth Rate)^Year
```

Example with 15% growth:
- Year 1: $1B × 1.15¹ = $1.15B
- Year 2: $1B × 1.15² = $1.32B
- Year 3: $1B × 1.15³ = $1.52B
- Year 4: $1B × 1.15⁴ = $1.75B
- Year 5: $1B × 1.15⁵ = $2.01B

### Net Income Projection Formula

```
Projected Net Income = Projected Revenue × (Net Margin / 100)
```

### EPS Projection Formula

To avoid ADR/foreign stock share count issues, EPS is grown proportionally:

```python
if base_net_income > 0 and base_eps:
    ni_growth_multiplier = projected_net_income / base_net_income
    projected_eps = base_eps × ni_growth_multiplier
else:
    projected_eps = projected_net_income / shares_outstanding
```

---

## Share Price Calculations

### Year 0 (Current Year)

Uses the **52-week low and high** as the price range:

```
Price Low = 52-Week Low
Price High = 52-Week High
```

### Future Years (Year 1-5)

Uses **P/E-based valuation**:

```
Price Low = Projected EPS × P/E Low
Price High = Projected EPS × P/E High
```

### Negative EPS Handling

For stocks with negative EPS, P/E-based valuation is not applicable. The system displays a warning:

> ⚠️ This stock has negative EPS. P/E-based valuation may not be applicable.

In this case, share price projections may show "—" or use alternative methods (see Watchlist Comparison documentation for P/S ratio fallback).

---

## CAGR Calculations

**CAGR (Compound Annual Growth Rate)** measures the annualized return from current price to projected price.

### Formula

```
CAGR = ((Future Price / Current Price)^(1/Years) - 1) × 100
```

### Example

If current price is $100 and Year 5 projected price is $200:

```
CAGR = ((200 / 100)^(1/5) - 1) × 100
     = (2^0.2 - 1) × 100
     = (1.1487 - 1) × 100
     = 14.87%
```

### CAGR Low & CAGR High

The table shows two CAGR values:
- **CAGR Low** — Based on Share Price Low projection
- **CAGR High** — Based on Share Price High projection

These represent the range of potential annualized returns under each scenario.

---

## Historical Data

When "Show Historical Data" is enabled, the table displays 3 years of actual historical performance before the projections.

### Historical Metrics Calculated

| Metric | Calculation |
|--------|-------------|
| Revenue | From income statement |
| Rev Growth | `(Current Year / Prior Year - 1) × 100` |
| Net Income | From income statement |
| NI Growth | `(Current Year / Prior Year - 1) × 100` |
| Net Margin | `(Net Income / Revenue) × 100` |
| EPS | From income statement or calculated |
| P/E | `Year-End Price / EPS` |
| Price Low/High | Min/Max of daily prices for that year |

### Year 0 Growth Calculation

The current year's (Year 0) growth rates are calculated by comparing to the most recent historical year:

```python
rev_growth = ((current_revenue / prior_year_revenue) - 1) × 100
ni_growth = ((current_net_income / prior_year_net_income) - 1) × 100
```

---

## Valuation Ratios

The Valuation Ratios section compares the stock's current multiples to sector averages.

### Metrics Displayed

| Ratio | Formula | Interpretation |
|-------|---------|----------------|
| **P/E Ratio** | Price / EPS | Lower may indicate undervaluation |
| **Forward P/E** | Price / Forward EPS | Based on analyst estimates |
| **PEG Ratio** | P/E / Growth Rate | <1 may indicate undervaluation |
| **P/S Ratio** | Market Cap / Revenue | Useful for unprofitable companies |
| **P/B Ratio** | Price / Book Value | <1 may indicate undervaluation |
| **EV/EBITDA** | Enterprise Value / EBITDA | Common for M&A comparisons |

### Sector Comparison

The "vs Sector" column shows how the stock compares to its sector average (data from Yahoo Finance when available).

---

## Assumptions & Limitations

### Key Assumptions

1. **Constant Growth Rate** — Revenue grows at a constant rate each year (no acceleration/deceleration)

2. **Constant Margin** — Net income margin remains stable throughout the projection period

3. **Constant P/E Multiple** — The market values the stock at the same P/E ratio in the future

4. **No Share Dilution** — Share count remains constant (no buybacks or issuances)

5. **No Dividends** — Model focuses on price appreciation, not total return

6. **Linear Extrapolation** — Past performance is used to estimate future performance

### Limitations

1. **Data Quality** — Relies on Yahoo Finance data, which may have gaps or errors for some stocks (especially ADRs, foreign stocks, or recent IPOs)

2. **P/E Limitations** — P/E-based valuation doesn't work for:
   - Companies with negative earnings
   - Cyclical companies at earnings peaks/troughs
   - High-growth companies with minimal current earnings

3. **No Macro Factors** — Model doesn't account for:
   - Interest rate changes
   - Economic cycles
   - Industry disruption
   - Competitive dynamics

4. **Historical Bias** — Default growth rates are based on recent performance, which may not persist

5. **Single Point Estimates** — While we show ranges (Low/High), these are still estimates with significant uncertainty

### Best Practices

1. **Adjust Scenarios** — Don't rely solely on defaults; adjust growth rates and margins based on your research

2. **Compare to History** — Enable historical data to see if projections are realistic vs. past performance

3. **Use Multiple Methods** — Cross-reference with DCF, comparable analysis, and other valuation methods

4. **Consider Qualitative Factors** — Management quality, competitive moat, and industry trends matter

5. **Margin of Safety** — Focus on the Bear case to understand downside risk

---

## Glossary

| Term | Definition |
|------|------------|
| **ADR** | American Depositary Receipt — US-traded shares of foreign companies |
| **CAGR** | Compound Annual Growth Rate |
| **EPS** | Earnings Per Share |
| **EV** | Enterprise Value = Market Cap + Debt - Cash |
| **EBITDA** | Earnings Before Interest, Taxes, Depreciation, Amortization |
| **P/E** | Price-to-Earnings Ratio |
| **P/S** | Price-to-Sales Ratio |
| **P/B** | Price-to-Book Ratio |
| **PEG** | P/E divided by Growth rate |
| **TTM** | Trailing Twelve Months |
| **YoY** | Year-over-Year |

---

*This documentation reflects the current implementation as of December 2025. Calculations may be updated in future versions.*
