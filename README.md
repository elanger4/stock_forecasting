
This document is a formal **Product Requirements Document (PRD)** designed for a coding agent like **Windsurf**. It provides a clear roadmap for data ingestion, logic processing, and UI rendering to recreate the exact stock valuation model from your screenshot.

---

# ## PRD: Stock Valuation Projection Engine (Forward-Estimate Model)

## 1. Objective

Build a Python-based application that generates a 5-year financial projection table. The tool will calculate future stock price targets by combining user-defined growth rates with **Implied P/E Multiples** derived from the current spread of analyst EPS estimates.

## 2. Data Acquisition (Data Layer)

### A. Automatic Ticker Queries (via `yfinance`)

The software must pull the following "Current Year" (Year 0) metrics:

* **Current Price:** Last closing price.
* **Total Revenue:** From the most recent annual income statement.
* **Net Income:** From the most recent annual income statement.
* **Shares Outstanding:** Total diluted shares.
* **Current EPS:** Trailing Twelve Months (TTM).

### B. Analyst Estimate Queries

To calculate the valuation multiples, query:

* **EPS Low Estimate:** The lowest analyst forecast for the next fiscal year.
* **EPS High Estimate:** The highest analyst forecast for the next fiscal year.

### C. User Inputs

* **Ticker Symbol:** (e.g., "AVGO")
* **Annual Revenue Growth Rate (%):** (e.g., 20%)
* **Target Net Income Margin (%):** (Default to current margin, user-adjustable).

---

## 3. Valuation & Projection Logic

### Step 1: Establish the "Forward P/E" Bounds

Instead of historical P/E, we use the inverse relationship of current price to analyst estimates:

* **P/E High Est** = `Current Price` / `EPS Low Estimate`
* *Logic: If the company only hits the low end of earnings, the market is currently "overpaying" at this multiple.*


* **P/E Low Est** = `Current Price` / `EPS High Estimate`
* *Logic: If the company hits the high end of earnings, the current price represents this more "conservative" multiple.*



### Step 2: 5-Year Projection Formulas

For each year  from 1 to 5:

| Row Item | Formula |
| --- | --- |
| **Revenue** |  |
| **Rev Growth %** | User Input (static across years) |
| **Net Income** |  |
| **Net Inc Growth %** |  |
| **Net Inc Margins** | User Input (static across years) |
| **EPS** |  |
| **Share Price Low** |  |
| **Share Price High** |  |
| **CAGR (Low/High)** |  |

---

## 4. Feature Requirements for Windsurf Implementation

### A. The Calculation Engine

* **Recursive Calculation:** Ensure Revenue and Net Income compound annually based on the growth rate.
* **Fixed Share Count:** Assume `Shares Outstanding` remains constant unless an optional "Dilution %" input is added.

### B. The User Interface (UI)

* **Table Display:** Render a table mirroring the provided screenshot with Years 0–5 as columns and metrics as rows.
* **Currency Formatting:** Automatically format large numbers (e.g., 160.5B) for scannability.
* **Dynamic Update:** When the user changes the Revenue Growth %, the entire table and CAGR should recalculate instantly.

### C. Error Handling

* **Missing Estimates:** If `yfinance` returns a `null` for analyst estimates, provide a fallback to 5-year historical average P/E.
* **Negative Earnings:** If the stock has negative EPS, display a warning that P/E-based valuation is not applicable.

---

## 5. Implementation Prompt for Windsurf

> "Create a Python Streamlit app that takes a stock ticker and a revenue growth rate as input. Use yfinance to fetch the current revenue, net income, and shares outstanding. Specifically, calculate a P/E High/Low range by dividing the current price by the 'earningsLow' and 'earningsHigh' analyst estimates. Use these bounds to project a 5-year table showing Revenue, Net Income, EPS, and Share Price targets, including the CAGR for the High and Low price scenarios."

**Would you like me to generate a starter Python script containing the core logic to give Windsurf a head start?**
