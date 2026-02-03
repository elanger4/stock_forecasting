#!/usr/bin/env python3
"""
Portfolio YTD Comparison Script
Compares current stock prices against January 2, 2026 baseline

To run:
    pip install yfinance tabulate matplotlib
    python portfolio_ytd.py
"""

import yfinance as yf
from datetime import datetime
from tabulate import tabulate
import matplotlib.pyplot as plt
import pandas as pd

# January 2, 2026 baseline prices (verified from stockanalysis.com)
BASELINE_DATA = {
    "BBUC": {"company": "Brookfield Business Corp", "jan2_price": 36.40},
    "BMBL": {"company": "Bumble Inc", "jan2_price": 3.62},
    "BTM": {"company": "Bitcoin Depot", "jan2_price": 1.32},
    "BTMD": {"company": "Biote Corp", "jan2_price": 2.47},
    "CPRX": {"company": "Catalyst Pharmaceuticals", "jan2_price": 23.15},
    "CROX": {"company": "Crocs Inc", "jan2_price": 85.52},
    "EGHSF": {"company": "Enghouse Systems", "jan2_price": 14.25},
    "GAMB": {"company": "Gambling.com Group", "jan2_price": 5.29},
    "HPQ": {"company": "HP Inc", "jan2_price": 22.12},
    "HRB": {"company": "H&R Block", "jan2_price": 42.61},
    "HRMY": {"company": "Harmony Biosciences", "jan2_price": 37.38},
    "INVA": {"company": "Innoviva Inc", "jan2_price": 19.87},
    "KROS": {"company": "Keros Therapeutics", "jan2_price": 18.54},
    "MD": {"company": "Pediatrix Medical", "jan2_price": 21.38},
    "OSPN": {"company": "OneSpan Inc", "jan2_price": 12.38},
    "PBI": {"company": "Pitney Bowes", "jan2_price": 10.58},
    "PBYI": {"company": "Puma Biotechnology", "jan2_price": 5.94},
    "PLTK": {"company": "Playtika Holding", "jan2_price": 3.98},
    "PTCT": {"company": "PTC Therapeutics", "jan2_price": 76.77},
    "PXED": {"company": "Pimco Rafi Dyn Multi-Fac ETF", "jan2_price": 29.82},
    "RGS": {"company": "Regis Corp", "jan2_price": 27.66},
    "RIGL": {"company": "Rigel Pharmaceuticals", "jan2_price": 41.83},
    "RMNI": {"company": "Rimini Street", "jan2_price": 3.91},
    "SIGA": {"company": "SIGA Technologies", "jan2_price": 6.27},
    "SSTK": {"company": "Shutterstock", "jan2_price": 19.21},
    "TZOO": {"company": "Travelzoo", "jan2_price": 7.11},
    "UIS": {"company": "Unisys Corp", "jan2_price": 2.78},
    "VSNT": {"company": "Versant Health", "jan2_price": 46.87},
    "WLY": {"company": "John Wiley & Sons", "jan2_price": 30.61},
    "XPOF": {"company": "Xponential Fitness", "jan2_price": 8.21},
}

INVESTMENT_PER_STOCK = 1000  # Change this to adjust investment amount


def fetch_current_prices(tickers: list) -> dict:
    """Fetch current prices for all tickers using yfinance."""
    print("Fetching current prices...")
    prices = {}
    
    # Fetch all at once for efficiency
    try:
        data = yf.download(tickers, period="1d", progress=False)
        
        for ticker in tickers:
            try:
                if len(tickers) == 1:
                    price = data["Close"].iloc[-1]
                else:
                    price = data["Close"][ticker].iloc[-1]
                prices[ticker] = round(float(price), 2)
            except Exception:
                prices[ticker] = None
    except Exception as e:
        print(f"Error fetching data: {e}")
        for ticker in tickers:
            prices[ticker] = None
            
    return prices


def calculate_returns(baseline: dict, current_prices: dict) -> list:
    """Calculate returns for each stock."""
    results = []
    
    for ticker, data in baseline.items():
        jan2 = data["jan2_price"]
        current = current_prices.get(ticker)
        
        if current and current > 0:
            pct_return = ((current - jan2) / jan2) * 100
            new_value = INVESTMENT_PER_STOCK * (current / jan2)
        else:
            pct_return = None
            new_value = None
            
        results.append({
            "ticker": ticker,
            "company": data["company"],
            "jan2_price": jan2,
            "current_price": current,
            "return_pct": pct_return,
            "value": new_value,
        })
    
    return results


def plot_cumulative_performance(tickers: list, baseline_date: str = "2026-01-02"):
    """Create a line chart comparing portfolio value vs SPY over time."""
    print("\nGenerating performance comparison chart...")

    # Fetch historical data for all stocks and SPY
    all_tickers = tickers + ["SPY"]
    hist_data = yf.download(all_tickers, start=baseline_date, progress=False)

    # Check if we got any data
    if hist_data.empty:
        print("Warning: No historical data retrieved")
        return

    # Extract adjusted close prices
    try:
        if isinstance(hist_data.columns, pd.MultiIndex):
            # Extract 'Close' prices (prefer Adj Close, fallback to Close)
            level_0_values = hist_data.columns.get_level_values(0).unique().tolist()
            price_col = 'Adj Close' if 'Adj Close' in level_0_values else 'Close'

            adj_close_cols = [col for col in hist_data.columns if col[0] == price_col]
            if adj_close_cols:
                prices_df = pd.DataFrame()
                for col in adj_close_cols:
                    ticker = col[1]
                    prices_df[ticker] = hist_data[col]
            else:
                print("Warning: No price data found")
                return
        else:
            if 'Adj Close' in hist_data.columns:
                prices_df = pd.DataFrame({all_tickers[0]: hist_data['Adj Close']})
            elif 'Close' in hist_data.columns:
                prices_df = pd.DataFrame({all_tickers[0]: hist_data['Close']})
            else:
                print("Warning: No price columns found")
                return
    except Exception as e:
        print(f"Error extracting price data: {e}")
        return

    # Get portfolio stocks and SPY
    portfolio_stocks = [t for t in tickers if t in prices_df.columns]

    if not portfolio_stocks:
        print("Warning: No valid stock data found")
        return

    if "SPY" not in prices_df.columns:
        print("Warning: SPY data not found")
        return

    # Clean data
    valid_data = prices_df[[*portfolio_stocks, "SPY"]].copy()
    valid_data = valid_data.dropna(subset=["SPY"])
    valid_data[portfolio_stocks] = valid_data[portfolio_stocks].ffill()

    # Calculate total initial investment
    num_stocks = len(portfolio_stocks)
    total_investment = num_stocks * INVESTMENT_PER_STOCK

    # Normalize portfolio stocks to their starting values and calculate total value
    portfolio_values = pd.DataFrame()
    for stock in portfolio_stocks:
        # Each stock starts with $INVESTMENT_PER_STOCK invested
        portfolio_values[stock] = (valid_data[stock] / valid_data[stock].iloc[0]) * INVESTMENT_PER_STOCK

    # Total portfolio value is sum of all stock values
    portfolio_total = portfolio_values.sum(axis=1)

    # SPY value starting with same total investment
    spy_value = (valid_data["SPY"] / valid_data["SPY"].iloc[0]) * total_investment

    # Create the line plot
    plt.figure(figsize=(12, 7))
    plt.plot(portfolio_total.index, portfolio_total.values, label='Magic Formula Portfolio', linewidth=2, color='#2E86AB')
    plt.plot(spy_value.index, spy_value.values, label='SPY', linewidth=2, color='#A23B72')

    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Portfolio Value ($)', fontsize=12)
    plt.title(f'Portfolio Performance vs SPY\n(Initial Investment: ${total_investment:,.0f})', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best', fontsize=11)

    # Add statistics box
    final_portfolio = portfolio_total.iloc[-1]
    final_spy = spy_value.iloc[-1]
    portfolio_return = ((final_portfolio - total_investment) / total_investment) * 100
    spy_return = ((final_spy - total_investment) / total_investment) * 100
    outperformance = portfolio_return - spy_return

    stats_text = (f'Portfolio: ${final_portfolio:,.0f} ({portfolio_return:+.2f}%)\n'
                  f'SPY: ${final_spy:,.0f} ({spy_return:+.2f}%)\n'
                  f'Outperformance: {outperformance:+.2f}%')

    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Format y-axis as currency
    ax = plt.gca()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

    plt.tight_layout()
    plt.show()


def print_report(results: list):
    """Print formatted report."""
    print("\n" + "=" * 80)
    print(f"PORTFOLIO YTD COMPARISON REPORT")
    print(f"Baseline: January 2, 2026 | Current: {datetime.now().strftime('%B %d, %Y %I:%M %p')}")
    print(f"Investment per stock: ${INVESTMENT_PER_STOCK:,}")
    print("=" * 80 + "\n")
    
    # Prepare table data
    table_data = []
    for r in results:
        if r["current_price"] and r["return_pct"] is not None:
            ret_str = f"{r['return_pct']:+.2f}%"
            val_str = f"${r['value']:,.2f}"
        else:
            ret_str = "N/A"
            val_str = "N/A"
            
        table_data.append([
            r["ticker"],
            r["company"][:25],
            f"${r['jan2_price']:.2f}",
            f"${r['current_price']:.2f}" if r["current_price"] else "N/A",
            ret_str,
            val_str,
        ])
    
    headers = ["Ticker", "Company", "Jan 2", "Current", "Return", f"${INVESTMENT_PER_STOCK:,} →"]
    print(tabulate(table_data, headers=headers, tablefmt="simple"))
    
    # Portfolio summary
    valid = [r for r in results if r["return_pct"] is not None]
    if valid:
        total_invested = len(valid) * INVESTMENT_PER_STOCK
        total_value = sum(r["value"] for r in valid)
        portfolio_return = ((total_value - total_invested) / total_invested) * 100
        
        print("\n" + "-" * 80)
        print("PORTFOLIO SUMMARY")
        print("-" * 80)
        print(f"Stocks tracked:     {len(valid)}/{len(results)}")
        print(f"Total invested:     ${total_invested:,.2f}")
        print(f"Current value:      ${total_value:,.2f}")
        print(f"Gain/Loss:          ${total_value - total_invested:+,.2f}")
        print(f"Portfolio return:   {portfolio_return:+.2f}%")
        
        # Top/Bottom performers
        sorted_results = sorted(valid, key=lambda x: x["return_pct"], reverse=True)
        
        print("\n" + "-" * 80)
        print("🏆 TOP 5 PERFORMERS")
        print("-" * 80)
        for i, r in enumerate(sorted_results[:5], 1):
            print(f"  {i}. {r['ticker']:6} {r['return_pct']:+8.2f}%  (${r['jan2_price']:.2f} → ${r['current_price']:.2f})")
        
        print("\n" + "-" * 80)
        print("📉 BOTTOM 5 PERFORMERS")
        print("-" * 80)
        for i, r in enumerate(sorted_results[-5:][::-1], 1):
            print(f"  {i}. {r['ticker']:6} {r['return_pct']:+8.2f}%  (${r['jan2_price']:.2f} → ${r['current_price']:.2f})")
    
    print("\n" + "=" * 80)


def main():
    tickers = list(BASELINE_DATA.keys())
    current_prices = fetch_current_prices(tickers)
    results = calculate_returns(BASELINE_DATA, current_prices)
    print_report(results)

    # Generate cumulative performance comparison chart
    plot_cumulative_performance(tickers)


if __name__ == "__main__":
    main()
