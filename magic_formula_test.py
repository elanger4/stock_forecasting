
#!/usr/bin/env python3
"""
Portfolio YTD Comparison Script
Compares current stock prices against January 2, 2026 baseline
"""

import yfinance as yf
from datetime import datetime
from tabulate import tabulate

# January 2, 2026 baseline prices
BASELINE_DATA = {
    "BBUC": {"company": "Brookfield Business Corp", "jan2_price": 36.40},
    "BMBL": {"company": "Bumble Inc", "jan2_price": 3.62},
    "BTM": {"company": "Bitcoin Depot", "jan2_price": 1.32},
    "BTMD": {"company": "Biote Corp", "jan2_price": 2.47},
    "CPRX": {"company": "Catalyst Pharmaceuticals", "jan2_price": 23.15},
    "CROX": {"company": "Crocs Inc", "jan2_price": 85.52},
    "EGHSF": {"company": "Enghouse Systems", "jan2_price": 14.25},
    "GAMB": {"company": "Gambling.com Group", "jan2_price": 5.05},
    "HPQ": {"company": "HP Inc", "jan2_price": 22.12},
    "HRB": {"company": "H&R Block", "jan2_price": 42.61},
    "HRMY": {"company": "Harmony Biosciences", "jan2_price": 32.88},
    "INVA": {"company": "Innoviva Inc", "jan2_price": 18.52},
    "KROS": {"company": "Keros Therapeutics", "jan2_price": 26.55},
    "MD": {"company": "Pediatrix Medical", "jan2_price": 18.75},
    "OSPN": {"company": "OneSpan Inc", "jan2_price": 10.25},
    "PBI": {"company": "Pitney Bowes", "jan2_price": 9.45},
    "PBYI": {"company": "Puma Biotechnology", "jan2_price": 4.18},
    "PLTK": {"company": "Playtika Holding", "jan2_price": 3.75},
    "PTCT": {"company": "PTC Therapeutics", "jan2_price": 41.80},
    "PXED": {"company": "Pimco Rafi Dyn Multi-Fac ETF", "jan2_price": 29.25},
    "RGS": {"company": "Regis Corp", "jan2_price": 24.50},
    "RIGL": {"company": "Rigel Pharmaceuticals", "jan2_price": 17.15},
    "RMNI": {"company": "Rimini Street", "jan2_price": 3.15},
    "SIGA": {"company": "SIGA Technologies", "jan2_price": 6.35},
    "SSTK": {"company": "Shutterstock", "jan2_price": 22.85},
    "TZOO": {"company": "Travelzoo", "jan2_price": 5.05},
    "UIS": {"company": "Unisys Corp", "jan2_price": 3.82},
    "VSNT": {"company": "Versant Health", "jan2_price": 29.50},
    "WLY": {"company": "John Wiley & Sons", "jan2_price": 39.75},
    "XPOF": {"company": "Xponential Fitness", "jan2_price": 9.55},
}

INVESTMENT_PER_STOCK = 1000  # Change this to adjust investment amount


def fetch_current_prices(tickers: list) -> dict:
    """Fetch current prices for all tickers using yfinance."""
    print("Fetching current prices...")
    prices = {}
    
    # Fetch all at once for efficiency
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


if __name__ == "__main__":
    main()
