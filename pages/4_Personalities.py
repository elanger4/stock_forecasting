import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(page_title="Personality Analysis", layout="wide")

# Hide Streamlit default elements
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    [data-testid="stSidebarNav"] {display: none;}
</style>
""", unsafe_allow_html=True)

# --- Page Navigation ---
nav_col1, nav_col2, nav_col3, nav_col4, nav_col5 = st.columns([1, 1, 1, 1, 1])
with nav_col1:
    st.page_link("app.py", label="👉 Stock Analysis")
with nav_col2:
    st.page_link("pages/1_Watchlist_Comparison.py", label="👉 Watchlist")
with nav_col3:
    st.page_link("pages/2_Monte_Carlo.py", label="👉 Monte Carlo")
with nav_col4:
    st.page_link("pages/3_Cross_Asset_Dashboard.py", label="👉 Cross-Asset")
with nav_col5:
    st.markdown("**🎭 Personalities** *(current)*")

st.markdown("---")

# --- Title and Description ---
st.title("🎭 Personality Stock Analysis")
st.markdown("""
Evaluate stocks through the lens of famous investors. Select one or more personalities to see how they would analyze a given stock based on their investment philosophies.
""")

# --- Personality Definitions ---
PERSONALITIES = {
    "warren_buffett": {
        "name": "Warren Buffett",
        "emoji": "🏛️",
        "tagline": "Quality + Moat + Management = Compounding Machine",
        "philosophy": "Buy wonderful businesses at fair prices, hold forever",
        "color": "#1E3A5F"
    },
    "roaring_kitty": {
        "name": "Roaring Kitty",
        "emoji": "🐱",
        "tagline": "Value + Catalyst + Controversy = Opportunity",
        "philosophy": "Contrarian deep value with short squeeze potential",
        "color": "#8B0000"
    },
    "stanley_druckenmiller": {
        "name": "Stanley Druckenmiller",
        "emoji": "📈",
        "tagline": "Macro + Momentum + Conviction = Outsized Returns",
        "philosophy": "Ride macro trends with aggressive sizing when conviction is high",
        "color": "#2E7D32"
    },
    "joel_greenblatt": {
        "name": "Joel Greenblatt",
        "emoji": "🧮",
        "tagline": "Quality + Value + Discipline = Market-Beating Returns",
        "philosophy": "Buy good companies (high ROC) at bargain prices (high earnings yield)",
        "color": "#6A0DAD"
    },
    "ray_dalio": {
        "name": "Ray Dalio",
        "emoji": "🌊",
        "tagline": "Diversification + Balance + Risk Parity = All-Weather Returns",
        "philosophy": "Balance across economic environments, not asset classes",
        "color": "#FF6600"
    },
    "peter_lynch": {
        "name": "Peter Lynch",
        "emoji": "📚",
        "tagline": "Know What You Own + Simple Businesses + Patience = Tenbaggers",
        "philosophy": "Invest in what you know, find simple businesses before Wall Street discovers them",
        "color": "#8B4513"
    }
}


def format_currency(value):
    """Format large numbers as currency with B/M suffix."""
    if value is None or pd.isna(value):
        return "N/A"
    if abs(value) >= 1e12:
        return f"${value/1e12:.1f}T"
    elif abs(value) >= 1e9:
        return f"${value/1e9:.1f}B"
    elif abs(value) >= 1e6:
        return f"${value/1e6:.1f}M"
    else:
        return f"${value:,.0f}"


def format_percent(value, decimals=1):
    """Format as percentage."""
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:.{decimals}f}%"


def format_ratio(value, decimals=2):
    """Format as ratio."""
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:.{decimals}f}"


@st.cache_data(ttl=600, show_spinner=False)
def fetch_stock_data(ticker_symbol: str) -> dict:
    """Fetch comprehensive stock data from yfinance."""
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info
        
        if not info or info.get('quoteType') not in ['EQUITY', None]:
            return {'success': False, 'error': 'Not a valid equity ticker'}
        
        # Get financial statements
        try:
            financials = ticker.financials
            balance_sheet = ticker.balance_sheet
            cashflow_stmt = ticker.cashflow
        except:
            financials = None
            balance_sheet = None
            cashflow_stmt = None
        
        # Extract all needed data
        data = {
            'success': True,
            'ticker': ticker_symbol,
            'company_name': info.get('longName') or info.get('shortName', ticker_symbol),
            'sector': info.get('sector', 'Unknown'),
            'industry': info.get('industry', 'Unknown'),
            
            # Price & Valuation
            'current_price': info.get('currentPrice') or info.get('regularMarketPrice'),
            'market_cap': info.get('marketCap'),
            'enterprise_value': info.get('enterpriseValue'),
            'trailing_pe': info.get('trailingPE'),
            'forward_pe': info.get('forwardPE'),
            'peg_ratio': info.get('pegRatio'),
            'price_to_book': info.get('priceToBook'),
            'price_to_sales': info.get('priceToSalesTrailing12Months'),
            'ev_to_ebitda': info.get('enterpriseToEbitda'),
            'ev_to_revenue': info.get('enterpriseToRevenue'),
            
            # Profitability
            'roe': info.get('returnOnEquity'),
            'roa': info.get('returnOnAssets'),
            'profit_margins': info.get('profitMargins'),
            'operating_margins': info.get('operatingMargins'),
            'gross_margins': info.get('grossMargins'),
            
            # Balance Sheet
            'total_cash': info.get('totalCash'),
            'total_debt': info.get('totalDebt'),
            'debt_to_equity': info.get('debtToEquity'),
            'current_ratio': info.get('currentRatio'),
            'quick_ratio': info.get('quickRatio'),
            
            # Cash Flow
            'free_cashflow': info.get('freeCashflow'),
            'operating_cashflow': info.get('operatingCashflow'),
            
            # Short Interest (for Roaring Kitty)
            'shares_short': info.get('sharesShort'),
            'short_ratio': info.get('shortRatio'),
            'short_percent_float': info.get('shortPercentOfFloat'),
            'float_shares': info.get('floatShares'),
            'shares_outstanding': info.get('sharesOutstanding'),
            
            # Ownership
            'held_percent_insiders': info.get('heldPercentInsiders'),
            'held_percent_institutions': info.get('heldPercentInstitutions'),
            
            # Price History
            'fifty_two_week_low': info.get('fiftyTwoWeekLow'),
            'fifty_two_week_high': info.get('fiftyTwoWeekHigh'),
            'fifty_day_average': info.get('fiftyDayAverage'),
            'two_hundred_day_average': info.get('twoHundredDayAverage'),
            
            # Statements for deeper analysis
            'financials': financials,
            'balance_sheet': balance_sheet,
            'cashflow': cashflow_stmt,
        }
        
        # Calculate derived metrics
        if data['total_cash'] and data['total_debt']:
            data['net_cash'] = data['total_cash'] - data['total_debt']
        else:
            data['net_cash'] = None
        
        if data['free_cashflow'] and data['market_cap'] and data['market_cap'] > 0:
            data['fcf_yield'] = (data['free_cashflow'] / data['market_cap']) * 100
        else:
            data['fcf_yield'] = None
        
        if data['current_price'] and data['fifty_two_week_high'] and data['fifty_two_week_high'] > 0:
            data['pct_from_high'] = ((data['fifty_two_week_high'] - data['current_price']) / data['fifty_two_week_high']) * 100
        else:
            data['pct_from_high'] = None
        
        if data['current_price'] and data['fifty_two_week_low'] and data['fifty_two_week_low'] > 0:
            data['pct_from_low'] = ((data['current_price'] - data['fifty_two_week_low']) / data['fifty_two_week_low']) * 100
        else:
            data['pct_from_low'] = None
        
        return data
        
    except Exception as e:
        return {'success': False, 'error': str(e)}


def evaluate_buffett(data: dict) -> dict:
    """
    Evaluate a stock using Warren Buffett's methodology.
    100-point scale: Quality (40), Moat (30), Balance Sheet (15), Valuation (15)
    """
    metrics = {}
    score = 0
    max_score = 0
    reasons = []
    red_flags = []
    
    # === QUALITY METRICS (40 points possible) ===
    
    # Return on Equity (10 points)
    max_score += 10
    roe = data.get('roe')
    if roe:
        metrics['ROE'] = {'value': f"{roe*100:.1f}%", 'raw': roe * 100}
        if roe > 0.20:
            score += 10
            reasons.append(f"Excellent ROE: {roe*100:.1f}%")
        elif roe > 0.15:
            score += 7
            reasons.append(f"Strong ROE: {roe*100:.1f}%")
        elif roe > 0.10:
            score += 4
            reasons.append(f"Good ROE: {roe*100:.1f}%")
        elif roe < 0.05:
            red_flags.append(f"Weak ROE: {roe*100:.1f}%")
    else:
        metrics['ROE'] = {'value': 'N/A', 'raw': None}
    
    # Profit Margins (10 points)
    max_score += 10
    profit_margin = data.get('profit_margins')
    if profit_margin:
        metrics['Profit Margin'] = {'value': f"{profit_margin*100:.1f}%", 'raw': profit_margin * 100}
        if profit_margin > 0.20:
            score += 10
            reasons.append(f"Exceptional margins: {profit_margin*100:.1f}%")
        elif profit_margin > 0.15:
            score += 7
            reasons.append(f"Strong margins: {profit_margin*100:.1f}%")
        elif profit_margin > 0.10:
            score += 4
            reasons.append(f"Good margins: {profit_margin*100:.1f}%")
        elif profit_margin < 0:
            red_flags.append("Negative profit margins")
    else:
        metrics['Profit Margin'] = {'value': 'N/A', 'raw': None}
    
    # Free Cash Flow Quality (10 points)
    max_score += 10
    fcf = data.get('free_cashflow')
    financials = data.get('financials')
    try:
        if financials is not None and not financials.empty:
            net_income = financials.loc['Net Income'].iloc[0] if 'Net Income' in financials.index else None
            if fcf and net_income and net_income > 0:
                cash_conversion = fcf / net_income
                metrics['Cash Conversion'] = {'value': f"{cash_conversion:.2f}x", 'raw': cash_conversion}
                if cash_conversion > 1.2:
                    score += 10
                    reasons.append(f"Excellent cash conversion: {cash_conversion:.2f}x")
                elif cash_conversion > 1.0:
                    score += 7
                    reasons.append(f"Strong cash conversion: {cash_conversion:.2f}x")
                elif cash_conversion > 0.8:
                    score += 4
                    reasons.append(f"Good cash conversion: {cash_conversion:.2f}x")
                elif cash_conversion < 0.5:
                    red_flags.append(f"Poor cash conversion: {cash_conversion:.2f}x")
            else:
                metrics['Cash Conversion'] = {'value': 'N/A', 'raw': None}
        else:
            metrics['Cash Conversion'] = {'value': 'N/A', 'raw': None}
    except:
        metrics['Cash Conversion'] = {'value': 'N/A', 'raw': None}
    
    # Earnings Consistency (10 points)
    max_score += 10
    try:
        if financials is not None and not financials.empty and 'Net Income' in financials.index:
            net_income_history = financials.loc['Net Income']
            if len(net_income_history) >= 3:
                negative_years = (net_income_history < 0).sum()
                earnings_growth = net_income_history.pct_change().dropna()
                earnings_volatility = earnings_growth.std() if len(earnings_growth) > 0 else 1.0
                
                metrics['Earnings Consistency'] = {'value': f"{4 - negative_years}/4 positive years", 'raw': 4 - negative_years}
                
                if negative_years == 0 and earnings_volatility < 0.3:
                    score += 10
                    reasons.append("Consistent earnings with low volatility")
                elif negative_years == 0:
                    score += 7
                    reasons.append("Positive earnings every year")
                elif negative_years <= 1:
                    score += 3
                    reasons.append("Mostly consistent earnings")
                else:
                    red_flags.append(f"Inconsistent earnings: {negative_years} loss years")
            else:
                metrics['Earnings Consistency'] = {'value': 'Limited data', 'raw': None}
        else:
            metrics['Earnings Consistency'] = {'value': 'N/A', 'raw': None}
    except:
        metrics['Earnings Consistency'] = {'value': 'N/A', 'raw': None}
    
    # === MOAT INDICATORS (30 points possible) ===
    
    # Gross Margin Strength (10 points)
    max_score += 10
    gross_margin = data.get('gross_margins')
    if gross_margin:
        metrics['Gross Margin'] = {'value': f"{gross_margin*100:.1f}%", 'raw': gross_margin * 100}
        if gross_margin > 0.50:
            score += 10
            reasons.append(f"Exceptional gross margins: {gross_margin*100:.1f}% (strong pricing power)")
        elif gross_margin > 0.40:
            score += 7
            reasons.append(f"Strong gross margins: {gross_margin*100:.1f}%")
        elif gross_margin > 0.30:
            score += 4
            reasons.append(f"Good gross margins: {gross_margin*100:.1f}%")
        elif gross_margin < 0.20:
            red_flags.append(f"Low gross margins: {gross_margin*100:.1f}%")
    else:
        metrics['Gross Margin'] = {'value': 'N/A', 'raw': None}
    
    # Capital Efficiency (10 points)
    max_score += 10
    try:
        cashflow_stmt = data.get('cashflow')
        if cashflow_stmt is not None and not cashflow_stmt.empty:
            operating_cf = None
            capex = None
            for key in ['Operating Cash Flow', 'Total Cash From Operating Activities']:
                if key in cashflow_stmt.index:
                    operating_cf = cashflow_stmt.loc[key].iloc[0]
                    break
            for key in ['Capital Expenditure', 'Capital Expenditures']:
                if key in cashflow_stmt.index:
                    capex = abs(cashflow_stmt.loc[key].iloc[0])
                    break
            
            if operating_cf and capex and operating_cf > 0:
                capex_ratio = capex / operating_cf
                metrics['CapEx/OCF'] = {'value': f"{capex_ratio*100:.0f}%", 'raw': capex_ratio * 100}
                if capex_ratio < 0.20:
                    score += 10
                    reasons.append(f"Asset-light business: {capex_ratio*100:.0f}% capex/OCF")
                elif capex_ratio < 0.35:
                    score += 6
                    reasons.append(f"Moderate capital needs: {capex_ratio*100:.0f}% capex/OCF")
                elif capex_ratio < 0.50:
                    score += 3
                elif capex_ratio > 0.70:
                    red_flags.append(f"Capital intensive: {capex_ratio*100:.0f}% capex/OCF")
            else:
                metrics['CapEx/OCF'] = {'value': 'N/A', 'raw': None}
        else:
            metrics['CapEx/OCF'] = {'value': 'N/A', 'raw': None}
    except:
        metrics['CapEx/OCF'] = {'value': 'N/A', 'raw': None}
    
    # Return on Invested Capital proxy (10 points)
    max_score += 10
    try:
        balance_sheet = data.get('balance_sheet')
        if financials is not None and balance_sheet is not None and not financials.empty and not balance_sheet.empty:
            operating_income = None
            for key in ['Operating Income', 'EBIT']:
                if key in financials.index:
                    operating_income = financials.loc[key].iloc[0]
                    break
            
            total_assets = balance_sheet.loc['Total Assets'].iloc[0] if 'Total Assets' in balance_sheet.index else None
            current_liabilities = None
            for key in ['Current Liabilities', 'Total Current Liabilities']:
                if key in balance_sheet.index:
                    current_liabilities = balance_sheet.loc[key].iloc[0]
                    break
            
            if operating_income and total_assets and current_liabilities:
                invested_capital = total_assets - current_liabilities
                if invested_capital > 0:
                    roic = operating_income / invested_capital
                    metrics['ROIC'] = {'value': f"{roic*100:.1f}%", 'raw': roic * 100}
                    if roic > 0.20:
                        score += 10
                        reasons.append(f"Excellent ROIC: {roic*100:.1f}%")
                    elif roic > 0.15:
                        score += 7
                        reasons.append(f"Strong ROIC: {roic*100:.1f}%")
                    elif roic > 0.10:
                        score += 4
                        reasons.append(f"Good ROIC: {roic*100:.1f}%")
                    elif roic < 0.05:
                        red_flags.append(f"Weak ROIC: {roic*100:.1f}%")
                else:
                    metrics['ROIC'] = {'value': 'N/A', 'raw': None}
            else:
                metrics['ROIC'] = {'value': 'N/A', 'raw': None}
        else:
            metrics['ROIC'] = {'value': 'N/A', 'raw': None}
    except:
        metrics['ROIC'] = {'value': 'N/A', 'raw': None}
    
    # === BALANCE SHEET (15 points possible) ===
    
    # Debt Level (10 points)
    max_score += 10
    debt_to_equity = data.get('debt_to_equity')
    if debt_to_equity is not None:
        metrics['Debt/Equity'] = {'value': f"{debt_to_equity:.2f}", 'raw': debt_to_equity}
        if debt_to_equity < 30:  # yfinance returns as percentage sometimes
            de_ratio = debt_to_equity / 100 if debt_to_equity > 5 else debt_to_equity
        else:
            de_ratio = debt_to_equity / 100
        
        if de_ratio < 0.3:
            score += 10
            reasons.append(f"Minimal debt: D/E = {de_ratio:.2f}")
        elif de_ratio < 0.5:
            score += 7
            reasons.append(f"Low debt: D/E = {de_ratio:.2f}")
        elif de_ratio < 1.0:
            score += 4
            reasons.append(f"Moderate debt: D/E = {de_ratio:.2f}")
        elif de_ratio > 2.0:
            red_flags.append(f"High debt: D/E = {de_ratio:.2f}")
    else:
        metrics['Debt/Equity'] = {'value': 'N/A', 'raw': None}
    
    # Liquidity (5 points)
    max_score += 5
    current_ratio = data.get('current_ratio')
    if current_ratio:
        metrics['Current Ratio'] = {'value': f"{current_ratio:.2f}", 'raw': current_ratio}
        if current_ratio > 2.0:
            score += 5
            reasons.append(f"Strong liquidity: CR = {current_ratio:.2f}")
        elif current_ratio > 1.5:
            score += 3
        elif current_ratio < 1.0:
            red_flags.append(f"Liquidity concern: CR = {current_ratio:.2f}")
    else:
        metrics['Current Ratio'] = {'value': 'N/A', 'raw': None}
    
    # === VALUATION (15 points possible) ===
    
    # P/E Ratio (10 points)
    max_score += 10
    trailing_pe = data.get('trailing_pe')
    peg = data.get('peg_ratio')
    if trailing_pe:
        metrics['P/E'] = {'value': f"{trailing_pe:.1f}", 'raw': trailing_pe}
        if peg:
            metrics['PEG'] = {'value': f"{peg:.2f}", 'raw': peg}
        
        if peg and peg < 1.0:
            score += 10
            reasons.append(f"Attractive valuation: P/E={trailing_pe:.1f}, PEG={peg:.2f}")
        elif peg and peg < 1.5:
            score += 7
            reasons.append(f"Fair valuation: P/E={trailing_pe:.1f}, PEG={peg:.2f}")
        elif trailing_pe < 20:
            score += 5
            reasons.append(f"Reasonable P/E: {trailing_pe:.1f}")
        elif trailing_pe < 25:
            score += 3
        elif trailing_pe > 40:
            red_flags.append(f"Expensive: P/E = {trailing_pe:.1f}")
    else:
        metrics['P/E'] = {'value': 'N/A', 'raw': None}
        metrics['PEG'] = {'value': 'N/A', 'raw': None}
    
    # Price to Book (5 points)
    max_score += 5
    price_to_book = data.get('price_to_book')
    if price_to_book:
        metrics['P/B'] = {'value': f"{price_to_book:.2f}", 'raw': price_to_book}
        if price_to_book < 3.0 and roe and roe > 0.15:
            score += 5
            reasons.append(f"Good value: P/B={price_to_book:.2f} with high ROE")
        elif price_to_book < 5.0 and roe and roe > 0.20:
            score += 3
        elif price_to_book > 10:
            red_flags.append(f"High P/B: {price_to_book:.2f}")
    else:
        metrics['P/B'] = {'value': 'N/A', 'raw': None}
    
    # Calculate final rating
    percentage_score = (score / max_score * 100) if max_score > 0 else 0
    
    if percentage_score >= 80:
        rating = "WONDERFUL BUSINESS"
        rating_color = "green"
        rating_emoji = "🟢"
    elif percentage_score >= 65:
        rating = "QUALITY BUSINESS"
        rating_color = "lightgreen"
        rating_emoji = "🟡"
    elif percentage_score >= 50:
        rating = "WATCH FOR BETTER PRICE"
        rating_color = "orange"
        rating_emoji = "🟠"
    elif percentage_score >= 35:
        rating = "LIKELY PASS"
        rating_color = "red"
        rating_emoji = "🔴"
    else:
        rating = "AVOID"
        rating_color = "darkred"
        rating_emoji = "⛔"
    
    return {
        'score': score,
        'max_score': max_score,
        'percentage': percentage_score,
        'rating': rating,
        'rating_color': rating_color,
        'rating_emoji': rating_emoji,
        'metrics': metrics,
        'reasons': reasons,
        'red_flags': red_flags
    }


def evaluate_druckenmiller(data: dict) -> dict:
    """
    Evaluate a stock using Stanley Druckenmiller's methodology.
    100-point scale focused on momentum, growth, and macro alignment.
    """
    metrics = {}
    score = 0
    max_score = 0
    reasons = []
    red_flags = []
    
    # === TECHNICAL MOMENTUM (30 points) ===
    
    # Price vs Moving Averages (10 points)
    max_score += 10
    current_price = data.get('current_price')
    sma_50 = data.get('fifty_day_average')
    sma_200 = data.get('two_hundred_day_average')
    
    if sma_50 and sma_200 and current_price:
        metrics['50 DMA'] = {'value': f"${sma_50:.2f}", 'raw': sma_50}
        metrics['200 DMA'] = {'value': f"${sma_200:.2f}", 'raw': sma_200}
        
        above_50 = current_price > sma_50
        above_200 = current_price > sma_200
        golden_cross = sma_50 > sma_200
        
        if above_50 and above_200 and golden_cross:
            score += 10
            reasons.append("Strong uptrend: Price above 50 & 200 DMA with golden cross")
        elif above_50 and above_200:
            score += 7
            reasons.append("Uptrend: Price above key moving averages")
        elif above_50:
            score += 4
            reasons.append("Short-term momentum: Above 50 DMA")
        elif not above_200:
            red_flags.append("Below 200 DMA - weak long-term trend")
    else:
        metrics['50 DMA'] = {'value': 'N/A', 'raw': None}
        metrics['200 DMA'] = {'value': 'N/A', 'raw': None}
    
    # 52-Week High Proximity (10 points)
    max_score += 10
    fifty_two_high = data.get('fifty_two_week_high')
    if fifty_two_high and current_price:
        pct_of_high = (current_price / fifty_two_high) * 100
        metrics['% of 52W High'] = {'value': f"{pct_of_high:.0f}%", 'raw': pct_of_high}
        
        if pct_of_high > 95:
            score += 10
            reasons.append(f"Near all-time high: {pct_of_high:.0f}% - strong momentum")
        elif pct_of_high > 85:
            score += 6
            reasons.append(f"Within striking distance of highs: {pct_of_high:.0f}%")
        elif pct_of_high > 70:
            score += 3
        elif pct_of_high < 60:
            red_flags.append(f"Far from highs: {pct_of_high:.0f}% of 52W high")
    else:
        metrics['% of 52W High'] = {'value': 'N/A', 'raw': None}
    
    # Institutional Ownership (10 points)
    max_score += 10
    institutional = data.get('held_percent_institutions')
    if institutional:
        inst_pct = institutional * 100
        metrics['Institutional %'] = {'value': f"{inst_pct:.0f}%", 'raw': inst_pct}
        
        if 40 <= inst_pct <= 80:
            score += 10
            reasons.append(f"Healthy institutional interest: {inst_pct:.0f}%")
        elif 30 <= inst_pct < 90:
            score += 6
        elif inst_pct > 90:
            red_flags.append(f"Crowded ownership: {inst_pct:.0f}% institutional")
    else:
        metrics['Institutional %'] = {'value': 'N/A', 'raw': None}
    
    # === GROWTH & EARNINGS MOMENTUM (30 points) ===
    
    # Revenue Growth (10 points)
    max_score += 10
    financials = data.get('financials')
    try:
        if financials is not None and not financials.empty and 'Total Revenue' in financials.index:
            revenue = financials.loc['Total Revenue']
            if len(revenue) >= 2:
                revenue_growth = (revenue.iloc[0] / revenue.iloc[1] - 1) * 100
                metrics['Revenue Growth'] = {'value': f"{revenue_growth:.1f}%", 'raw': revenue_growth}
                
                if revenue_growth > 40:
                    score += 10
                    reasons.append(f"Explosive revenue growth: {revenue_growth:.1f}%")
                elif revenue_growth > 25:
                    score += 7
                    reasons.append(f"Strong revenue growth: {revenue_growth:.1f}%")
                elif revenue_growth > 15:
                    score += 4
                    reasons.append(f"Solid revenue growth: {revenue_growth:.1f}%")
                elif revenue_growth < 0:
                    red_flags.append(f"Revenue declining: {revenue_growth:.1f}%")
            else:
                metrics['Revenue Growth'] = {'value': 'N/A', 'raw': None}
        else:
            metrics['Revenue Growth'] = {'value': 'N/A', 'raw': None}
    except:
        metrics['Revenue Growth'] = {'value': 'N/A', 'raw': None}
    
    # Earnings Momentum (10 points)
    max_score += 10
    forward_pe = data.get('forward_pe')
    trailing_pe = data.get('trailing_pe')
    
    if forward_pe and trailing_pe and forward_pe > 0 and trailing_pe > 0:
        eps_momentum = ((trailing_pe / forward_pe) - 1) * 100
        metrics['EPS Momentum'] = {'value': f"{eps_momentum:+.1f}%", 'raw': eps_momentum}
        
        if eps_momentum > 20:
            score += 10
            reasons.append(f"Strong earnings momentum: estimates up {eps_momentum:.0f}%")
        elif eps_momentum > 10:
            score += 7
            reasons.append(f"Good earnings momentum: +{eps_momentum:.0f}%")
        elif eps_momentum > 0:
            score += 4
            reasons.append(f"Positive estimate revisions: +{eps_momentum:.0f}%")
        elif eps_momentum < -10:
            red_flags.append(f"Negative earnings revisions: {eps_momentum:.0f}%")
    else:
        metrics['EPS Momentum'] = {'value': 'N/A', 'raw': None}
    
    # Operating Leverage (10 points)
    max_score += 10
    operating_margin = data.get('operating_margins')
    if operating_margin:
        op_margin_pct = operating_margin * 100
        metrics['Operating Margin'] = {'value': f"{op_margin_pct:.1f}%", 'raw': op_margin_pct}
        
        try:
            if financials is not None and not financials.empty:
                if 'Operating Income' in financials.index and 'Total Revenue' in financials.index:
                    op_income = financials.loc['Operating Income']
                    revenue = financials.loc['Total Revenue']
                    if len(op_income) >= 2 and len(revenue) >= 2:
                        recent_margin = op_income.iloc[0] / revenue.iloc[0]
                        old_margin = op_income.iloc[-1] / revenue.iloc[-1]
                        margin_expansion = (recent_margin - old_margin) * 100
                        
                        metrics['Margin Trend'] = {'value': f"{margin_expansion:+.1f}pp", 'raw': margin_expansion}
                        
                        if margin_expansion > 3:
                            score += 10
                            reasons.append(f"Expanding margins: +{margin_expansion:.1f}pp")
                        elif margin_expansion > 0:
                            score += 6
                            reasons.append(f"Margin expansion: +{margin_expansion:.1f}pp")
                        elif margin_expansion < -3:
                            red_flags.append(f"Margin compression: {margin_expansion:.1f}pp")
        except:
            pass
        
        if operating_margin > 0.25:
            score += 3
        elif operating_margin > 0.15:
            score += 2
    else:
        metrics['Operating Margin'] = {'value': 'N/A', 'raw': None}
    
    # === VALUATION & RISK/REWARD (20 points) ===
    
    # PEG Ratio (10 points)
    max_score += 10
    peg_ratio = data.get('peg_ratio')
    if peg_ratio and peg_ratio > 0:
        metrics['PEG'] = {'value': f"{peg_ratio:.2f}", 'raw': peg_ratio}
        
        if peg_ratio < 1.0:
            score += 10
            reasons.append(f"Attractive valuation: PEG = {peg_ratio:.2f}")
        elif peg_ratio < 1.5:
            score += 7
            reasons.append(f"Reasonable valuation: PEG = {peg_ratio:.2f}")
        elif peg_ratio < 2.0:
            score += 4
        elif peg_ratio > 3.0:
            red_flags.append(f"Expensive relative to growth: PEG = {peg_ratio:.2f}")
    else:
        metrics['PEG'] = {'value': 'N/A', 'raw': None}
    
    # Forward P/E (10 points)
    max_score += 10
    if forward_pe and forward_pe > 0:
        metrics['Forward P/E'] = {'value': f"{forward_pe:.1f}", 'raw': forward_pe}
        
        if forward_pe < 15:
            score += 10
            reasons.append(f"Cheap on earnings: Forward P/E = {forward_pe:.1f}")
        elif forward_pe < 25:
            score += 6
        elif forward_pe < 35:
            score += 3
        elif forward_pe > 50:
            red_flags.append(f"Very expensive: Forward P/E = {forward_pe:.1f}")
    else:
        metrics['Forward P/E'] = {'value': 'N/A', 'raw': None}
    
    # === QUALITY CHECK (20 points) ===
    
    # Profitability (10 points)
    max_score += 10
    profit_margin = data.get('profit_margins')
    if profit_margin:
        profit_pct = profit_margin * 100
        metrics['Profit Margin'] = {'value': f"{profit_pct:.1f}%", 'raw': profit_pct}
        
        if profit_margin > 0.20:
            score += 10
            reasons.append(f"Excellent profitability: {profit_pct:.1f}% margins")
        elif profit_margin > 0.10:
            score += 6
        elif profit_margin > 0.05:
            score += 3
        elif profit_margin < 0:
            red_flags.append("Unprofitable - proceed with caution")
    else:
        metrics['Profit Margin'] = {'value': 'N/A', 'raw': None}
    
    # Market Cap (10 points - preference for leaders)
    max_score += 10
    market_cap = data.get('market_cap')
    if market_cap:
        metrics['Market Cap'] = {'value': format_currency(market_cap), 'raw': market_cap}
        
        if market_cap > 100e9:
            score += 10
            reasons.append(f"Large cap leader: {format_currency(market_cap)}")
        elif market_cap > 20e9:
            score += 7
            reasons.append(f"Established mid-cap: {format_currency(market_cap)}")
        elif market_cap > 5e9:
            score += 4
        elif market_cap < 1e9:
            red_flags.append("Small cap - higher risk")
    else:
        metrics['Market Cap'] = {'value': 'N/A', 'raw': None}
    
    # Calculate final rating
    percentage_score = (score / max_score * 100) if max_score > 0 else 0
    
    if percentage_score >= 80:
        rating = "HIGH CONVICTION BUY"
        rating_color = "green"
        rating_emoji = "🟢"
        position_size = "15-30%"
    elif percentage_score >= 65:
        rating = "STRONG BUY"
        rating_color = "lightgreen"
        rating_emoji = "🟡"
        position_size = "10-20%"
    elif percentage_score >= 50:
        rating = "BUY"
        rating_color = "orange"
        rating_emoji = "🟠"
        position_size = "5-10%"
    elif percentage_score >= 35:
        rating = "WATCH"
        rating_color = "red"
        rating_emoji = "🔴"
        position_size = "2-5%"
    else:
        rating = "PASS"
        rating_color = "darkred"
        rating_emoji = "⛔"
        position_size = "0%"
    
    return {
        'score': score,
        'max_score': max_score,
        'percentage': percentage_score,
        'rating': rating,
        'rating_color': rating_color,
        'rating_emoji': rating_emoji,
        'metrics': metrics,
        'reasons': reasons,
        'red_flags': red_flags,
        'position_size': position_size
    }


def evaluate_roaring_kitty(data: dict) -> dict:
    """
    Evaluate a stock using Roaring Kitty's methodology.
    10-point scale focused on deep value + short squeeze potential.
    """
    metrics = {}
    score = 0
    max_score = 13  # Maximum possible score
    reasons = []
    red_flags = []
    
    # === VALUATION METRICS ===
    
    # Price-to-Book (2 points)
    price_to_book = data.get('price_to_book')
    if price_to_book:
        metrics['P/B'] = {'value': f"{price_to_book:.2f}", 'raw': price_to_book}
        if price_to_book < 1.0:
            score += 2
            reasons.append(f"Deep value: P/B = {price_to_book:.2f}")
        elif price_to_book < 1.5:
            score += 1
            reasons.append(f"Value territory: P/B = {price_to_book:.2f}")
        elif price_to_book > 5.0:
            red_flags.append(f"Expensive on book value: P/B = {price_to_book:.2f}")
    else:
        metrics['P/B'] = {'value': 'N/A', 'raw': None}
    
    # Trading Below Net Cash (3 points)
    net_cash = data.get('net_cash')
    market_cap = data.get('market_cap')
    if net_cash is not None and market_cap:
        metrics['Net Cash'] = {'value': format_currency(net_cash), 'raw': net_cash}
        if net_cash > 0 and market_cap < net_cash:
            score += 3
            reasons.append("🔥 Trading below net cash!")
        elif net_cash > 0 and market_cap < net_cash * 2:
            score += 1
            reasons.append("Strong cash position relative to market cap")
    else:
        metrics['Net Cash'] = {'value': 'N/A', 'raw': None}
    
    # FCF Yield (2 points)
    fcf_yield = data.get('fcf_yield')
    if fcf_yield is not None:
        metrics['FCF Yield'] = {'value': f"{fcf_yield:.1f}%", 'raw': fcf_yield}
        if fcf_yield > 15:
            score += 2
            reasons.append(f"Exceptional FCF yield: {fcf_yield:.1f}%")
        elif fcf_yield > 10:
            score += 1
            reasons.append(f"Strong FCF yield: {fcf_yield:.1f}%")
        elif fcf_yield < 0:
            red_flags.append(f"Negative FCF yield: {fcf_yield:.1f}%")
    else:
        metrics['FCF Yield'] = {'value': 'N/A', 'raw': None}
    
    # === BALANCE SHEET STRENGTH ===
    
    # Current Ratio (1 point)
    current_ratio = data.get('current_ratio')
    if current_ratio:
        metrics['Current Ratio'] = {'value': f"{current_ratio:.2f}", 'raw': current_ratio}
        if current_ratio > 2.0:
            score += 1
            reasons.append(f"Strong liquidity: CR = {current_ratio:.2f}")
        elif current_ratio < 1.0:
            red_flags.append(f"Liquidity risk: CR = {current_ratio:.2f}")
    else:
        metrics['Current Ratio'] = {'value': 'N/A', 'raw': None}
    
    # Debt-to-Equity (1 point implicit in not having red flag)
    debt_to_equity = data.get('debt_to_equity')
    if debt_to_equity is not None:
        de_ratio = debt_to_equity / 100 if debt_to_equity > 5 else debt_to_equity
        metrics['D/E'] = {'value': f"{de_ratio:.2f}", 'raw': de_ratio}
        if de_ratio > 1.0:
            red_flags.append(f"High debt: D/E = {de_ratio:.2f}")
    else:
        metrics['D/E'] = {'value': 'N/A', 'raw': None}
    
    # === SHORT INTEREST (Critical for RK Strategy) ===
    
    # Short % of Float (2 points base + 2 bonus for extreme)
    short_pct = data.get('short_percent_float')
    if short_pct:
        short_pct_display = short_pct * 100 if short_pct < 1 else short_pct
        metrics['Short % Float'] = {'value': f"{short_pct_display:.1f}%", 'raw': short_pct_display}
        if short_pct_display > 40:
            score += 4
            reasons.append(f"🚀 EXTREME short interest: {short_pct_display:.1f}% - squeeze potential!")
        elif short_pct_display > 20:
            score += 2
            reasons.append(f"High short interest: {short_pct_display:.1f}% of float")
        elif short_pct_display > 10:
            score += 1
            reasons.append(f"Elevated short interest: {short_pct_display:.1f}%")
    else:
        metrics['Short % Float'] = {'value': 'N/A', 'raw': None}
    
    # Days to Cover (1 point)
    short_ratio = data.get('short_ratio')
    if short_ratio:
        metrics['Days to Cover'] = {'value': f"{short_ratio:.1f} days", 'raw': short_ratio}
        if short_ratio > 5:
            score += 1
            reasons.append(f"High days to cover: {short_ratio:.1f} days")
    else:
        metrics['Days to Cover'] = {'value': 'N/A', 'raw': None}
    
    # === PRICE POSITIONING ===
    
    # Near 52-Week Low (1 point)
    pct_from_high = data.get('pct_from_high')
    if pct_from_high is not None:
        metrics['% From 52W High'] = {'value': f"{pct_from_high:.0f}%", 'raw': pct_from_high}
        if pct_from_high > 50:
            score += 1
            reasons.append(f"Near 52-week low: {pct_from_high:.0f}% off high")
        elif pct_from_high > 30:
            reasons.append(f"Pulled back: {pct_from_high:.0f}% off high")
    else:
        metrics['% From 52W High'] = {'value': 'N/A', 'raw': None}
    
    # === ADDITIONAL CONTEXT ===
    
    # Insider Ownership
    insider_pct = data.get('held_percent_insiders')
    if insider_pct:
        metrics['Insider Ownership'] = {'value': f"{insider_pct*100:.1f}%", 'raw': insider_pct * 100}
        if insider_pct > 0.10:
            reasons.append(f"Significant insider ownership: {insider_pct*100:.1f}%")
    else:
        metrics['Insider Ownership'] = {'value': 'N/A', 'raw': None}
    
    # Market Cap (context)
    if market_cap:
        metrics['Market Cap'] = {'value': format_currency(market_cap), 'raw': market_cap}
        if market_cap < 5e9:  # Under $5B - RK's sweet spot
            reasons.append(f"Small/mid-cap: {format_currency(market_cap)} (potentially overlooked)")
    
    # Calculate final rating
    percentage_score = (score / max_score * 100) if max_score > 0 else 0
    
    if score >= 8:
        rating = "STRONG BUY"
        rating_color = "green"
        rating_emoji = "🟢"
    elif score >= 5:
        rating = "BUY"
        rating_color = "lightgreen"
        rating_emoji = "🟡"
    elif score >= 3:
        rating = "WATCH"
        rating_color = "orange"
        rating_emoji = "🟠"
    else:
        rating = "PASS"
        rating_color = "red"
        rating_emoji = "🔴"
    
    return {
        'score': score,
        'max_score': max_score,
        'percentage': percentage_score,
        'rating': rating,
        'rating_color': rating_color,
        'rating_emoji': rating_emoji,
        'metrics': metrics,
        'reasons': reasons,
        'red_flags': red_flags
    }


def evaluate_greenblatt(data: dict) -> dict:
    """
    Evaluate a stock using Joel Greenblatt's Magic Formula methodology.
    Focuses on Return on Capital (ROC) and Earnings Yield.
    100-point scale.
    """
    metrics = {}
    score = 0
    max_score = 100
    reasons = []
    red_flags = []
    
    # === RETURN ON CAPITAL (40 points) ===
    # ROC = EBIT / (Net Working Capital + Net Fixed Assets)
    # Using ROE as proxy since yfinance doesn't give us exact components
    
    roe = data.get('return_on_equity')
    operating_margin = data.get('operating_margins')
    profit_margin = data.get('profit_margins')
    
    # ROE scoring (20 points)
    if roe is not None:
        roe_pct = roe * 100 if roe < 1 else roe
        metrics['ROE'] = {'value': f"{roe_pct:.1f}%", 'raw': roe_pct}
        if roe_pct > 25:
            score += 20
            reasons.append(f"Excellent ROE: {roe_pct:.1f}% (high return on capital)")
        elif roe_pct > 15:
            score += 15
            reasons.append(f"Strong ROE: {roe_pct:.1f}%")
        elif roe_pct > 10:
            score += 10
            reasons.append(f"Decent ROE: {roe_pct:.1f}%")
        elif roe_pct > 5:
            score += 5
        elif roe_pct < 0:
            red_flags.append(f"Negative ROE: {roe_pct:.1f}%")
    else:
        metrics['ROE'] = {'value': 'N/A', 'raw': None}
    
    # Operating Margin (10 points) - proxy for business quality
    if operating_margin is not None:
        op_margin_pct = operating_margin * 100 if operating_margin < 1 else operating_margin
        metrics['Operating Margin'] = {'value': f"{op_margin_pct:.1f}%", 'raw': op_margin_pct}
        if op_margin_pct > 25:
            score += 10
            reasons.append(f"Exceptional operating margin: {op_margin_pct:.1f}%")
        elif op_margin_pct > 15:
            score += 7
            reasons.append(f"Strong operating margin: {op_margin_pct:.1f}%")
        elif op_margin_pct > 10:
            score += 5
        elif op_margin_pct > 5:
            score += 3
        elif op_margin_pct < 0:
            red_flags.append(f"Negative operating margin: {op_margin_pct:.1f}%")
    else:
        metrics['Operating Margin'] = {'value': 'N/A', 'raw': None}
    
    # ROIC if available (10 points)
    # Approximate using profit margin * asset turnover
    total_revenue = data.get('total_revenue')
    total_assets = data.get('total_assets')
    if total_revenue and total_assets and profit_margin:
        asset_turnover = total_revenue / total_assets if total_assets > 0 else 0
        roic_proxy = (profit_margin * asset_turnover) * 100
        metrics['ROIC (est)'] = {'value': f"{roic_proxy:.1f}%", 'raw': roic_proxy}
        if roic_proxy > 20:
            score += 10
            reasons.append(f"High capital efficiency (ROIC ~{roic_proxy:.1f}%)")
        elif roic_proxy > 12:
            score += 7
        elif roic_proxy > 8:
            score += 4
    else:
        metrics['ROIC (est)'] = {'value': 'N/A', 'raw': None}
    
    # === EARNINGS YIELD (40 points) ===
    # Earnings Yield = EBIT / Enterprise Value
    
    enterprise_value = data.get('enterprise_value')
    market_cap = data.get('market_cap')
    trailing_pe = data.get('trailing_pe')
    forward_pe = data.get('forward_pe')
    
    # Earnings Yield from P/E (20 points)
    if trailing_pe and trailing_pe > 0:
        earnings_yield = (1 / trailing_pe) * 100
        metrics['Earnings Yield'] = {'value': f"{earnings_yield:.1f}%", 'raw': earnings_yield}
        if earnings_yield > 15:
            score += 20
            reasons.append(f"🔥 Exceptional earnings yield: {earnings_yield:.1f}%")
        elif earnings_yield > 10:
            score += 15
            reasons.append(f"High earnings yield: {earnings_yield:.1f}%")
        elif earnings_yield > 7:
            score += 10
            reasons.append(f"Good earnings yield: {earnings_yield:.1f}%")
        elif earnings_yield > 5:
            score += 5
        elif earnings_yield < 3:
            red_flags.append(f"Low earnings yield: {earnings_yield:.1f}% (expensive)")
    else:
        metrics['Earnings Yield'] = {'value': 'N/A', 'raw': None}
    
    # EV/EBIT proxy using EV/EBITDA (10 points)
    ev_to_ebitda = data.get('ev_to_ebitda')
    if ev_to_ebitda and ev_to_ebitda > 0:
        metrics['EV/EBITDA'] = {'value': f"{ev_to_ebitda:.1f}x", 'raw': ev_to_ebitda}
        if ev_to_ebitda < 8:
            score += 10
            reasons.append(f"Cheap on EV/EBITDA: {ev_to_ebitda:.1f}x")
        elif ev_to_ebitda < 12:
            score += 7
            reasons.append(f"Reasonable EV/EBITDA: {ev_to_ebitda:.1f}x")
        elif ev_to_ebitda < 15:
            score += 4
        elif ev_to_ebitda > 20:
            red_flags.append(f"Expensive EV/EBITDA: {ev_to_ebitda:.1f}x")
    else:
        metrics['EV/EBITDA'] = {'value': 'N/A', 'raw': None}
    
    # Forward P/E discount (10 points)
    if forward_pe and trailing_pe and forward_pe > 0 and trailing_pe > 0:
        pe_improvement = ((trailing_pe - forward_pe) / trailing_pe) * 100
        metrics['Forward P/E'] = {'value': f"{forward_pe:.1f}x", 'raw': forward_pe}
        if pe_improvement > 20:
            score += 10
            reasons.append(f"Earnings expected to grow: Forward P/E {forward_pe:.1f}x vs Trailing {trailing_pe:.1f}x")
        elif pe_improvement > 10:
            score += 6
        elif pe_improvement > 0:
            score += 3
    elif forward_pe:
        metrics['Forward P/E'] = {'value': f"{forward_pe:.1f}x", 'raw': forward_pe}
    else:
        metrics['Forward P/E'] = {'value': 'N/A', 'raw': None}
    
    # === QUALITY CHECKS (20 points) ===
    
    # Consistent profitability - positive earnings (5 points)
    eps = data.get('trailing_eps')
    if eps and eps > 0:
        score += 5
        metrics['EPS'] = {'value': f"${eps:.2f}", 'raw': eps}
    elif eps and eps < 0:
        red_flags.append(f"Negative EPS: ${eps:.2f}")
        metrics['EPS'] = {'value': f"${eps:.2f}", 'raw': eps}
    else:
        metrics['EPS'] = {'value': 'N/A', 'raw': None}
    
    # Debt check - Greenblatt prefers low debt (5 points)
    debt_to_equity = data.get('debt_to_equity')
    if debt_to_equity is not None:
        de_ratio = debt_to_equity / 100 if debt_to_equity > 5 else debt_to_equity
        metrics['D/E'] = {'value': f"{de_ratio:.2f}", 'raw': de_ratio}
        if de_ratio < 0.3:
            score += 5
            reasons.append(f"Low debt: D/E = {de_ratio:.2f}")
        elif de_ratio < 0.7:
            score += 3
        elif de_ratio > 1.5:
            red_flags.append(f"High debt: D/E = {de_ratio:.2f}")
    else:
        metrics['D/E'] = {'value': 'N/A', 'raw': None}
    
    # Market cap - Greenblatt found formula works better in small/mid caps (5 points)
    if market_cap:
        metrics['Market Cap'] = {'value': format_currency(market_cap), 'raw': market_cap}
        if market_cap < 2e9:  # Under $2B
            score += 5
            reasons.append(f"Small cap: {format_currency(market_cap)} (potentially underfollowed)")
        elif market_cap < 10e9:  # Under $10B
            score += 3
            reasons.append(f"Mid cap: {format_currency(market_cap)}")
        elif market_cap > 100e9:
            reasons.append(f"Mega cap: {format_currency(market_cap)} (formula less effective)")
    
    # Sector exclusion check (5 points if not excluded sector)
    sector = data.get('sector')
    if sector:
        metrics['Sector'] = {'value': sector, 'raw': sector}
        if sector in ['Financial Services', 'Utilities']:
            red_flags.append(f"Excluded sector: {sector} (different capital structure)")
        else:
            score += 5
    
    # Calculate final rating
    percentage_score = (score / max_score * 100) if max_score > 0 else 0
    
    if percentage_score >= 70:
        rating = "STRONG BUY"
        rating_color = "green"
        rating_emoji = "🟢"
    elif percentage_score >= 55:
        rating = "BUY"
        rating_color = "lightgreen"
        rating_emoji = "🟡"
    elif percentage_score >= 40:
        rating = "HOLD"
        rating_color = "orange"
        rating_emoji = "🟠"
    else:
        rating = "PASS"
        rating_color = "red"
        rating_emoji = "🔴"
    
    return {
        'score': score,
        'max_score': max_score,
        'percentage': percentage_score,
        'rating': rating,
        'rating_color': rating_color,
        'rating_emoji': rating_emoji,
        'metrics': metrics,
        'reasons': reasons,
        'red_flags': red_flags
    }


def evaluate_dalio(data: dict) -> dict:
    """
    Evaluate a stock using Ray Dalio's All Weather / macro perspective.
    Focuses on how well a stock fits different economic environments and portfolio balance.
    100-point scale.
    """
    metrics = {}
    score = 0
    max_score = 100
    reasons = []
    red_flags = []
    
    # === ENVIRONMENT FIT (30 points) ===
    # Dalio thinks about 4 environments: Rising/Falling Growth × Rising/Falling Inflation
    
    sector = data.get('sector')
    industry = data.get('industry')
    beta = data.get('beta')
    dividend_yield = data.get('dividend_yield')
    
    # Beta assessment - Dalio likes understanding risk contribution (10 points)
    if beta is not None:
        metrics['Beta'] = {'value': f"{beta:.2f}", 'raw': beta}
        if 0.8 <= beta <= 1.2:
            score += 10
            reasons.append(f"Market-aligned beta: {beta:.2f} (predictable risk)")
        elif beta < 0.8:
            score += 8
            reasons.append(f"Defensive beta: {beta:.2f} (lower risk contribution)")
        elif beta > 1.5:
            score += 4
            red_flags.append(f"High beta: {beta:.2f} (concentrated risk)")
        else:
            score += 6
    else:
        metrics['Beta'] = {'value': 'N/A', 'raw': None}
    
    # Sector environment fit (10 points)
    if sector:
        metrics['Sector'] = {'value': sector, 'raw': sector}
        # Sectors that perform in multiple environments
        balanced_sectors = ['Consumer Defensive', 'Healthcare', 'Utilities']
        growth_sectors = ['Technology', 'Consumer Cyclical', 'Communication Services']
        inflation_sectors = ['Energy', 'Basic Materials', 'Real Estate']
        
        if sector in balanced_sectors:
            score += 10
            reasons.append(f"Defensive sector: {sector} (performs in multiple environments)")
        elif sector in growth_sectors:
            score += 6
            reasons.append(f"Growth sector: {sector} (best in rising growth/falling inflation)")
        elif sector in inflation_sectors:
            score += 7
            reasons.append(f"Inflation-linked sector: {sector} (hedge against rising inflation)")
        else:
            score += 5
    
    # Dividend yield - income across environments (10 points)
    if dividend_yield is not None:
        div_pct = dividend_yield * 100 if dividend_yield < 1 else dividend_yield
        metrics['Dividend Yield'] = {'value': f"{div_pct:.2f}%", 'raw': div_pct}
        if div_pct > 4:
            score += 10
            reasons.append(f"Strong dividend: {div_pct:.2f}% (income in any environment)")
        elif div_pct > 2:
            score += 7
            reasons.append(f"Meaningful dividend: {div_pct:.2f}%")
        elif div_pct > 0.5:
            score += 4
    else:
        metrics['Dividend Yield'] = {'value': 'N/A', 'raw': None}
    
    # === BALANCE SHEET STRENGTH (25 points) ===
    # Dalio emphasizes surviving different environments
    
    current_ratio = data.get('current_ratio')
    debt_to_equity = data.get('debt_to_equity')
    total_cash = data.get('total_cash')
    total_debt = data.get('total_debt')
    
    # Current ratio - liquidity (8 points)
    if current_ratio:
        metrics['Current Ratio'] = {'value': f"{current_ratio:.2f}", 'raw': current_ratio}
        if current_ratio > 2.0:
            score += 8
            reasons.append(f"Strong liquidity: CR = {current_ratio:.2f}")
        elif current_ratio > 1.5:
            score += 6
        elif current_ratio > 1.0:
            score += 3
        elif current_ratio < 1.0:
            red_flags.append(f"Liquidity concern: CR = {current_ratio:.2f}")
    else:
        metrics['Current Ratio'] = {'value': 'N/A', 'raw': None}
    
    # Debt level - survivability (9 points)
    if debt_to_equity is not None:
        de_ratio = debt_to_equity / 100 if debt_to_equity > 5 else debt_to_equity
        metrics['D/E'] = {'value': f"{de_ratio:.2f}", 'raw': de_ratio}
        if de_ratio < 0.3:
            score += 9
            reasons.append(f"Very low debt: D/E = {de_ratio:.2f} (survives any environment)")
        elif de_ratio < 0.7:
            score += 6
        elif de_ratio < 1.0:
            score += 3
        elif de_ratio > 1.5:
            red_flags.append(f"High leverage: D/E = {de_ratio:.2f} (vulnerable in downturns)")
    else:
        metrics['D/E'] = {'value': 'N/A', 'raw': None}
    
    # Cash position (8 points)
    if total_cash and total_debt:
        net_cash = total_cash - total_debt
        metrics['Net Cash'] = {'value': format_currency(net_cash), 'raw': net_cash}
        if net_cash > 0:
            score += 8
            reasons.append(f"Net cash position: {format_currency(net_cash)}")
        elif total_cash > total_debt * 0.5:
            score += 4
    elif total_cash:
        metrics['Cash'] = {'value': format_currency(total_cash), 'raw': total_cash}
    
    # === QUALITY & CONSISTENCY (25 points) ===
    # Dalio values predictable, high-quality earnings
    
    profit_margin = data.get('profit_margins')
    revenue_growth = data.get('revenue_growth')
    earnings_growth = data.get('earnings_growth')
    
    # Profit margin stability (10 points)
    if profit_margin is not None:
        margin_pct = profit_margin * 100 if profit_margin < 1 else profit_margin
        metrics['Profit Margin'] = {'value': f"{margin_pct:.1f}%", 'raw': margin_pct}
        if margin_pct > 20:
            score += 10
            reasons.append(f"High profit margin: {margin_pct:.1f}% (pricing power)")
        elif margin_pct > 10:
            score += 7
        elif margin_pct > 5:
            score += 4
        elif margin_pct < 0:
            red_flags.append(f"Negative margin: {margin_pct:.1f}%")
    else:
        metrics['Profit Margin'] = {'value': 'N/A', 'raw': None}
    
    # Revenue growth (8 points)
    if revenue_growth is not None:
        rev_growth_pct = revenue_growth * 100 if abs(revenue_growth) < 2 else revenue_growth
        metrics['Revenue Growth'] = {'value': f"{rev_growth_pct:.1f}%", 'raw': rev_growth_pct}
        if rev_growth_pct > 15:
            score += 8
            reasons.append(f"Strong revenue growth: {rev_growth_pct:.1f}%")
        elif rev_growth_pct > 5:
            score += 5
        elif rev_growth_pct > 0:
            score += 3
        elif rev_growth_pct < -10:
            red_flags.append(f"Revenue declining: {rev_growth_pct:.1f}%")
    else:
        metrics['Revenue Growth'] = {'value': 'N/A', 'raw': None}
    
    # Earnings growth (7 points)
    if earnings_growth is not None:
        earn_growth_pct = earnings_growth * 100 if abs(earnings_growth) < 2 else earnings_growth
        metrics['Earnings Growth'] = {'value': f"{earn_growth_pct:.1f}%", 'raw': earn_growth_pct}
        if earn_growth_pct > 20:
            score += 7
        elif earn_growth_pct > 10:
            score += 5
        elif earn_growth_pct > 0:
            score += 3
    else:
        metrics['Earnings Growth'] = {'value': 'N/A', 'raw': None}
    
    # === VALUATION (20 points) ===
    # Dalio isn't a pure value investor but doesn't ignore price
    
    trailing_pe = data.get('trailing_pe')
    peg_ratio = data.get('peg_ratio')
    price_to_book = data.get('price_to_book')
    
    # P/E reasonableness (8 points)
    if trailing_pe and trailing_pe > 0:
        metrics['P/E'] = {'value': f"{trailing_pe:.1f}x", 'raw': trailing_pe}
        if trailing_pe < 15:
            score += 8
            reasons.append(f"Attractive P/E: {trailing_pe:.1f}x")
        elif trailing_pe < 22:
            score += 5
        elif trailing_pe < 30:
            score += 2
        elif trailing_pe > 40:
            red_flags.append(f"High P/E: {trailing_pe:.1f}x")
    else:
        metrics['P/E'] = {'value': 'N/A', 'raw': None}
    
    # PEG ratio (7 points)
    if peg_ratio and peg_ratio > 0:
        metrics['PEG'] = {'value': f"{peg_ratio:.2f}", 'raw': peg_ratio}
        if peg_ratio < 1.0:
            score += 7
            reasons.append(f"Attractive PEG: {peg_ratio:.2f}")
        elif peg_ratio < 1.5:
            score += 4
        elif peg_ratio < 2.0:
            score += 2
    else:
        metrics['PEG'] = {'value': 'N/A', 'raw': None}
    
    # P/B (5 points)
    if price_to_book and price_to_book > 0:
        metrics['P/B'] = {'value': f"{price_to_book:.2f}x", 'raw': price_to_book}
        if price_to_book < 2:
            score += 5
        elif price_to_book < 4:
            score += 3
        elif price_to_book > 10:
            red_flags.append(f"High P/B: {price_to_book:.2f}x")
    else:
        metrics['P/B'] = {'value': 'N/A', 'raw': None}
    
    # Calculate final rating
    percentage_score = (score / max_score * 100) if max_score > 0 else 0
    
    # Dalio ratings reflect portfolio fit, not just buy/sell
    if percentage_score >= 70:
        rating = "CORE HOLDING"
        rating_color = "green"
        rating_emoji = "🟢"
        position_size = "5-10%"
    elif percentage_score >= 55:
        rating = "PORTFOLIO FIT"
        rating_color = "lightgreen"
        rating_emoji = "🟡"
        position_size = "3-5%"
    elif percentage_score >= 40:
        rating = "SELECTIVE"
        rating_color = "orange"
        rating_emoji = "🟠"
        position_size = "1-3%"
    else:
        rating = "AVOID"
        rating_color = "red"
        rating_emoji = "🔴"
        position_size = "0%"
    
    return {
        'score': score,
        'max_score': max_score,
        'percentage': percentage_score,
        'rating': rating,
        'rating_color': rating_color,
        'rating_emoji': rating_emoji,
        'metrics': metrics,
        'reasons': reasons,
        'red_flags': red_flags,
        'position_size': position_size
    }


def evaluate_lynch(data: dict) -> dict:
    """
    Evaluate stock using Peter Lynch's methodology.
    
    Lynch's approach:
    - PEG ratio is the key metric (P/E / Growth Rate)
    - Categorize stocks: Fast Grower, Stalwart, Slow Grower, Cyclical, Turnaround, Asset Play
    - Favor simple, boring businesses
    - Look for low institutional ownership (under-followed)
    - Insider ownership matters
    - Balance sheet strength (low debt)
    """
    score = 0
    max_score = 100
    metrics = {}
    reasons = []
    red_flags = []
    stock_category = "Unknown"
    
    # === CATEGORIZE THE STOCK (informational) ===
    revenue_growth = data.get('revenue_growth')
    earnings_growth = data.get('earnings_growth')
    market_cap = data.get('market_cap')
    dividend_yield = data.get('dividend_yield')
    trailing_pe = data.get('trailing_pe')
    
    # Determine category based on Lynch's framework
    rev_growth_pct = (revenue_growth * 100) if revenue_growth and abs(revenue_growth) < 2 else (revenue_growth or 0)
    earn_growth_pct = (earnings_growth * 100) if earnings_growth and abs(earnings_growth) < 2 else (earnings_growth or 0)
    
    if market_cap and market_cap > 50e9 and rev_growth_pct < 6 and dividend_yield and dividend_yield > 0.03:
        stock_category = "Slow Grower"
    elif market_cap and market_cap > 10e9 and 8 <= rev_growth_pct <= 15:
        stock_category = "Stalwart"
    elif rev_growth_pct > 20 and earn_growth_pct > 15:
        stock_category = "Fast Grower"
    elif data.get('sector') in ['Basic Materials', 'Energy', 'Industrials']:
        stock_category = "Cyclical"
    elif trailing_pe and trailing_pe < 0:
        stock_category = "Turnaround"
    elif data.get('price_to_book') and data.get('price_to_book') < 1.0:
        stock_category = "Asset Play"
    elif rev_growth_pct > 15:
        stock_category = "Fast Grower"
    else:
        stock_category = "Stalwart"
    
    metrics['Category'] = {'value': stock_category, 'raw': stock_category}
    
    # === PEG RATIO (35 points) - Lynch's Key Metric ===
    peg_ratio = data.get('peg_ratio')
    
    if peg_ratio and peg_ratio > 0:
        metrics['PEG Ratio'] = {'value': f"{peg_ratio:.2f}", 'raw': peg_ratio}
        if peg_ratio < 0.5:
            score += 35
            reasons.append(f"Excellent PEG: {peg_ratio:.2f} (Very undervalued)")
        elif peg_ratio < 1.0:
            score += 28
            reasons.append(f"Attractive PEG: {peg_ratio:.2f} (Growth at reasonable price)")
        elif peg_ratio < 1.5:
            score += 18
        elif peg_ratio < 2.0:
            score += 10
        else:
            score += 3
            red_flags.append(f"High PEG: {peg_ratio:.2f} (Overvalued for growth)")
    else:
        metrics['PEG Ratio'] = {'value': 'N/A', 'raw': None}
        # Try to calculate from P/E and growth
        if trailing_pe and trailing_pe > 0 and earn_growth_pct > 0:
            calc_peg = trailing_pe / earn_growth_pct
            metrics['PEG (Calc)'] = {'value': f"{calc_peg:.2f}", 'raw': calc_peg}
            if calc_peg < 1.0:
                score += 20
                reasons.append(f"Calculated PEG: {calc_peg:.2f}")
            elif calc_peg < 1.5:
                score += 12
    
    # === GROWTH METRICS (25 points) ===
    # Revenue Growth (12 points)
    if revenue_growth is not None:
        metrics['Revenue Growth'] = {'value': f"{rev_growth_pct:.1f}%", 'raw': rev_growth_pct}
        if rev_growth_pct > 25:
            score += 12
            reasons.append(f"Strong revenue growth: {rev_growth_pct:.1f}% (Fast Grower territory)")
        elif rev_growth_pct > 15:
            score += 10
        elif rev_growth_pct > 10:
            score += 7
        elif rev_growth_pct > 5:
            score += 4
        elif rev_growth_pct < 0:
            red_flags.append(f"Declining revenue: {rev_growth_pct:.1f}%")
    else:
        metrics['Revenue Growth'] = {'value': 'N/A', 'raw': None}
    
    # Earnings Growth (13 points)
    if earnings_growth is not None:
        metrics['Earnings Growth'] = {'value': f"{earn_growth_pct:.1f}%", 'raw': earn_growth_pct}
        if earn_growth_pct > 25:
            score += 13
            reasons.append(f"Excellent earnings growth: {earn_growth_pct:.1f}%")
        elif earn_growth_pct > 15:
            score += 10
        elif earn_growth_pct > 10:
            score += 7
        elif earn_growth_pct > 0:
            score += 4
        elif earn_growth_pct < -10:
            red_flags.append(f"Earnings declining: {earn_growth_pct:.1f}%")
    else:
        metrics['Earnings Growth'] = {'value': 'N/A', 'raw': None}
    
    # === BALANCE SHEET (15 points) ===
    debt_to_equity = data.get('debt_to_equity')
    current_ratio = data.get('current_ratio')
    
    # Debt level (10 points) - Lynch prefers low debt
    if debt_to_equity is not None:
        metrics['Debt/Equity'] = {'value': f"{debt_to_equity:.2f}", 'raw': debt_to_equity}
        if debt_to_equity < 0.25:
            score += 10
            reasons.append(f"Very low debt: D/E {debt_to_equity:.2f}")
        elif debt_to_equity < 0.50:
            score += 8
        elif debt_to_equity < 1.0:
            score += 5
        elif debt_to_equity > 2.0:
            red_flags.append(f"High debt load: D/E {debt_to_equity:.2f}")
    else:
        metrics['Debt/Equity'] = {'value': 'N/A', 'raw': None}
    
    # Current ratio (5 points)
    if current_ratio is not None:
        metrics['Current Ratio'] = {'value': f"{current_ratio:.2f}", 'raw': current_ratio}
        if current_ratio > 2.0:
            score += 5
        elif current_ratio > 1.5:
            score += 4
        elif current_ratio > 1.0:
            score += 2
        elif current_ratio < 1.0:
            red_flags.append(f"Weak current ratio: {current_ratio:.2f}")
    else:
        metrics['Current Ratio'] = {'value': 'N/A', 'raw': None}
    
    # === OWNERSHIP & COVERAGE (15 points) ===
    insider_ownership = data.get('held_percent_insiders')
    institutional_ownership = data.get('held_percent_institutions')
    
    # Insider ownership (8 points) - Lynch likes skin in the game
    if insider_ownership is not None:
        insider_pct = insider_ownership * 100 if insider_ownership < 1 else insider_ownership
        metrics['Insider Ownership'] = {'value': f"{insider_pct:.1f}%", 'raw': insider_pct}
        if insider_pct > 15:
            score += 8
            reasons.append(f"High insider ownership: {insider_pct:.1f}%")
        elif insider_pct > 10:
            score += 6
        elif insider_pct > 5:
            score += 4
        elif insider_pct > 1:
            score += 2
    else:
        metrics['Insider Ownership'] = {'value': 'N/A', 'raw': None}
    
    # Low institutional ownership (7 points) - Lynch likes under-followed stocks
    if institutional_ownership is not None:
        inst_pct = institutional_ownership * 100 if institutional_ownership < 1 else institutional_ownership
        metrics['Institutional Ownership'] = {'value': f"{inst_pct:.1f}%", 'raw': inst_pct}
        if inst_pct < 30:
            score += 7
            reasons.append(f"Under-followed by institutions: {inst_pct:.1f}%")
        elif inst_pct < 50:
            score += 5
        elif inst_pct < 70:
            score += 3
        # High institutional ownership isn't necessarily bad, just less upside from discovery
    else:
        metrics['Institutional Ownership'] = {'value': 'N/A', 'raw': None}
    
    # === VALUATION (10 points) ===
    if trailing_pe and trailing_pe > 0:
        metrics['P/E'] = {'value': f"{trailing_pe:.1f}x", 'raw': trailing_pe}
        if trailing_pe < 12:
            score += 10
            reasons.append(f"Low P/E: {trailing_pe:.1f}x")
        elif trailing_pe < 18:
            score += 7
        elif trailing_pe < 25:
            score += 4
        elif trailing_pe > 40:
            red_flags.append(f"High P/E: {trailing_pe:.1f}x")
    else:
        metrics['P/E'] = {'value': 'N/A', 'raw': None}
    
    # === CATEGORY BONUS/PENALTY ===
    # Lynch prefers Fast Growers and Stalwarts
    if stock_category == "Fast Grower":
        reasons.append("📈 Fast Grower: Lynch's favorite category for tenbaggers")
    elif stock_category == "Stalwart":
        reasons.append("🏢 Stalwart: Good for steady 30-50% gains")
    elif stock_category == "Slow Grower":
        red_flags.append("🐌 Slow Grower: Lynch typically avoids these")
    elif stock_category == "Cyclical":
        red_flags.append("🔄 Cyclical: Requires timing the economic cycle")
    elif stock_category == "Asset Play":
        reasons.append("💎 Asset Play: Hidden value potential")
    elif stock_category == "Turnaround":
        red_flags.append("⚠️ Turnaround: High risk, requires evidence of recovery")
    
    # Calculate final rating
    percentage_score = (score / max_score * 100) if max_score > 0 else 0
    
    if percentage_score >= 75:
        rating = "STRONG BUY"
        rating_color = "green"
        rating_emoji = "🟢"
    elif percentage_score >= 60:
        rating = "BUY"
        rating_color = "lightgreen"
        rating_emoji = "🟡"
    elif percentage_score >= 45:
        rating = "HOLD"
        rating_color = "orange"
        rating_emoji = "🟠"
    else:
        rating = "PASS"
        rating_color = "red"
        rating_emoji = "🔴"
    
    return {
        'score': score,
        'max_score': max_score,
        'percentage': percentage_score,
        'rating': rating,
        'rating_color': rating_color,
        'rating_emoji': rating_emoji,
        'metrics': metrics,
        'reasons': reasons,
        'red_flags': red_flags,
        'category': stock_category
    }


def display_personality_card(personality_key: str, result: dict, data: dict):
    """Display a personality evaluation card."""
    personality = PERSONALITIES[personality_key]
    
    st.markdown(f"### {personality['emoji']} {personality['name']}")
    st.caption(f"*\"{personality['tagline']}\"*")
    
    # Score display
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric(
            label="Score",
            value=f"{result['score']}/{result['max_score']}",
            delta=f"{result['percentage']:.0f}%"
        )
    with col2:
        st.markdown(f"### {result['rating_emoji']} **{result['rating']}**")
    
    # Key Metrics Table
    st.markdown("#### 📊 Key Metrics")
    metrics_data = []
    for name, info in result['metrics'].items():
        metrics_data.append({
            'Metric': name,
            'Value': info['value']
        })
    
    if metrics_data:
        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    
    # Strengths
    if result['reasons']:
        st.markdown("#### ✅ Strengths")
        for reason in result['reasons']:
            st.markdown(f"- {reason}")
    
    # Red Flags
    if result['red_flags']:
        st.markdown("#### ⚠️ Red Flags")
        for flag in result['red_flags']:
            st.markdown(f"- {flag}")


# --- Main UI ---
st.markdown("---")

# Ticker input
col1, col2, col3 = st.columns([2, 1, 2])
with col2:
    ticker_input = st.text_input(
        "Enter Ticker Symbol",
        placeholder="e.g., AAPL",
        key="personality_ticker"
    ).upper().strip()

# Analyze button
analyze_col1, analyze_col2, analyze_col3 = st.columns([1, 2, 1])
with analyze_col2:
    analyze_button = st.button("🔍 Analyze Stock", use_container_width=True, type="primary")

st.markdown("---")

# Run analysis
if analyze_button and ticker_input:
    with st.spinner(f"Fetching data for {ticker_input}..."):
        data = fetch_stock_data(ticker_input)
    
    if not data.get('success'):
        st.error(f"❌ Could not fetch data for {ticker_input}: {data.get('error', 'Unknown error')}")
    else:
        # Display company header
        st.markdown(f"## {data['company_name']} ({data['ticker']})")
        st.caption(f"{data['sector']} • {data['industry']}")
        
        # Quick stats
        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
        with stat_col1:
            st.metric("Price", f"${data['current_price']:.2f}" if data['current_price'] else "N/A")
        with stat_col2:
            st.metric("Market Cap", format_currency(data['market_cap']))
        with stat_col3:
            st.metric("P/E", f"{data['trailing_pe']:.1f}" if data['trailing_pe'] else "N/A")
        with stat_col4:
            st.metric("52W Range", f"${data['fifty_two_week_low']:.0f} - ${data['fifty_two_week_high']:.0f}" if data['fifty_two_week_low'] and data['fifty_two_week_high'] else "N/A")
        
        st.markdown("---")
        
        # Run all evaluations
        buffett_result = evaluate_buffett(data)
        rk_result = evaluate_roaring_kitty(data)
        druckenmiller_result = evaluate_druckenmiller(data)
        greenblatt_result = evaluate_greenblatt(data)
        dalio_result = evaluate_dalio(data)
        lynch_result = evaluate_lynch(data)
        
        # === SUMMARY SECTION ===
        st.markdown("### 📊 Summary Ratings")
        
        # First row - 3 personalities
        summary_cols1 = st.columns(3)
        
        with summary_cols1[0]:
            st.markdown(f"""
            <div style="background-color: #1E3A5F; padding: 12px; border-radius: 10px; text-align: center;">
                <h4 style="margin: 0; color: white; font-size: 14px;">🏛️ Buffett</h4>
                <h3 style="margin: 8px 0; color: white;">{buffett_result['rating_emoji']} {buffett_result['rating']}</h3>
                <p style="margin: 0; color: #ccc; font-size: 12px;">{buffett_result['percentage']:.0f}%</p>
            </div>
            """, unsafe_allow_html=True)
        
        with summary_cols1[1]:
            st.markdown(f"""
            <div style="background-color: #8B0000; padding: 12px; border-radius: 10px; text-align: center;">
                <h4 style="margin: 0; color: white; font-size: 14px;">🐱 Roaring Kitty</h4>
                <h3 style="margin: 8px 0; color: white;">{rk_result['rating_emoji']} {rk_result['rating']}</h3>
                <p style="margin: 0; color: #ccc; font-size: 12px;">{rk_result['percentage']:.0f}%</p>
            </div>
            """, unsafe_allow_html=True)
        
        with summary_cols1[2]:
            st.markdown(f"""
            <div style="background-color: #2E7D32; padding: 12px; border-radius: 10px; text-align: center;">
                <h4 style="margin: 0; color: white; font-size: 14px;">📈 Druckenmiller</h4>
                <h3 style="margin: 8px 0; color: white;">{druckenmiller_result['rating_emoji']} {druckenmiller_result['rating']}</h3>
                <p style="margin: 0; color: #ccc; font-size: 12px;">{druckenmiller_result['percentage']:.0f}%</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Second row - 3 personalities
        st.markdown("")  # Spacing
        summary_cols2 = st.columns(3)
        
        with summary_cols2[0]:
            st.markdown(f"""
            <div style="background-color: #6A0DAD; padding: 12px; border-radius: 10px; text-align: center;">
                <h4 style="margin: 0; color: white; font-size: 14px;">🧮 Greenblatt</h4>
                <h3 style="margin: 8px 0; color: white;">{greenblatt_result['rating_emoji']} {greenblatt_result['rating']}</h3>
                <p style="margin: 0; color: #ccc; font-size: 12px;">{greenblatt_result['percentage']:.0f}%</p>
            </div>
            """, unsafe_allow_html=True)
        
        with summary_cols2[1]:
            st.markdown(f"""
            <div style="background-color: #FF6600; padding: 12px; border-radius: 10px; text-align: center;">
                <h4 style="margin: 0; color: white; font-size: 14px;">🌊 Dalio</h4>
                <h3 style="margin: 8px 0; color: white;">{dalio_result['rating_emoji']} {dalio_result['rating']}</h3>
                <p style="margin: 0; color: #ccc; font-size: 12px;">{dalio_result['percentage']:.0f}%</p>
            </div>
            """, unsafe_allow_html=True)
        
        with summary_cols2[2]:
            st.markdown(f"""
            <div style="background-color: #8B4513; padding: 12px; border-radius: 10px; text-align: center;">
                <h4 style="margin: 0; color: white; font-size: 14px;">📚 Lynch</h4>
                <h3 style="margin: 8px 0; color: white;">{lynch_result['rating_emoji']} {lynch_result['rating']}</h3>
                <p style="margin: 0; color: #ccc; font-size: 12px;">{lynch_result['percentage']:.0f}%</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # === TABBED DETAILED BREAKDOWN ===
        st.markdown("### 🔍 Detailed Analysis")
        
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "🏛️ Warren Buffett",
            "🐱 Roaring Kitty", 
            "📈 Druckenmiller",
            "🧮 Greenblatt",
            "🌊 Ray Dalio",
            "📚 Peter Lynch"
        ])
        
        with tab1:
            display_personality_card("warren_buffett", buffett_result, data)
        
        with tab2:
            display_personality_card("roaring_kitty", rk_result, data)
        
        with tab3:
            display_personality_card("stanley_druckenmiller", druckenmiller_result, data)
            if druckenmiller_result.get('position_size'):
                st.info(f"💡 **Suggested Position Size:** {druckenmiller_result['position_size']}")
        
        with tab4:
            display_personality_card("joel_greenblatt", greenblatt_result, data)
        
        with tab5:
            display_personality_card("ray_dalio", dalio_result, data)
            if dalio_result.get('position_size'):
                st.info(f"💡 **Suggested Position Size:** {dalio_result['position_size']}")
        
        with tab6:
            display_personality_card("peter_lynch", lynch_result, data)
            if lynch_result.get('category'):
                st.info(f"📊 **Lynch Category:** {lynch_result['category']}")

elif analyze_button and not ticker_input:
    st.warning("Please enter a ticker symbol.")

# --- Methodology Expanders ---
st.markdown("---")
with st.expander("📖 Warren Buffett Methodology"):
    st.markdown("""
    **Philosophy:** Buy wonderful businesses at fair prices, hold forever.
    
    **Scoring (100 points total):**
    - **Quality (40 pts):** ROE, profit margins, cash conversion, earnings consistency
    - **Moat (30 pts):** Gross margins, capital efficiency (capex/OCF), ROIC
    - **Balance Sheet (15 pts):** Debt/Equity, current ratio
    - **Valuation (15 pts):** P/E, PEG ratio, P/B
    
    **Ratings:**
    - 🟢 **WONDERFUL BUSINESS** (80%+): Buffett would likely be interested
    - 🟡 **QUALITY BUSINESS** (65-80%): Good business, monitor for opportunity
    - 🟠 **WATCH FOR BETTER PRICE** (50-65%): Decent but wait for pullback
    - 🔴 **LIKELY PASS** (35-50%): Doesn't meet quality standards
    - ⛔ **AVOID** (<35%): Significant red flags
    """)

with st.expander("📖 Roaring Kitty Methodology"):
    st.markdown("""
    **Philosophy:** Contrarian deep value with short squeeze potential.
    
    **Scoring (13 points max):**
    - **Deep Value:** P/B < 1.0, trading below net cash, high FCF yield
    - **Balance Sheet:** Strong current ratio, low debt
    - **Short Squeeze Setup:** High short % of float, days to cover > 5
    - **Price Positioning:** Near 52-week lows
    
    **Ratings:**
    - 🟢 **STRONG BUY** (8+ pts): Classic RK opportunity
    - 🟡 **BUY** (5-7 pts): Good value setup
    - 🟠 **WATCH** (3-4 pts): Some potential, needs catalyst
    - 🔴 **PASS** (<3 pts): Doesn't fit the strategy
    
    **Key Insight:** RK specifically looked for stocks where shorts were overly aggressive 
    on fundamentally sound companies - creating asymmetric risk/reward.
    """)

with st.expander("📖 Stanley Druckenmiller Methodology"):
    st.markdown("""
    **Philosophy:** Ride macro trends with aggressive sizing when conviction is high.
    
    **Scoring (100 points total):**
    - **Technical Momentum (30 pts):** Price vs MAs, proximity to 52W high, institutional ownership
    - **Growth & Earnings (30 pts):** Revenue growth, EPS momentum, operating leverage
    - **Valuation (20 pts):** PEG ratio, Forward P/E
    - **Quality (20 pts):** Profit margins, market cap/leadership
    
    **Ratings:**
    - 🟢 **HIGH CONVICTION BUY** (80%+): Size up aggressively (15-30%)
    - 🟡 **STRONG BUY** (65-80%): Core position (10-20%)
    - 🟠 **BUY** (50-65%): Standard position (5-10%)
    - 🔴 **WATCH** (35-50%): Starter position only (2-5%)
    - ⛔ **PASS** (<35%): Insufficient momentum
    
    **Key Insight:** Druckenmiller focuses on riding macro trends with the wind at your back.
    He sizes positions based on conviction and risk/reward, cutting losers quickly while letting winners run.
    """)

with st.expander("📖 Joel Greenblatt Methodology"):
    st.markdown("""
    **Philosophy:** Buy good companies (high ROC) at bargain prices (high earnings yield).
    
    **Scoring (100 points total):**
    - **Return on Capital (40 pts):** ROE, operating margin, ROIC estimate
    - **Earnings Yield (40 pts):** Earnings yield (1/P/E), EV/EBITDA, forward P/E discount
    - **Quality Checks (20 pts):** Positive EPS, low debt, market cap, non-excluded sector
    
    **Ratings:**
    - 🟢 **STRONG BUY** (70%+): Classic Magic Formula candidate
    - 🟡 **BUY** (55-70%): Good quality + value combination
    - 🟠 **HOLD** (40-55%): Mixed signals
    - 🔴 **PASS** (<40%): Doesn't meet criteria
    
    **Key Insight:** The Magic Formula systematically ranks stocks by combining quality (ROC) and 
    value (earnings yield). Works best in small/mid caps. Excludes financials and utilities.
    """)

with st.expander("📖 Ray Dalio Methodology"):
    st.markdown("""
    **Philosophy:** Balance across economic environments, not just asset classes.
    
    **Scoring (100 points total):**
    - **Environment Fit (30 pts):** Beta (risk contribution), sector environment fit, dividend yield
    - **Balance Sheet Strength (25 pts):** Current ratio, debt level, cash position
    - **Quality & Consistency (25 pts):** Profit margin, revenue growth, earnings growth
    - **Valuation (20 pts):** P/E, PEG ratio, P/B
    
    **Ratings:**
    - 🟢 **CORE HOLDING** (70%+): Suitable for 5-10% portfolio weight
    - 🟡 **PORTFOLIO FIT** (55-70%): Good for 3-5% weight
    - 🟠 **SELECTIVE** (40-55%): Small position only (1-3%)
    - 🔴 **AVOID** (<40%): Poor fit for balanced portfolio
    
    **Key Insight:** Dalio thinks about 4 environments (rising/falling growth × rising/falling inflation).
    Stocks suited for multiple environments or with defensive characteristics score higher.
    """)

with st.expander("📖 Peter Lynch Methodology"):
    st.markdown("""
    **Philosophy:** Invest in what you know, find simple businesses before Wall Street discovers them.
    
    **The Six Stock Categories:**
    - **Fast Growers:** 20%+ growth, Lynch's favorites for tenbaggers
    - **Stalwarts:** Large caps with 10-15% growth, good for 30-50% gains
    - **Slow Growers:** Under 6% growth with high dividends, typically avoid
    - **Cyclicals:** Tied to economic cycles, requires timing
    - **Turnarounds:** Troubled companies restructuring, high risk
    - **Asset Plays:** Hidden value not reflected in stock price
    
    **Scoring (100 points total):**
    - **PEG Ratio (35 pts):** The key metric - P/E divided by growth rate
    - **Growth Metrics (25 pts):** Revenue and earnings growth rates
    - **Balance Sheet (15 pts):** Low debt, strong current ratio
    - **Ownership (15 pts):** High insider ownership, low institutional (under-followed)
    - **Valuation (10 pts):** P/E reasonableness
    
    **PEG Rules:**
    - PEG < 0.5: Very undervalued
    - PEG < 1.0: Fairly valued (growth equals P/E)
    - PEG > 1.5: Overvalued
    
    **Ratings:**
    - 🟢 **STRONG BUY** (75%+): Potential tenbagger
    - 🟡 **BUY** (60-75%): Good Lynch-style investment
    - 🟠 **HOLD** (45-60%): Mixed signals
    - 🔴 **PASS** (<45%): Doesn't fit Lynch criteria
    
    **Key Insight:** Lynch emphasizes knowing what you own. If you can't explain why you own 
    a stock in two minutes ("the two-minute drill"), you don't understand it well enough.
    """)
