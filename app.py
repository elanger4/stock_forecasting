import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests

st.set_page_config(page_title="Stock Valuation Projection Engine", layout="wide")

# Hide Streamlit default elements but keep sidebar visible
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    .stDeployButton {display: none;}
    [data-testid="stSidebarNav"] {display: none;}
    /* Ensure sidebar stays visible */
    [data-testid="stSidebar"] {
        display: block !important;
        visibility: visible !important;
    }
    /* Add padding at top since header is visible */
    .block-container {
        padding-top: 1rem !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Custom CSS to make everything bigger ---
st.markdown("""
<style>
    /* Increase base font size */
    html, body, [class*="css"] {
        font-size: 18px;
    }
    
    /* Larger headers */
    h1 {
        font-size: 2.5rem !important;
    }
    h2 {
        font-size: 2rem !important;
    }
    h3 {
        font-size: 1.6rem !important;
    }
    
    /* Larger metrics */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 1.1rem !important;
    }
    
    /* Larger table text */
    .stTable table {
        font-size: 1.1rem !important;
    }
    .stTable th {
        font-size: 1.15rem !important;
        padding: 12px 16px !important;
    }
    .stTable td {
        font-size: 1.1rem !important;
        padding: 10px 16px !important;
    }
    
    /* Larger input labels and values */
    .stNumberInput label, .stTextInput label {
        font-size: 1.1rem !important;
    }
    .stNumberInput input, .stTextInput input {
        font-size: 1.1rem !important;
        padding: 10px 14px !important;
    }
    
    /* Larger buttons */
    .stButton button {
        font-size: 1.1rem !important;
        padding: 12px 24px !important;
    }
    
    /* Larger sidebar text */
    [data-testid="stSidebar"] {
        font-size: 1rem !important;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2 {
        font-size: 1.5rem !important;
    }
    
    /* Larger expander text */
    .streamlit-expanderHeader {
        font-size: 1.15rem !important;
    }
    
    /* More padding in main content */
    .block-container {
        padding: 2rem 3rem !important;
    }
    
    /* ========== MOBILE RESPONSIVE STYLES ========== */
    @media (max-width: 768px) {
        /* Reduce base padding on mobile - ensure content isn't cut off */
        .block-container {
            padding: 1rem 1rem !important;
            margin-left: 0 !important;
        }
        
        /* Ensure main content area has proper padding */
        [data-testid="stAppViewContainer"] {
            padding-left: 0.5rem !important;
        }
        
        /* Fix metric labels being cut off */
        [data-testid="stMetric"] {
            padding-left: 0.25rem !important;
        }
        
        /* Stack metric columns vertically */
        [data-testid="column"] {
            width: 100% !important;
            flex: 1 1 100% !important;
            min-width: 100% !important;
            padding-left: 0.5rem !important;
        }
        
        /* Make metrics more compact */
        [data-testid="stMetricValue"] {
            font-size: 1.4rem !important;
        }
        [data-testid="stMetricLabel"] {
            font-size: 0.9rem !important;
        }
        [data-testid="stMetricDelta"] {
            font-size: 0.8rem !important;
        }
        
        /* Smaller headers on mobile */
        h1 {
            font-size: 1.8rem !important;
        }
        h2 {
            font-size: 1.4rem !important;
        }
        h3 {
            font-size: 1.2rem !important;
        }
        
        /* Make tables horizontally scrollable */
        [data-testid="stDataFrame"], 
        [data-testid="stTable"],
        .stDataFrame {
            overflow-x: auto !important;
            -webkit-overflow-scrolling: touch;
        }
        
        /* Smaller table text on mobile */
        .stTable table, [data-testid="stDataFrame"] table {
            font-size: 0.85rem !important;
        }
        .stTable th, .stTable td {
            padding: 6px 8px !important;
            font-size: 0.85rem !important;
        }
        
        /* Larger touch targets for buttons */
        .stButton button {
            min-height: 48px !important;
            padding: 12px 16px !important;
            font-size: 1rem !important;
            width: 100% !important;
        }
        
        /* Make toggles easier to tap */
        [data-testid="stToggle"] {
            min-height: 44px !important;
        }
        
        /* Sidebar adjustments */
        [data-testid="stSidebar"] {
            min-width: 280px !important;
        }
        [data-testid="stSidebar"] .stSlider {
            padding: 0.5rem 0 !important;
        }
        
        /* Navigation links stack vertically */
        .row-widget.stHorizontalBlock {
            flex-wrap: wrap !important;
        }
        
        /* Expanders full width */
        .streamlit-expanderHeader {
            font-size: 1rem !important;
            padding: 12px 8px !important;
        }
        
        /* Number inputs more touch-friendly */
        .stNumberInput input {
            min-height: 44px !important;
            font-size: 1rem !important;
        }
        
        /* Select boxes */
        .stSelectbox > div > div {
            min-height: 44px !important;
        }
        
        /* Text inputs */
        .stTextInput input {
            min-height: 44px !important;
            font-size: 1rem !important;
        }
        
        /* Warning/Info boxes */
        .stAlert {
            padding: 0.75rem !important;
            font-size: 0.9rem !important;
        }
        
        /* Tabs */
        .stTabs [data-baseweb="tab"] {
            padding: 10px 12px !important;
            font-size: 0.9rem !important;
        }
    }
    
    /* Extra small screens (phones in portrait) */
    @media (max-width: 480px) {
        .block-container {
            padding: 0.5rem 0.25rem !important;
        }
        
        [data-testid="stMetricValue"] {
            font-size: 1.2rem !important;
        }
        
        h1 {
            font-size: 1.5rem !important;
        }
        h2 {
            font-size: 1.2rem !important;
        }
        
        /* Hide less critical navigation text */
        .row-widget.stHorizontalBlock a {
            font-size: 0.85rem !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# Import centralized watchlist configuration
from watchlists import WATCHLISTS, get_watchlist_names, get_watchlist_stocks, get_default_watchlist

# Initialize watchlist selection in session state
if 'selected_watchlist' not in st.session_state:
    st.session_state.selected_watchlist = get_default_watchlist()

def get_current_watchlist_stocks():
    """Get stocks for the currently selected watchlist with empty option for custom input."""
    stocks = get_watchlist_stocks(st.session_state.selected_watchlist)
    return [""] + stocks  # Add empty option for custom input

# For backward compatibility - this will be dynamically updated
WATCHLIST_STOCKS = get_current_watchlist_stocks()

# --- Helper Functions ---

def format_large_number(value: float) -> str:
    """Format large numbers with B/M/K suffixes."""
    if value is None or np.isnan(value):
        return "N/A"
    abs_val = abs(value)
    if abs_val >= 1e12:
        return f"${value/1e12:.2f}T"
    elif abs_val >= 1e9:
        return f"${value/1e9:.2f}B"
    elif abs_val >= 1e6:
        return f"${value/1e6:.2f}M"
    elif abs_val >= 1e3:
        return f"${value/1e3:.2f}K"
    else:
        return f"${value:.2f}"

def format_currency(value: float) -> str:
    """Format as currency."""
    if value is None or np.isnan(value):
        return "N/A"
    return f"${value:,.2f}"

def format_percent(value: float) -> str:
    """Format as percentage."""
    if value is None or np.isnan(value):
        return "N/A"
    return f"{value:.1f}%"

def fetch_watchlist_news():
    """Fetch news for all watchlist stocks (most recent articles)."""
    from datetime import datetime, timedelta
    all_news = []
    
    for stock_ticker in WATCHLIST_STOCKS[1:]:  # Skip empty string
        try:
            stock = yf.Ticker(stock_ticker)
            news = stock.news
            if news:
                for article in news[:3]:  # Limit per stock
                    # Try multiple ways to get title
                    title = None
                    link = '#'
                    pub_timestamp = None
                    
                    # Check if article has 'content' structure (newer yfinance format)
                    content = article.get('content', {})
                    publisher = ''
                    if isinstance(content, dict) and content:
                        title = content.get('title', '')
                        # Get link from canonicalUrl
                        canonical = content.get('canonicalUrl', {})
                        if isinstance(canonical, dict):
                            link = canonical.get('url', '#')
                        # Get publisher
                        provider = content.get('provider', {})
                        if isinstance(provider, dict):
                            publisher = provider.get('displayName', '')
                        # Get pubDate (string format like "2024-12-23T10:30:00Z")
                        pub_date_str = content.get('pubDate', '')
                        if pub_date_str:
                            try:
                                # Parse ISO format date
                                pub_timestamp = datetime.fromisoformat(pub_date_str.replace('Z', '+00:00'))
                                pub_timestamp = pub_timestamp.replace(tzinfo=None)  # Remove timezone for comparison
                            except:
                                pass
                    
                    # Fallback to older yfinance format
                    if not title:
                        title = article.get('title', '')
                    if link == '#':
                        link = article.get('link', '#')
                    if not publisher:
                        publisher = article.get('publisher', '')
                    if not pub_timestamp:
                        # providerPublishTime is Unix timestamp
                        unix_time = article.get('providerPublishTime', 0)
                        if isinstance(unix_time, (int, float)) and unix_time > 0:
                            try:
                                pub_timestamp = datetime.fromtimestamp(unix_time)
                            except:
                                pass
                    
                    if not title:
                        continue
                    
                    # Default timestamp if still none
                    if not pub_timestamp:
                        pub_timestamp = datetime.now() - timedelta(days=1)  # Default to 1 day ago
                    
                    all_news.append({
                        'ticker': stock_ticker,
                        'title': title,
                        'link': link,
                        'publisher': publisher,
                        'timestamp': pub_timestamp
                    })
        except Exception:
            pass
    
    # Sort by timestamp (most recent first)
    all_news.sort(key=lambda x: x['timestamp'], reverse=True)
    return all_news[:30]  # Limit total to 30 articles

def fetch_calendar_events(ticker: str) -> list:
    """Fetch calendar events for a single stock. Returns list of (date, event_type, ticker) tuples."""
    from datetime import datetime
    import pandas as pd
    today = datetime.now().date()
    events = []
    
    try:
        stock = yf.Ticker(ticker)
        
        # Get calendar data
        try:
            calendar = stock.calendar
            if calendar is not None:
                # Handle both dict and DataFrame formats
                if isinstance(calendar, pd.DataFrame):
                    calendar = calendar.to_dict()
                
                if isinstance(calendar, dict):
                    # Earnings dates
                    if 'Earnings Date' in calendar:
                        earnings = calendar['Earnings Date']
                        if isinstance(earnings, list):
                            for e in earnings:
                                try:
                                    e_date = e.date() if hasattr(e, 'date') else e
                                    if e_date >= today:
                                        events.append((e_date, 'Earnings', ticker))
                                except:
                                    pass
                        elif earnings is not None:
                            try:
                                e_date = earnings.date() if hasattr(earnings, 'date') else earnings
                                if e_date >= today:
                                    events.append((e_date, 'Earnings', ticker))
                            except:
                                pass
                    
                    # Dividend date
                    if 'Dividend Date' in calendar and calendar['Dividend Date'] is not None:
                        div_date = calendar['Dividend Date']
                        try:
                            d_date = div_date.date() if hasattr(div_date, 'date') else div_date
                            if d_date >= today:
                                events.append((d_date, 'Dividend', ticker))
                        except:
                            pass
                    
                    # Ex-Dividend date
                    if 'Ex-Dividend Date' in calendar and calendar['Ex-Dividend Date'] is not None:
                        ex_div = calendar['Ex-Dividend Date']
                        try:
                            ex_date = ex_div.date() if hasattr(ex_div, 'date') else ex_div
                            if ex_date >= today:
                                events.append((ex_date, 'Ex-Dividend', ticker))
                        except:
                            pass
        except Exception:
            pass
        
        # Get earnings dates from earnings_dates property
        try:
            earnings_dates = stock.earnings_dates
            if earnings_dates is not None and not earnings_dates.empty:
                for date_idx in earnings_dates.index[:4]:  # Limit to next 4
                    try:
                        e_date = date_idx.date() if hasattr(date_idx, 'date') else date_idx
                        if e_date >= today:
                            events.append((e_date, 'Earnings', ticker))
                    except:
                        pass
        except Exception:
            pass
    except Exception:
        pass
    
    return events

def fetch_stock_data(ticker: str) -> dict:
    """Fetch all required stock data from yfinance."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Current Price
        current_price = info.get('currentPrice') or info.get('regularMarketPrice') or info.get('previousClose')
        
        # Shares Outstanding
        shares_outstanding = info.get('sharesOutstanding')
        
        # Get financials for Revenue and Net Income
        income_stmt = stock.income_stmt
        fiscal_year = None
        revenue_growth = None
        historical_data = []  # Store historical years data
        
        ebit = None  # For Magic Formula calculations
        if income_stmt is not None and not income_stmt.empty:
            total_revenue = income_stmt.loc['Total Revenue'].iloc[0] if 'Total Revenue' in income_stmt.index else None
            net_income = income_stmt.loc['Net Income'].iloc[0] if 'Net Income' in income_stmt.index else None
            # Extract EBIT (Operating Income) for Magic Formula
            if 'EBIT' in income_stmt.index:
                ebit = income_stmt.loc['EBIT'].iloc[0]
            elif 'Operating Income' in income_stmt.index:
                ebit = income_stmt.loc['Operating Income'].iloc[0]
            # Get the fiscal year from the most recent column
            fiscal_year = income_stmt.columns[0].year if hasattr(income_stmt.columns[0], 'year') else None
            # Calculate revenue growth from last 2 years
            if 'Total Revenue' in income_stmt.index and len(income_stmt.columns) >= 2:
                rev_current = income_stmt.loc['Total Revenue'].iloc[0]
                rev_prior = income_stmt.loc['Total Revenue'].iloc[1]
                if rev_prior and rev_prior > 0:
                    revenue_growth = ((rev_current / rev_prior) - 1) * 100
            
            # Extract historical data (up to 3 previous years, excluding current)
            for i in range(1, min(4, len(income_stmt.columns))):
                try:
                    col = income_stmt.columns[i]
                    year = col.year if hasattr(col, 'year') else fiscal_year - i
                    hist_revenue = income_stmt.loc['Total Revenue'].iloc[i] if 'Total Revenue' in income_stmt.index else None
                    hist_net_income = income_stmt.loc['Net Income'].iloc[i] if 'Net Income' in income_stmt.index else None
                    
                    # Try to get Basic EPS from income statement
                    hist_eps = None
                    if 'Basic EPS' in income_stmt.index:
                        hist_eps = income_stmt.loc['Basic EPS'].iloc[i]
                    elif 'Diluted EPS' in income_stmt.index:
                        hist_eps = income_stmt.loc['Diluted EPS'].iloc[i]
                    elif hist_net_income and shares_outstanding:
                        # Calculate EPS from net income / shares
                        hist_eps = hist_net_income / shares_outstanding
                    
                    # Calculate margin
                    hist_margin = None
                    if hist_revenue and hist_net_income and hist_revenue > 0:
                        hist_margin = (hist_net_income / hist_revenue) * 100
                    
                    # Calculate revenue growth (compared to prior year)
                    hist_rev_growth = None
                    if i + 1 < len(income_stmt.columns) and 'Total Revenue' in income_stmt.index:
                        prior_rev = income_stmt.loc['Total Revenue'].iloc[i + 1]
                        if prior_rev and prior_rev > 0:
                            hist_rev_growth = ((hist_revenue / prior_rev) - 1) * 100
                    
                    # Calculate net income growth
                    hist_ni_growth = None
                    if i + 1 < len(income_stmt.columns) and 'Net Income' in income_stmt.index:
                        prior_ni = income_stmt.loc['Net Income'].iloc[i + 1]
                        if prior_ni and prior_ni != 0:
                            hist_ni_growth = ((hist_net_income / prior_ni) - 1) * 100
                    
                    # Try to get historical stock price for year-end
                    hist_price_low = None
                    hist_price_high = None
                    hist_pe = None
                    try:
                        # Get historical data for that year
                        from datetime import datetime
                        start_date = f"{year}-01-01"
                        end_date = f"{year}-12-31"
                        hist_prices = stock.history(start=start_date, end=end_date)
                        if hist_prices is not None and not hist_prices.empty:
                            hist_price_low = hist_prices['Low'].min()
                            hist_price_high = hist_prices['High'].max()
                            # Calculate P/E using year-end price
                            year_end_price = hist_prices['Close'].iloc[-1]
                            if hist_eps and hist_eps > 0:
                                hist_pe = year_end_price / hist_eps
                    except:
                        pass
                    
                    historical_data.append({
                        'year': year,
                        'revenue': hist_revenue,
                        'net_income': hist_net_income,
                        'margin': hist_margin,
                        'rev_growth': hist_rev_growth,
                        'ni_growth': hist_ni_growth,
                        'eps': hist_eps,
                        'pe': hist_pe,
                        'price_low': hist_price_low,
                        'price_high': hist_price_high
                    })
                except:
                    pass
        else:
            total_revenue = info.get('totalRevenue')
            net_income = info.get('netIncomeToCommon')
        
        # Fallback to info dict for revenue growth
        if revenue_growth is None:
            revenue_growth = info.get('revenueGrowth')
            if revenue_growth:
                revenue_growth = revenue_growth * 100  # Convert from decimal
        
        # Current EPS (TTM)
        current_eps = info.get('trailingEps')
        
        # Analyst Estimates - try multiple sources
        eps_low = None
        eps_high = None
        
        # Try analyst_price_targets for EPS estimates
        try:
            earnings_estimate = stock.earnings_estimate
            if earnings_estimate is not None and not earnings_estimate.empty:
                # Look for next year estimates
                if '+1y' in earnings_estimate.columns:
                    eps_low = earnings_estimate.loc['low', '+1y'] if 'low' in earnings_estimate.index else None
                    eps_high = earnings_estimate.loc['high', '+1y'] if 'high' in earnings_estimate.index else None
                elif '0y' in earnings_estimate.columns:
                    eps_low = earnings_estimate.loc['low', '0y'] if 'low' in earnings_estimate.index else None
                    eps_high = earnings_estimate.loc['high', '0y'] if 'high' in earnings_estimate.index else None
        except:
            pass
        
        # Fallback to info dict
        if eps_low is None:
            eps_low = info.get('earningsLow')
        if eps_high is None:
            eps_high = info.get('earningsHigh')
        
        # Calculate current net income margin
        current_margin = None
        if total_revenue and net_income and total_revenue > 0:
            current_margin = (net_income / total_revenue) * 100
        
        # Historical P/E as fallback
        historical_pe = info.get('trailingPE')
        
        # Valuation Ratios
        market_cap = info.get('marketCap')
        enterprise_value = info.get('enterpriseValue')
        book_value = info.get('bookValue')
        
        # P/Sales = Market Cap / Revenue
        p_sales = None
        if market_cap and total_revenue and total_revenue > 0:
            p_sales = market_cap / total_revenue
        
        # P/Book = Price / Book Value per Share
        p_book = info.get('priceToBook')
        
        # EV/EBITDA
        ev_ebitda = info.get('enterpriseToEbitda')
        
        # EV/Revenue
        ev_revenue = info.get('enterpriseToRevenue')
        
        # Trailing P/E
        trailing_pe = info.get('trailingPE')
        
        # Forward P/E
        forward_pe = info.get('forwardPE')
        
        # Get sector for comparison
        sector = info.get('sector', 'Unknown')
        industry = info.get('industry', 'Unknown')
        
        # If no fiscal year found, use current year
        if fiscal_year is None:
            from datetime import datetime
            fiscal_year = datetime.now().year
        
        # Get news
        try:
            news = stock.news
        except:
            news = []
        
        # Get calendar (earnings, dividends, etc.)
        try:
            calendar = stock.calendar
        except:
            calendar = None
        
        # Get upcoming earnings dates
        try:
            earnings_dates = stock.earnings_dates
            if earnings_dates is not None and not earnings_dates.empty:
                # Get only future dates
                from datetime import datetime
                today = datetime.now()
                earnings_dates = earnings_dates[earnings_dates.index >= today].head(4)
            else:
                earnings_dates = None
        except:
            earnings_dates = None
        
        # 52-week high and low
        fifty_two_week_low = info.get('fiftyTwoWeekLow')
        fifty_two_week_high = info.get('fiftyTwoWeekHigh')
        
        # Financial statements for detailed view
        balance_sheet = stock.balance_sheet
        cash_flow = stock.cashflow
        
        # Holders data
        try:
            institutional_holders = stock.institutional_holders
        except:
            institutional_holders = None
        
        try:
            major_holders = stock.major_holders
        except:
            major_holders = None
        
        try:
            mutualfund_holders = stock.mutualfund_holders
        except:
            mutualfund_holders = None
        
        # Analyst price targets
        target_low = info.get('targetLowPrice')
        target_mean = info.get('targetMeanPrice')
        target_high = info.get('targetHighPrice')
        target_median = info.get('targetMedianPrice')
        recommendation = info.get('recommendationKey')
        recommendation_mean = info.get('recommendationMean')
        num_analysts = info.get('numberOfAnalystOpinions')
        
        # Analyst recommendations history
        try:
            recommendations = stock.recommendations
        except:
            recommendations = None
        
        # Insider transactions
        try:
            insider_transactions = stock.insider_transactions
        except:
            insider_transactions = None
        
        try:
            insider_purchases = stock.insider_purchases
        except:
            insider_purchases = None
        
        # DCF data - Free Cash Flow
        free_cash_flow = info.get('freeCashflow')
        operating_cash_flow = info.get('operatingCashflow')
        total_debt = info.get('totalDebt', 0) or 0
        total_cash = info.get('totalCash', 0) or 0
        
        # Financial Health data - try info first, then fall back to balance sheet
        total_stockholder_equity = info.get('totalStockholderEquity')
        current_assets = info.get('totalCurrentAssets')
        current_liabilities = info.get('totalCurrentLiabilities')
        ebitda = info.get('ebitda')
        total_assets = info.get('totalAssets')
        retained_earnings = info.get('retainedEarnings')
        
        # Fall back to balance sheet if info doesn't have the data
        if balance_sheet is not None and not balance_sheet.empty:
            try:
                # Get most recent column (latest fiscal year)
                latest_col = balance_sheet.columns[0]
                
                if total_stockholder_equity is None:
                    for key in ['Stockholders Equity', 'Total Stockholders Equity', 'Total Equity Gross Minority Interest', 'Common Stock Equity']:
                        if key in balance_sheet.index:
                            total_stockholder_equity = balance_sheet.loc[key, latest_col]
                            break
                
                if current_assets is None:
                    if 'Current Assets' in balance_sheet.index:
                        current_assets = balance_sheet.loc['Current Assets', latest_col]
                    elif 'Total Current Assets' in balance_sheet.index:
                        current_assets = balance_sheet.loc['Total Current Assets', latest_col]
                
                if current_liabilities is None:
                    if 'Current Liabilities' in balance_sheet.index:
                        current_liabilities = balance_sheet.loc['Current Liabilities', latest_col]
                    elif 'Total Current Liabilities' in balance_sheet.index:
                        current_liabilities = balance_sheet.loc['Total Current Liabilities', latest_col]
                
                if total_assets is None:
                    if 'Total Assets' in balance_sheet.index:
                        total_assets = balance_sheet.loc['Total Assets', latest_col]
                
                if retained_earnings is None:
                    if 'Retained Earnings' in balance_sheet.index:
                        retained_earnings = balance_sheet.loc['Retained Earnings', latest_col]
            except Exception:
                pass
        
        return {
            'current_price': current_price,
            'total_revenue': total_revenue,
            'net_income': net_income,
            'shares_outstanding': shares_outstanding,
            'current_eps': current_eps,
            'eps_low': eps_low,
            'eps_high': eps_high,
            'current_margin': current_margin,
            'historical_pe': historical_pe,
            'company_name': info.get('shortName', ticker),
            'fiscal_year': fiscal_year,
            'revenue_growth': revenue_growth if revenue_growth else 10.0,
            'news': news if news else [],
            'calendar': calendar,
            'earnings_dates': earnings_dates,
            # Valuation ratios
            'market_cap': market_cap,
            'enterprise_value': enterprise_value,
            'p_sales': p_sales,
            'p_book': p_book,
            'ev_ebitda': ev_ebitda,
            'ev_revenue': ev_revenue,
            'trailing_pe': trailing_pe,
            'forward_pe': forward_pe,
            'sector': sector,
            'industry': industry,
            # 52-week range
            'fifty_two_week_low': fifty_two_week_low,
            'fifty_two_week_high': fifty_two_week_high,
            # Financial statements
            'income_stmt': income_stmt,
            'balance_sheet': balance_sheet,
            'cash_flow': cash_flow,
            # Holders data
            'institutional_holders': institutional_holders,
            'major_holders': major_holders,
            'mutualfund_holders': mutualfund_holders,
            # Historical data for tables
            'historical_data': historical_data,
            # Analyst data
            'target_low': target_low,
            'target_mean': target_mean,
            'target_high': target_high,
            'target_median': target_median,
            'recommendation': recommendation,
            'recommendation_mean': recommendation_mean,
            'num_analysts': num_analysts,
            'recommendations': recommendations,
            # Insider data
            'insider_transactions': insider_transactions,
            'insider_purchases': insider_purchases,
            # DCF data
            'free_cash_flow': free_cash_flow,
            'operating_cash_flow': operating_cash_flow,
            'total_debt': total_debt,
            'total_cash': total_cash,
            # Financial Health data
            'total_stockholder_equity': total_stockholder_equity,
            'current_assets': current_assets,
            'current_liabilities': current_liabilities,
            'ebitda': ebitda,
            'ebit': ebit,  # For Magic Formula calculations
            'total_assets': total_assets,
            'retained_earnings': retained_earnings,
            # Graham Value data
            'book_value': book_value,
            'total_liabilities': info.get('totalLiab') or info.get('totalDebt', 0),
            # Dividend data for DDM
            'dividend_rate': info.get('dividendRate'),
            'dividend_yield': info.get('dividendYield'),
            'five_year_avg_dividend_yield': info.get('fiveYearAvgDividendYield'),
            'payout_ratio': info.get('payoutRatio'),
            'success': True,
            'error': None
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_treasury_yield() -> float:
    """Fetch the current 10-Year Treasury yield from yfinance."""
    try:
        treasury = yf.Ticker("^TNX")
        hist = treasury.history(period="1d")
        if not hist.empty:
            # ^TNX is quoted as a percentage (e.g., 4.5 for 4.5%)
            return hist['Close'].iloc[-1] / 100  # Convert to decimal
        return 0.045  # Default to 4.5% if fetch fails
    except:
        return 0.045  # Default to 4.5%


@st.cache_data(ttl=86400)  # Cache for 24 hours
def get_sec_cik(ticker: str) -> str:
    """Get SEC CIK number for a ticker symbol."""
    try:
        # SEC provides a mapping of tickers to CIKs
        headers = {'User-Agent': 'StockAnalysisTool/1.0 (contact@example.com)'}
        url = "https://www.sec.gov/files/company_tickers.json"
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            # Search for ticker (case-insensitive)
            for entry in data.values():
                if entry.get('ticker', '').upper() == ticker.upper():
                    # CIK needs to be zero-padded to 10 digits
                    cik = str(entry.get('cik_str', '')).zfill(10)
                    return cik
        return None
    except Exception:
        return None


@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_sec_filings(ticker: str, filing_types: list = None, limit: int = 10) -> list:
    """
    Fetch recent SEC filings for a company.
    
    Args:
        ticker: Stock ticker symbol
        filing_types: List of filing types to filter (e.g., ['10-K', '10-Q', '8-K'])
        limit: Maximum number of filings to return
    
    Returns:
        List of dicts with filing info: form, filingDate, accessionNumber, primaryDocument, url
    """
    if filing_types is None:
        filing_types = ['10-K', '10-Q', '8-K', '10-K/A', '10-Q/A']
    
    try:
        # Get CIK for the ticker
        cik = get_sec_cik(ticker)
        if not cik:
            return []
        
        # Fetch company submissions from SEC EDGAR
        headers = {'User-Agent': 'StockAnalysisTool/1.0 (contact@example.com)'}
        url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code != 200:
            return []
        
        data = response.json()
        filings = []
        
        # Recent filings are in the 'filings' -> 'recent' section
        recent = data.get('filings', {}).get('recent', {})
        forms = recent.get('form', [])
        filing_dates = recent.get('filingDate', [])
        accession_numbers = recent.get('accessionNumber', [])
        primary_documents = recent.get('primaryDocument', [])
        
        for i, form in enumerate(forms):
            if form in filing_types:
                accession = accession_numbers[i].replace('-', '')
                filing = {
                    'form': form,
                    'filingDate': filing_dates[i],
                    'accessionNumber': accession_numbers[i],
                    'primaryDocument': primary_documents[i],
                    'url': f"https://www.sec.gov/Archives/edgar/data/{cik.lstrip('0')}/{accession}/{primary_documents[i]}"
                }
                filings.append(filing)
                
                if len(filings) >= limit:
                    break
        
        return filings
    
    except Exception:
        return []

def calculate_dcf(
    free_cash_flow: float,
    fcf_growth_rate: float,  # As decimal (e.g., 0.15 for 15%)
    discount_rate: float,  # As decimal (e.g., 0.10 for 10%)
    terminal_growth_rate: float = 0.03,  # 3% default
    projection_years: int = 5,
    total_debt: float = 0,
    total_cash: float = 0,
    shares_outstanding: float = 1
) -> dict:
    """
    Calculate DCF intrinsic value per share.
    
    Returns dict with:
    - intrinsic_value: Fair value per share
    - pv_fcfs: Present value of projected FCFs
    - terminal_value: Present value of terminal value
    - enterprise_value: Total enterprise value
    - equity_value: Enterprise value - debt + cash
    - fcf_negative: Whether FCF is negative (warning flag)
    """
    if free_cash_flow is None or shares_outstanding is None or shares_outstanding == 0:
        return {'intrinsic_value': None, 'error': 'Missing data'}
    
    fcf_negative = free_cash_flow < 0
    
    # Project FCFs for each year and discount them
    projected_fcfs = []
    discounted_fcfs = []
    fcf = free_cash_flow
    
    for year in range(1, projection_years + 1):
        fcf = fcf * (1 + fcf_growth_rate)
        projected_fcfs.append(fcf)
        # Discount factor = 1 / (1 + r)^n
        discount_factor = 1 / ((1 + discount_rate) ** year)
        discounted_fcfs.append(fcf * discount_factor)
    
    pv_fcfs = sum(discounted_fcfs)
    
    # Terminal value using Gordon Growth Model
    # TV = FCF_final * (1 + g) / (r - g)
    final_fcf = projected_fcfs[-1]
    
    # Ensure discount rate > terminal growth rate to avoid division issues
    if discount_rate <= terminal_growth_rate:
        terminal_growth_rate = discount_rate - 0.01  # Adjust to be slightly below
    
    terminal_value = final_fcf * (1 + terminal_growth_rate) / (discount_rate - terminal_growth_rate)
    
    # Discount terminal value to present
    terminal_discount_factor = 1 / ((1 + discount_rate) ** projection_years)
    pv_terminal = terminal_value * terminal_discount_factor
    
    # Enterprise value = PV of FCFs + PV of Terminal Value
    enterprise_value = pv_fcfs + pv_terminal
    
    # Equity value = Enterprise Value - Debt + Cash
    equity_value = enterprise_value - total_debt + total_cash
    
    # Intrinsic value per share
    intrinsic_value = equity_value / shares_outstanding
    
    return {
        'intrinsic_value': intrinsic_value,
        'pv_fcfs': pv_fcfs,
        'pv_terminal': pv_terminal,
        'enterprise_value': enterprise_value,
        'equity_value': equity_value,
        'fcf_negative': fcf_negative,
        'discount_rate': discount_rate,
        'projected_fcfs': projected_fcfs,
        'error': None
    }

def calculate_ddm(
    dividend_rate: float,
    current_price: float,
    dividend_growth_rate: float = 0.03,  # 3% default growth
    required_return: float = 0.10  # 10% default required return
) -> dict:
    """
    Calculate Dividend Discount Model (Gordon Growth Model) intrinsic value.
    
    Formula: Intrinsic Value = Dividend / (required_return - dividend_growth)
    
    Returns None for non-dividend stocks.
    """
    if dividend_rate is None or dividend_rate <= 0:
        return {'intrinsic_value': None, 'error': 'No dividend', 'is_dividend_stock': False}
    
    if required_return <= dividend_growth_rate:
        return {'intrinsic_value': None, 'error': 'Growth exceeds required return', 'is_dividend_stock': True}
    
    intrinsic_value = dividend_rate / (required_return - dividend_growth_rate)
    
    upside_pct = None
    if current_price and current_price > 0:
        upside_pct = ((intrinsic_value / current_price) - 1) * 100
    
    return {
        'intrinsic_value': intrinsic_value,
        'dividend_rate': dividend_rate,
        'dividend_growth_rate': dividend_growth_rate,
        'required_return': required_return,
        'upside_pct': upside_pct,
        'is_dividend_stock': True,
        'error': None
    }

def calculate_multiples_value(
    eps: float,
    current_price: float,
    sector: str = 'Unknown'
) -> dict:
    """
    Calculate fair value using sector average P/E multiple.
    
    Formula: Fair Value = Sector_Avg_PE × EPS
    """
    sector_multiples = {
        'Technology': {'pe': 28, 'pb': 6.0},
        'Healthcare': {'pe': 22, 'pb': 4.0},
        'Financial Services': {'pe': 12, 'pb': 1.3},
        'Consumer Cyclical': {'pe': 18, 'pb': 3.5},
        'Consumer Defensive': {'pe': 20, 'pb': 4.0},
        'Industrials': {'pe': 20, 'pb': 3.5},
        'Energy': {'pe': 12, 'pb': 1.5},
        'Real Estate': {'pe': 35, 'pb': 2.0},
        'Communication Services': {'pe': 18, 'pb': 3.0},
        'Basic Materials': {'pe': 14, 'pb': 2.0},
        'Utilities': {'pe': 18, 'pb': 1.8},
    }
    default_multiples = {'pe': 20, 'pb': 3.0}
    
    if eps is None or eps <= 0:
        return {'intrinsic_value': None, 'error': 'No positive EPS', 'sector_pe': None}
    
    multiples = sector_multiples.get(sector, default_multiples)
    sector_pe = multiples['pe']
    
    intrinsic_value = sector_pe * eps
    
    upside_pct = None
    if current_price and current_price > 0:
        upside_pct = ((intrinsic_value / current_price) - 1) * 100
    
    return {
        'intrinsic_value': intrinsic_value,
        'sector_pe': sector_pe,
        'sector': sector,
        'upside_pct': upside_pct,
        'error': None
    }

def calculate_reverse_dcf(
    current_price: float,
    shares_outstanding: float,
    free_cash_flow: float,
    discount_rate: float,
    terminal_growth_rate: float = 0.03,
    projection_years: int = 5,
    total_debt: float = 0,
    total_cash: float = 0
) -> dict:
    """
    Calculate the implied FCF growth rate that justifies the current market price.
    Uses bisection method to solve for growth rate.
    """
    if (current_price is None or current_price <= 0 or 
        shares_outstanding is None or shares_outstanding <= 0 or
        free_cash_flow is None or free_cash_flow <= 0):
        return {'implied_growth': None, 'error': 'Missing data or negative FCF'}
    
    # Target equity value based on current price
    target_equity = current_price * shares_outstanding
    
    # Bisection search for growth rate
    low, high = -0.50, 1.00  # -50% to 100% growth range
    tolerance = 0.0001
    max_iterations = 50
    
    for _ in range(max_iterations):
        mid = (low + high) / 2
        
        # Calculate DCF value at this growth rate
        fcf = free_cash_flow
        discounted_fcfs = []
        for year in range(1, projection_years + 1):
            fcf = fcf * (1 + mid)
            discount_factor = 1 / ((1 + discount_rate) ** year)
            discounted_fcfs.append(fcf * discount_factor)
        
        pv_fcfs = sum(discounted_fcfs)
        
        # Terminal value
        final_fcf = fcf
        if discount_rate > terminal_growth_rate:
            terminal_value = final_fcf * (1 + terminal_growth_rate) / (discount_rate - terminal_growth_rate)
            terminal_discount = 1 / ((1 + discount_rate) ** projection_years)
            pv_terminal = terminal_value * terminal_discount
        else:
            pv_terminal = 0
        
        enterprise_value = pv_fcfs + pv_terminal
        equity_value = enterprise_value - total_debt + total_cash
        
        if abs(equity_value - target_equity) < tolerance * target_equity:
            break
        
        if equity_value < target_equity:
            low = mid
        else:
            high = mid
    
    implied_growth = mid
    
    return {
        'implied_growth': implied_growth,
        'implied_growth_pct': implied_growth * 100,
        'error': None
    }

def calculate_dynamic_growth_rate(data: dict) -> dict:
    """
    Calculate a dynamic FCF growth rate based on blended historical and analyst estimates.
    
    Blend:
    - Historical Revenue CAGR (3-5 years): 40% weight
    - Historical FCF growth: 30% weight
    - Analyst Revenue Growth Estimate: 30% weight
    
    Guardrails: Floor 3%, Cap 25%, Default 10%
    """
    components = []
    weights = []
    
    # Sector-based defaults as fallback
    sector_defaults = {
        'Technology': 0.15,
        'Healthcare': 0.12,
        'Consumer Cyclical': 0.10,
        'Consumer Defensive': 0.06,
        'Financial Services': 0.08,
        'Industrials': 0.08,
        'Energy': 0.06,
        'Utilities': 0.04,
        'Real Estate': 0.06,
        'Communication Services': 0.10,
        'Basic Materials': 0.07,
    }
    
    # 1. Historical Revenue Growth (from yfinance revenueGrowth - YoY)
    revenue_growth = data.get('revenue_growth')
    if revenue_growth is not None and revenue_growth != 0:
        # revenue_growth is in percentage (e.g., 10.0 for 10%)
        rev_growth_decimal = revenue_growth / 100 if abs(revenue_growth) > 1 else revenue_growth
        # Sanity check
        if -0.5 <= rev_growth_decimal <= 1.0:
            components.append(rev_growth_decimal)
            weights.append(0.4)
    
    # 2. Analyst estimates (forward revenue growth or earnings growth)
    # Use forward P/E vs trailing P/E as proxy for expected earnings growth
    forward_pe = data.get('forward_pe')
    trailing_pe = data.get('trailing_pe')
    if forward_pe and trailing_pe and forward_pe > 0 and trailing_pe > 0:
        # If forward PE < trailing PE, earnings expected to grow
        implied_earnings_growth = (trailing_pe / forward_pe) - 1
        if -0.3 <= implied_earnings_growth <= 0.5:
            components.append(implied_earnings_growth)
            weights.append(0.3)
    
    # 3. Use revenue growth again as FCF proxy if we don't have FCF history
    # (FCF tends to track revenue growth for mature companies)
    if revenue_growth is not None and revenue_growth != 0:
        rev_growth_decimal = revenue_growth / 100 if abs(revenue_growth) > 1 else revenue_growth
        if -0.5 <= rev_growth_decimal <= 1.0:
            components.append(rev_growth_decimal)
            weights.append(0.3)
    
    # Calculate weighted average if we have components
    if components and weights:
        total_weight = sum(weights)
        blended_growth = sum(c * w for c, w in zip(components, weights)) / total_weight
    else:
        # Fallback to sector default
        sector = data.get('sector', 'Unknown')
        blended_growth = sector_defaults.get(sector, 0.10)
    
    # Apply guardrails
    final_growth = max(0.03, min(0.25, blended_growth))
    
    return {
        'growth_rate': final_growth,
        'growth_rate_pct': final_growth * 100,
        'components_used': len(components),
        'sector_fallback': len(components) == 0
    }

def calculate_dcf_sensitivity(
    free_cash_flow: float,
    base_growth_rate: float,
    discount_rate: float,
    terminal_growth_rate: float,
    projection_years: int,
    total_debt: float,
    total_cash: float,
    shares_outstanding: float,
    current_price: float
) -> list:
    """
    Calculate DCF at multiple growth rate scenarios (10%, 15%, 20%).
    Returns list of dicts with growth_rate, value, and upside.
    """
    if free_cash_flow is None or shares_outstanding is None or shares_outstanding <= 0:
        return []
    
    # Fixed scenarios: 10%, 15%, 20% (matching reference model)
    scenarios = [
        ('Low', 0.10),
        ('Medium', 0.15),
        ('High', 0.20),
    ]
    
    results = []
    for label, growth_rate in scenarios:
        dcf = calculate_dcf(
            free_cash_flow=free_cash_flow,
            fcf_growth_rate=growth_rate,
            discount_rate=discount_rate,
            terminal_growth_rate=terminal_growth_rate,
            projection_years=projection_years,
            total_debt=total_debt,
            total_cash=total_cash,
            shares_outstanding=shares_outstanding
        )
        
        if dcf.get('intrinsic_value') is not None:
            value = dcf['intrinsic_value']
            upside = ((value / current_price) - 1) * 100 if current_price and current_price > 0 else 0
            results.append({
                'label': label,
                'growth_rate': growth_rate * 100,
                'value': value,
                'upside': upside
            })
    
    return results

def calculate_combined_intrinsic(
    dcf_value: float,
    ddm_value: float,
    multiples_value: float
) -> dict:
    """
    Calculate combined intrinsic value as average of available valuation methods.
    Excludes Graham (shown separately per user request).
    """
    values = []
    methods = []
    
    if dcf_value is not None and dcf_value > 0:
        values.append(dcf_value)
        methods.append('DCF')
    
    if ddm_value is not None and ddm_value > 0:
        values.append(ddm_value)
        methods.append('DDM')
    
    if multiples_value is not None and multiples_value > 0:
        values.append(multiples_value)
        methods.append('Multiples')
    
    if not values:
        return {'combined_value': None, 'methods_used': [], 'count': 0}
    
    combined = sum(values) / len(values)
    
    return {
        'combined_value': combined,
        'methods_used': methods,
        'count': len(values),
        'individual_values': dict(zip(methods, values))
    }

def calculate_financial_health(data: dict) -> dict:
    """
    Calculate Financial Health Score based on key financial ratios.
    
    Components:
    1. Debt-to-Equity Ratio - Lower is better
    2. Current Ratio - Measures short-term liquidity (>1.5 is healthy)
    3. Interest Coverage Ratio - Ability to pay interest (higher is better)
    4. Altman Z-Score - Bankruptcy risk predictor
    
    Returns dict with individual metrics and overall letter grade.
    """
    results = {
        'debt_to_equity': None,
        'current_ratio': None,
        'interest_coverage': None,
        'altman_z': None,
        'overall_score': None,
        'letter_grade': None,
        'health_status': None,
        'component_scores': {}
    }
    
    # Extract data
    total_debt = data.get('total_debt', 0) or 0
    total_equity = data.get('total_stockholder_equity')
    current_assets = data.get('current_assets')
    current_liabilities = data.get('current_liabilities')
    ebitda = data.get('ebitda')
    total_assets = data.get('total_assets')
    total_revenue = data.get('total_revenue')
    retained_earnings = data.get('retained_earnings')
    market_cap = data.get('market_cap')
    net_income = data.get('net_income')
    
    component_scores = []
    
    # 1. Debt-to-Equity Ratio (lower is better)
    if total_equity and total_equity > 0:
        debt_to_equity = total_debt / total_equity
        results['debt_to_equity'] = debt_to_equity
        # Score: 0 D/E = 100, 0.5 D/E = 80, 1.0 D/E = 60, 2.0 D/E = 40, 3+ D/E = 20
        if debt_to_equity <= 0.3:
            de_score = 100
        elif debt_to_equity <= 0.5:
            de_score = 90
        elif debt_to_equity <= 1.0:
            de_score = 75
        elif debt_to_equity <= 2.0:
            de_score = 55
        elif debt_to_equity <= 3.0:
            de_score = 35
        else:
            de_score = 20
        results['component_scores']['debt_to_equity'] = de_score
        component_scores.append(de_score)
    
    # 2. Current Ratio (higher is better, but too high can indicate inefficiency)
    if current_assets and current_liabilities and current_liabilities > 0:
        current_ratio = current_assets / current_liabilities
        results['current_ratio'] = current_ratio
        # Score: <1.0 = 30, 1.0-1.5 = 60, 1.5-2.5 = 90, 2.5-3.5 = 80, >3.5 = 70
        if current_ratio < 1.0:
            cr_score = 30
        elif current_ratio < 1.5:
            cr_score = 60
        elif current_ratio <= 2.5:
            cr_score = 95
        elif current_ratio <= 3.5:
            cr_score = 80
        else:
            cr_score = 70  # Too much idle capital
        results['component_scores']['current_ratio'] = cr_score
        component_scores.append(cr_score)
    
    # 3. Interest Coverage Ratio (EBITDA / Interest Expense approximation)
    # We'll estimate interest expense from debt * average rate (~5%)
    if ebitda and total_debt and total_debt > 0:
        estimated_interest = total_debt * 0.05  # Assume 5% average interest rate
        interest_coverage = ebitda / estimated_interest if estimated_interest > 0 else 999
        results['interest_coverage'] = interest_coverage
        # Score: <1 = 10, 1-2 = 40, 2-4 = 60, 4-8 = 80, >8 = 95
        if interest_coverage < 1:
            ic_score = 10
        elif interest_coverage < 2:
            ic_score = 40
        elif interest_coverage < 4:
            ic_score = 60
        elif interest_coverage < 8:
            ic_score = 80
        else:
            ic_score = 95
        results['component_scores']['interest_coverage'] = ic_score
        component_scores.append(ic_score)
    elif ebitda and (total_debt == 0 or total_debt is None):
        # No debt = excellent interest coverage
        results['interest_coverage'] = 999
        results['component_scores']['interest_coverage'] = 100
        component_scores.append(100)
    
    # 4. Altman Z-Score (for public companies)
    # Z = 1.2*A + 1.4*B + 3.3*C + 0.6*D + 1.0*E
    # A = Working Capital / Total Assets
    # B = Retained Earnings / Total Assets
    # C = EBIT / Total Assets (we'll use EBITDA as proxy)
    # D = Market Cap / Total Liabilities
    # E = Sales / Total Assets
    if total_assets and total_assets > 0:
        working_capital = (current_assets or 0) - (current_liabilities or 0)
        total_liabilities = total_assets - (total_equity or 0)
        
        A = working_capital / total_assets if total_assets else 0
        B = (retained_earnings or 0) / total_assets if total_assets else 0
        C = (ebitda or 0) / total_assets if total_assets else 0
        D = (market_cap or 0) / total_liabilities if total_liabilities and total_liabilities > 0 else 5
        E = (total_revenue or 0) / total_assets if total_assets else 0
        
        altman_z = 1.2 * A + 1.4 * B + 3.3 * C + 0.6 * D + 1.0 * E
        results['altman_z'] = altman_z
        
        # Score: <1.1 = 10 (distress), 1.1-1.8 = 35, 1.8-2.7 = 55, 2.7-3.0 = 75, >3.0 = 95 (safe)
        if altman_z < 1.1:
            az_score = 10
        elif altman_z < 1.8:
            az_score = 35
        elif altman_z < 2.7:
            az_score = 55
        elif altman_z < 3.0:
            az_score = 75
        else:
            az_score = 95
        results['component_scores']['altman_z'] = az_score
        component_scores.append(az_score)
    
    # Calculate overall score (weighted average)
    if component_scores:
        # Weight: D/E 25%, Current Ratio 20%, Interest Coverage 25%, Altman Z 30%
        weights = {
            'debt_to_equity': 0.25,
            'current_ratio': 0.20,
            'interest_coverage': 0.25,
            'altman_z': 0.30
        }
        
        weighted_sum = 0
        weight_total = 0
        for key, score in results['component_scores'].items():
            if key in weights:
                weighted_sum += score * weights[key]
                weight_total += weights[key]
        
        if weight_total > 0:
            overall_score = weighted_sum / weight_total
            results['overall_score'] = overall_score
            
            # Convert to letter grade
            if overall_score >= 90:
                results['letter_grade'] = 'A'
                results['health_status'] = 'Excellent'
            elif overall_score >= 80:
                results['letter_grade'] = 'B+'
                results['health_status'] = 'Very Good'
            elif overall_score >= 70:
                results['letter_grade'] = 'B'
                results['health_status'] = 'Good'
            elif overall_score >= 60:
                results['letter_grade'] = 'C+'
                results['health_status'] = 'Fair'
            elif overall_score >= 50:
                results['letter_grade'] = 'C'
                results['health_status'] = 'Below Average'
            elif overall_score >= 40:
                results['letter_grade'] = 'D'
                results['health_status'] = 'Poor'
            else:
                results['letter_grade'] = 'F'
                results['health_status'] = 'Distressed'
    
    return results


def calculate_graham_value(data: dict) -> dict:
    """
    Calculate Graham Value metrics for value investing analysis.
    
    Uses sector-adjusted multipliers instead of Graham's original 22.5.
    Modern Graham Number = √(Sector_PE × Sector_PB × EPS × Book Value)
    
    Metrics:
    1. Graham Number - √(Sector_PE × Sector_PB × EPS × Book Value per Share)
    2. NCAV per Share - (Current Assets - Total Liabilities) / Shares
    3. P/E × P/B - Combined valuation ratio vs sector threshold
    4. Earnings Yield - EPS / Price (compare to bond yields)
    """
    # Sector-based average P/E and P/B ratios (approximate current market averages)
    sector_multiples = {
        'Technology': {'pe': 28, 'pb': 6.0},
        'Healthcare': {'pe': 22, 'pb': 4.0},
        'Financial Services': {'pe': 12, 'pb': 1.3},
        'Consumer Cyclical': {'pe': 20, 'pb': 4.0},
        'Consumer Defensive': {'pe': 22, 'pb': 5.0},
        'Industrials': {'pe': 20, 'pb': 4.0},
        'Energy': {'pe': 12, 'pb': 1.8},
        'Communication Services': {'pe': 18, 'pb': 3.0},
        'Real Estate': {'pe': 35, 'pb': 2.0},
        'Basic Materials': {'pe': 14, 'pb': 2.0},
        'Utilities': {'pe': 18, 'pb': 1.8},
    }
    default_multiples = {'pe': 20, 'pb': 3.0}  # Market average fallback
    
    sector = data.get('sector', 'Unknown')
    multiples = sector_multiples.get(sector, default_multiples)
    sector_pe = multiples['pe']
    sector_pb = multiples['pb']
    sector_threshold = sector_pe * sector_pb  # Sector-adjusted P/E × P/B threshold
    
    results = {
        'graham_number': None,
        'ncav_per_share': None,
        'pe_times_pb': None,
        'earnings_yield': None,
        'graham_margin': None,  # % difference from Graham Number
        'ncav_margin': None,    # % difference from NCAV
        'is_undervalued': False,
        'is_net_net': False,    # Trading below NCAV
        'sector_pe': sector_pe,
        'sector_pb': sector_pb,
        'sector_threshold': sector_threshold,
        'sector': sector,
    }
    
    current_price = data.get('current_price')
    eps = data.get('current_eps')
    book_value = data.get('book_value')  # Book value per share
    shares = data.get('shares_outstanding')
    current_assets = data.get('current_assets')
    total_liabilities = data.get('total_liabilities')
    trailing_pe = data.get('trailing_pe')
    p_book = data.get('p_book')
    
    if not current_price or current_price <= 0:
        return results
    
    # 1. Modern Graham Number: √(Sector_PE × Sector_PB × EPS × BVPS)
    # Uses sector-appropriate multiples instead of fixed 22.5
    if eps and eps > 0 and book_value and book_value > 0:
        graham_multiplier = sector_pe * sector_pb
        graham_number = (graham_multiplier * eps * book_value) ** 0.5
        results['graham_number'] = graham_number
        
        # Margin vs current price (positive = undervalued)
        results['graham_margin'] = ((graham_number / current_price) - 1) * 100
        
        if current_price < graham_number:
            results['is_undervalued'] = True
    
    # 2. NCAV per Share: (Current Assets - Total Liabilities) / Shares
    if current_assets and total_liabilities is not None and shares and shares > 0:
        ncav = current_assets - total_liabilities
        ncav_per_share = ncav / shares
        results['ncav_per_share'] = ncav_per_share
        
        # Margin vs current price
        if ncav_per_share > 0:
            results['ncav_margin'] = ((ncav_per_share / current_price) - 1) * 100
        
        # Net-net: trading below NCAV (rare bargain)
        if ncav_per_share > 0 and current_price < ncav_per_share:
            results['is_net_net'] = True
    
    # 3. P/E × P/B vs sector threshold (instead of fixed 22.5)
    if trailing_pe and trailing_pe > 0 and p_book and p_book > 0:
        results['pe_times_pb'] = trailing_pe * p_book
    
    # 4. Earnings Yield (inverse of P/E, as percentage)
    if eps and current_price > 0:
        results['earnings_yield'] = (eps / current_price) * 100
    
    return results


def calculate_projections(data: dict, revenue_growth: float, net_margin: float) -> pd.DataFrame:
    """Calculate 5-year projections."""
    current_price = data['current_price']
    revenue = data['total_revenue']
    shares = data['shares_outstanding']
    eps_low = data['eps_low']
    eps_high = data['eps_high']
    historical_pe = data['historical_pe']
    fiscal_year = data.get('fiscal_year', 2024)
    fifty_two_week_low = data.get('fifty_two_week_low')
    fifty_two_week_high = data.get('fifty_two_week_high')
    
    # Calculate P/E bounds from analyst estimates
    pe_high = None
    pe_low = None
    using_fallback = False
    
    if eps_low and eps_high and eps_low > 0 and eps_high > 0:
        pe_high = current_price / eps_low  # High P/E when earnings are low
        pe_low = current_price / eps_high   # Low P/E when earnings are high
    elif historical_pe and historical_pe > 0:
        # Fallback to historical P/E with +/- 20% range
        pe_low = historical_pe * 0.8
        pe_high = historical_pe * 1.2
        using_fallback = True
    
    # Build projection table - Years 0-5 (current + 5 future years)
    num_years = 6
    calendar_years = [fiscal_year + i for i in range(num_years)]
    projections = {
        'Year': calendar_years,
        'Revenue': [],
        'Rev Growth %': [],
        'Net Income': [],
        'Net Inc Growth %': [],
        'Net Inc Margins': [],
        'EPS': [],
        'P/E Low Est': [],
        'P/E High Est': [],
        'Share Price Low': [],
        'Share Price High': [],
    }
    
    prev_net_income = data['net_income']
    base_net_income = data['net_income']
    base_eps = data['current_eps']
    
    for idx in range(num_years):
        if idx == 0:
            # Current year (Year 0) - use 52-week low/high for share prices
            proj_revenue = revenue
            proj_net_income = data['net_income']
            proj_eps = base_eps
            
            # Calculate Year 0 growth from historical data (compare to prior year)
            historical_data = data.get('historical_data', [])
            rev_growth = None
            ni_growth = None
            if historical_data and len(historical_data) >= 1:
                # Historical data is ordered most recent first, so index 0 is the prior year
                prior_year = historical_data[0]
                prior_revenue = prior_year.get('revenue')
                prior_net_income = prior_year.get('net_income')
                
                if prior_revenue and prior_revenue > 0 and proj_revenue:
                    rev_growth = ((proj_revenue / prior_revenue) - 1) * 100
                if prior_net_income and prior_net_income != 0 and proj_net_income:
                    ni_growth = ((proj_net_income / prior_net_income) - 1) * 100
            
            margin = data['current_margin']
            price_low = fifty_two_week_low if fifty_two_week_low else current_price
            price_high = fifty_two_week_high if fifty_two_week_high else current_price
        else:
            # Future years
            proj_revenue = revenue * ((1 + revenue_growth / 100) ** idx)
            proj_net_income = proj_revenue * (net_margin / 100)
            
            # Grow EPS from current EPS using net income growth rate
            # This avoids ADR/foreign stock share count mismatches
            if base_net_income and base_net_income > 0 and base_eps:
                ni_growth_multiplier = proj_net_income / base_net_income
                proj_eps = base_eps * ni_growth_multiplier
            elif shares:
                proj_eps = proj_net_income / shares
            else:
                proj_eps = None
            
            rev_growth = revenue_growth
            
            # Net Income Growth %
            if prev_net_income and prev_net_income > 0:
                ni_growth = ((proj_net_income / prev_net_income) - 1) * 100
            else:
                ni_growth = None
            
            margin = net_margin
            
            # Share price projections
            if pe_low and pe_high and proj_eps:
                price_low = proj_eps * pe_low
                price_high = proj_eps * pe_high
            else:
                price_low = None
                price_high = None
        
        projections['Revenue'].append(proj_revenue)
        projections['Rev Growth %'].append(rev_growth)
        projections['Net Income'].append(proj_net_income)
        projections['Net Inc Growth %'].append(ni_growth)
        projections['Net Inc Margins'].append(margin)
        projections['EPS'].append(proj_eps)
        projections['P/E Low Est'].append(pe_low)
        projections['P/E High Est'].append(pe_high)
        projections['Share Price Low'].append(price_low)
        projections['Share Price High'].append(price_high)
        
        prev_net_income = proj_net_income
    
    # Calculate CAGR for final year (Year 5)
    cagr_low = None
    cagr_high = None
    final_idx = num_years - 1
    if projections['Share Price Low'][final_idx] and current_price and current_price > 0:
        cagr_low = ((projections['Share Price Low'][final_idx] / current_price) ** (1/final_idx) - 1) * 100
    if projections['Share Price High'][final_idx] and current_price and current_price > 0:
        cagr_high = ((projections['Share Price High'][final_idx] / current_price) ** (1/final_idx) - 1) * 100
    
    return projections, pe_low, pe_high, cagr_low, cagr_high, using_fallback

def build_scenario_table(data: dict, revenue_growth: float, net_margin: float, show_history: bool = False) -> pd.DataFrame:
    """Build a formatted DataFrame for a scenario."""
    projections, pe_low, pe_high, cagr_low, cagr_high, _ = calculate_projections(
        data, revenue_growth, net_margin
    )
    
    # Get historical data if requested
    historical_data = data.get('historical_data', []) if show_history else []
    # Reverse to show oldest first (chronological order)
    historical_data = list(reversed(historical_data))
    
    display_data = []
    metrics = [
        ('Revenue', 'Revenue', format_large_number),
        ('Rev Growth %', 'Rev Growth', lambda x: format_percent(x) if x else "—"),
        ('Net Income', 'Net Income', format_large_number),
        ('Net Inc Growth %', 'Net Inc. Growth', lambda x: format_percent(x) if x else "—"),
        ('Net Inc Margins', 'Net Inc. Margins', lambda x: format_percent(x) if x else "—"),
        ('EPS', 'EPS', format_currency),
        ('P/E Low Est', 'P/E Low Est', lambda x: f"{x:.0f}" if x else "—"),
        ('P/E High Est', 'P/E High Est', lambda x: f"{x:.0f}" if x else "—"),
        ('Share Price Low', 'Share Price Low', format_currency),
        ('Share Price High', 'Share Price High', format_currency),
    ]
    
    # Historical data mapping
    hist_metric_map = {
        'Revenue': 'revenue',
        'Rev Growth %': 'rev_growth',
        'Net Income': 'net_income',
        'Net Inc Growth %': 'ni_growth',
        'Net Inc Margins': 'margin',
        'EPS': 'eps',
        'P/E Low Est': 'pe',
        'P/E High Est': 'pe',
        'Share Price Low': 'price_low',
        'Share Price High': 'price_high',
    }
    
    for metric_key, metric_name, formatter in metrics:
        row = {'Metric': metric_name}
        
        # Add historical columns first (if enabled)
        if show_history and historical_data:
            hist_key = hist_metric_map.get(metric_key)
            for hist in historical_data:
                year = hist['year']
                if hist_key:
                    value = hist.get(hist_key)
                    row[f'{year}'] = formatter(value) if value is not None else "—"
                else:
                    row[f'{year}'] = "—"  # Metrics not available in history (EPS, P/E, etc.)
        
        # Add projection columns
        for i, year in enumerate(projections['Year']):
            row[f'{year}'] = formatter(projections[metric_key][i])
        display_data.append(row)
    
    # Add CAGR rows
    num_years = len(projections['Year'])
    cagr_low_row = {'Metric': 'CAGR Low'}
    cagr_high_row = {'Metric': 'CAGR High'}
    
    # Historical columns for CAGR (just dashes)
    if show_history and historical_data:
        for hist in historical_data:
            year = hist['year']
            cagr_low_row[f'{year}'] = "—"
            cagr_high_row[f'{year}'] = "—"
    
    for i, year in enumerate(projections['Year']):
        if i == 0:
            cagr_low_row[f'{year}'] = "—"
            cagr_high_row[f'{year}'] = "—"
        elif i > 0 and projections['Share Price Low'][i] and data['current_price']:
            year_cagr_low = ((projections['Share Price Low'][i] / data['current_price']) ** (1/i) - 1) * 100
            year_cagr_high = ((projections['Share Price High'][i] / data['current_price']) ** (1/i) - 1) * 100
            cagr_low_row[f'{year}'] = f"{year_cagr_low:.0f}%"
            cagr_high_row[f'{year}'] = f"{year_cagr_high:.0f}%"
        else:
            cagr_low_row[f'{year}'] = "—"
            cagr_high_row[f'{year}'] = "—"
    display_data.append(cagr_low_row)
    display_data.append(cagr_high_row)
    
    df = pd.DataFrame(display_data)
    df = df.set_index('Metric')
    return df, projections, cagr_low, cagr_high

# --- Page Navigation & Watchlist Selector ---
nav_col1, nav_col2, nav_col3, nav_col4, nav_col5, nav_col6, nav_col7 = st.columns([1, 1, 1, 1, 1, 1, 1])
with nav_col1:
    st.markdown("**📊 Stock Analysis** *(current)*")
with nav_col2:
    st.page_link("pages/1_Watchlist_Comparison.py", label="👉 Watchlist")
with nav_col3:
    st.page_link("pages/2_Monte_Carlo.py", label="👉 Monte Carlo")
with nav_col4:
    st.page_link("pages/3_Cross_Asset_Dashboard.py", label="👉 Cross-Asset")
with nav_col7:
    st.page_link("pages/4_Magic_Screener.py", label="👉 Magic Screener")
with nav_col5:
    # Watchlist selector
    watchlist_names = get_watchlist_names()
    current_idx = watchlist_names.index(st.session_state.selected_watchlist) if st.session_state.selected_watchlist in watchlist_names else 0
    new_watchlist = st.selectbox(
        "📋 Watchlist",
        options=watchlist_names,
        index=current_idx,
        key="watchlist_selector",
        label_visibility="collapsed"
    )
    if new_watchlist != st.session_state.selected_watchlist:
        st.session_state.selected_watchlist = new_watchlist
        # Clear cached data when watchlist changes
        if 'show_watchlist_calendar' in st.session_state:
            st.session_state.show_watchlist_calendar = False
        if 'watchlist_news_loaded' in st.session_state:
            st.session_state.watchlist_news_loaded = False
        st.rerun()

# Update WATCHLIST_STOCKS based on current selection
WATCHLIST_STOCKS = get_current_watchlist_stocks()

st.markdown("---")

# --- TOP HEADER WITH SEARCH ---
st.title("📈 Stock Valuation Projection Engine")
st.markdown("*5-Year Forward-Estimate Model with Bull, Base & Bear Scenarios*")

# --- Watchlist Calendar Section (at top) ---
with st.expander("📅 Watchlist Calendar", expanded=False):
    if st.button("🔄 Load Watchlist Events", key="load_watchlist_calendar", use_container_width=True, help="View upcoming events for all watchlist stocks"):
        st.session_state.show_watchlist_calendar = True
    
    if st.session_state.get('show_watchlist_calendar', False):
        st.markdown("---")
        
        # Fetch events for all watchlist stocks
        all_events = []
        with st.spinner("Loading calendar events..."):
            for stock_ticker in WATCHLIST_STOCKS[1:]:  # Skip empty string
                events = fetch_calendar_events(stock_ticker)
                all_events.extend(events)
        
        # Remove duplicates and sort by date
        unique_events = list(set(all_events))
        unique_events.sort(key=lambda x: x[0])
        
        if unique_events:
            from collections import defaultdict
            from datetime import datetime, timedelta
            import calendar
            
            # Group events by date
            events_by_date = defaultdict(list)
            for event_date, event_type, ticker in unique_events:
                events_by_date[event_date].append((event_type, ticker))
            
            # Get date range for calendar (current month + next 2 months)
            today = datetime.now().date()
            
            # Build calendar for 3 months
            for month_offset in range(3):
                # Calculate the month
                month_date = today.replace(day=1)
                for _ in range(month_offset):
                    # Move to next month
                    if month_date.month == 12:
                        month_date = month_date.replace(year=month_date.year + 1, month=1)
                    else:
                        month_date = month_date.replace(month=month_date.month + 1)
                
                year = month_date.year
                month = month_date.month
                month_name = calendar.month_name[month]
                
                st.markdown(f"### {month_name} {year}")
                
                # Create calendar grid
                cal = calendar.Calendar(firstweekday=6)  # Start on Sunday
                month_days = cal.monthdayscalendar(year, month)
                
                # Header row
                header_cols = st.columns(7)
                for i, day_name in enumerate(['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']):
                    header_cols[i].markdown(f"**{day_name}**")
                
                # Calendar grid with scrollable container
                for week in month_days:
                    week_cols = st.columns(7)
                    for i, day in enumerate(week):
                        if day == 0:
                            week_cols[i].markdown("")
                        else:
                            try:
                                cell_date = datetime(year, month, day).date()
                                events_today = events_by_date.get(cell_date, [])
                                
                                if events_today:
                                    # Has events - highlight
                                    event_text = f"**{day}**\n"
                                    for event_type, ticker in events_today[:2]:  # Max 2 per cell
                                        emoji = "📊" if event_type == "Earnings" else ("💰" if event_type == "Dividend" else "📆")
                                        event_text += f"{emoji}{ticker}\n"
                                    if len(events_today) > 2:
                                        event_text += f"+{len(events_today)-2} more"
                                    week_cols[i].markdown(event_text)
                                elif cell_date == today:
                                    # Today
                                    week_cols[i].markdown(f"🔵 **{day}**")
                                else:
                                    week_cols[i].markdown(f"{day}")
                            except:
                                week_cols[i].markdown(f"{day}")
                
                st.markdown("---")
            
            # Legend
            st.markdown("**Legend:** 📊 Earnings | 💰 Dividend | 📆 Ex-Dividend | 🔵 Today")
        else:
            st.info("No upcoming events found for watchlist stocks.")
        
        if st.button("✕ Close", key="close_watchlist_calendar"):
            st.session_state.show_watchlist_calendar = False
            st.rerun()

# --- Watchlist News Feed Expander (at top) ---
with st.expander("📰 Watchlist News Feed (Last 48 Hours)", expanded=False):
    if st.button("🔄 Load Latest News", key="refresh_watchlist_news", use_container_width=True):
        st.session_state.watchlist_news_loaded = True
    
    if st.session_state.get('watchlist_news_loaded', False):
        watchlist_news = fetch_watchlist_news()
        
        if watchlist_news:
            from datetime import datetime
            
            # Calculate minutes ago for each article and sort by that
            for article in watchlist_news:
                time_diff = datetime.now() - article['timestamp']
                article['minutes_ago'] = abs(int(time_diff.total_seconds() / 60))
            
            # Sort by minutes_ago (smallest = most recent first)
            sorted_news = sorted(watchlist_news, key=lambda x: x['minutes_ago'])
            
            for article in sorted_news[:20]:  # Limit to 20 articles
                ticker = article['ticker']
                title = article['title']
                link = article['link']
                publisher = article.get('publisher', '')
                total_minutes = article['minutes_ago']
                
                # Format time ago
                if total_minutes < 60:
                    time_str = f"{total_minutes}m ago"
                elif total_minutes < 1440:  # Less than 24 hours
                    hours = total_minutes // 60
                    time_str = f"{hours}h ago"
                else:
                    days = total_minutes // 1440
                    time_str = f"{days}d ago"
                
                # Display with publisher if available
                if publisher:
                    st.markdown(f"**[{ticker}]** [{title}]({link})")
                    st.caption(f"🔗 {publisher} • ⏰ {time_str}")
                else:
                    st.markdown(f"**[{ticker}]** [{title}]({link}) • ⏰ {time_str}")
        else:
            st.info("No recent news found for watchlist stocks.")
    else:
        st.caption("Click 'Load Latest News' to fetch stories from all watchlist stocks.")

# Search bar at the top with watchlist dropdown and Fetch Data button
col_watchlist, col_search, col_fetch = st.columns([2, 3, 1])
with col_watchlist:
    selected_watchlist = st.selectbox(
        "📋 Watchlist", 
        options=WATCHLIST_STOCKS,
        format_func=lambda x: "Select from watchlist..." if x == "" else x,
        label_visibility="collapsed"
    )
with col_search:
    # Use watchlist selection if available, otherwise allow custom input
    default_ticker = selected_watchlist if selected_watchlist else "PLTR"
    ticker = st.text_input("🔍 Enter Stock Ticker", value=default_ticker, placeholder="e.g., AAPL, MSFT, PLTR", label_visibility="collapsed")
with col_fetch:
    fetch_clicked = st.button("Fetch Data", type="primary", use_container_width=True)

# --- Earnings Alert Banner (shows if any watchlist stocks have earnings this week) ---
@st.cache_data(ttl=3600)
def get_upcoming_week_earnings():
    """Fetch earnings happening in the next 7 days for watchlist stocks."""
    from datetime import datetime, timedelta
    today = datetime.now().date()
    week_from_now = today + timedelta(days=7)
    upcoming = []
    
    for stock_ticker in WATCHLIST_STOCKS[1:]:  # Skip empty string
        events = fetch_calendar_events(stock_ticker)
        for event_date, event_type, ticker in events:
            if event_type == 'Earnings' and today <= event_date <= week_from_now:
                upcoming.append((event_date, ticker))
    
    # Remove duplicates and sort
    upcoming = list(set(upcoming))
    upcoming.sort(key=lambda x: x[0])
    return upcoming

# Check for upcoming earnings and display banner
upcoming_earnings = get_upcoming_week_earnings()
if upcoming_earnings:
    earnings_text = " | ".join([f"**{ticker}** ({date.strftime('%b %d')})" for date, ticker in upcoming_earnings])
    st.info(f"📊 **Earnings This Week:** {earnings_text}")

# Auto-fetch when ticker changes, on first load, or when button clicked
if 'last_ticker' not in st.session_state:
    st.session_state.last_ticker = None

# Check if we need to fetch new data
should_fetch = fetch_clicked or 'stock_data' not in st.session_state
ticker_changed = ticker.upper() != st.session_state.last_ticker and ticker.strip() != ""

if should_fetch or ticker_changed:
    # Clear scenario settings when ticker changes so they reset to new stock's defaults
    for key in ['bear_growth', 'bear_margin', 'base_growth', 'base_margin', 'bull_growth', 'bull_margin']:
        if key in st.session_state:
            del st.session_state[key]
    
    with st.spinner(f"Fetching data for {ticker.upper()}..."):
        st.session_state.stock_data = fetch_stock_data(ticker.upper())
        st.session_state.ticker = ticker.upper()
        st.session_state.last_ticker = ticker.upper()
        st.rerun()  # Rerun to apply new defaults

# --- Sidebar with Calendar and News ---
with st.sidebar:
    st.markdown("### 📊 Stock Analysis Tool")
    st.caption("Select a stock to see events and news")
    st.markdown("---")

if 'stock_data' in st.session_state and st.session_state.stock_data.get('success'):
    news_data = st.session_state.stock_data.get('news', [])
    calendar_data = st.session_state.stock_data.get('calendar')
    earnings_dates = st.session_state.stock_data.get('earnings_dates')
    
    with st.sidebar:
        # Calendar / Upcoming Events Section
        st.header(f"📅 {st.session_state.ticker} Upcoming Events")
        
        has_events = False
        from datetime import datetime
        today = datetime.now().date()
        
        # Display calendar events (earnings, dividends, etc.) - only future dates
        if calendar_data is not None:
            if isinstance(calendar_data, dict):
                # Earnings Date
                if 'Earnings Date' in calendar_data:
                    earnings = calendar_data['Earnings Date']
                    if earnings:
                        # Filter to only future earnings dates
                        if isinstance(earnings, list):
                            future_earnings = []
                            for e in earnings:
                                try:
                                    e_date = e.date() if hasattr(e, 'date') else e
                                    if e_date >= today:
                                        future_earnings.append(e)
                                except:
                                    pass
                            if len(future_earnings) >= 2:
                                has_events = True
                                st.markdown(f"📊 **Earnings:** {future_earnings[0].strftime('%b %d, %Y')} - {future_earnings[1].strftime('%b %d, %Y')}")
                            elif len(future_earnings) == 1:
                                has_events = True
                                st.markdown(f"📊 **Earnings:** {future_earnings[0].strftime('%b %d, %Y')}")
                
                # Dividend Date - only show if in the future
                if 'Dividend Date' in calendar_data and calendar_data['Dividend Date']:
                    div_date = calendar_data['Dividend Date']
                    try:
                        d_date = div_date.date() if hasattr(div_date, 'date') else div_date
                        if d_date >= today:
                            has_events = True
                            st.markdown(f"💰 **Dividend Date:** {div_date.strftime('%b %d, %Y')}")
                    except:
                        pass
                
                # Ex-Dividend Date - only show if in the future
                if 'Ex-Dividend Date' in calendar_data and calendar_data['Ex-Dividend Date']:
                    ex_div = calendar_data['Ex-Dividend Date']
                    try:
                        ex_date = ex_div.date() if hasattr(ex_div, 'date') else ex_div
                        if ex_date >= today:
                            has_events = True
                            st.markdown(f"📆 **Ex-Dividend:** {ex_div.strftime('%b %d, %Y')}")
                    except:
                        pass
        
        # Display earnings dates table
        if earnings_dates is not None and not earnings_dates.empty:
            has_events = True
            st.markdown("**📈 Upcoming Earnings:**")
            for date_idx in earnings_dates.index[:4]:
                date_str = date_idx.strftime('%b %d, %Y')
                eps_est = earnings_dates.loc[date_idx, 'EPS Estimate'] if 'EPS Estimate' in earnings_dates.columns else None
                if eps_est and not pd.isna(eps_est):
                    st.caption(f"• {date_str} (Est. EPS: ${eps_est:.2f})")
                else:
                    st.caption(f"• {date_str}")
        
        if not has_events:
            st.info("No upcoming events found.")
        
        st.markdown("---")
        
        # News Section
        st.header(f"📰 {st.session_state.ticker} News")
        if news_data:
            for article in news_data[:10]:  # Show up to 10 articles
                content = article.get('content', {})
                title = content.get('title', article.get('title', 'No title'))
                link = article.get('link', content.get('canonicalUrl', {}).get('url', '#'))
                provider = content.get('provider', {}).get('displayName', '')
                pub_date = content.get('pubDate', article.get('providerPublishTime', ''))
                
                st.markdown(f"📄 **[{title}]({link})**")
                
                # Format and display date and provider
                date_str = ""
                if pub_date:
                    try:
                        from datetime import datetime
                        # Handle Unix timestamp
                        if isinstance(pub_date, (int, float)):
                            date_obj = datetime.fromtimestamp(pub_date)
                            date_str = date_obj.strftime('%b %d, %Y')
                        # Handle string date
                        elif isinstance(pub_date, str):
                            date_str = pub_date[:10] if len(pub_date) >= 10 else pub_date
                    except:
                        pass
                
                if provider and date_str:
                    st.caption(f"🔗 {provider} • 📅 {date_str}")
                elif provider:
                    st.caption(f"🔗 {provider}")
                elif date_str:
                    st.caption(f"📅 {date_str}")
                st.markdown("---")
        else:
            st.info("No recent news available.")

# --- Main Content ---
if 'stock_data' in st.session_state and st.session_state.stock_data.get('success'):
    data = st.session_state.stock_data
    
    st.markdown("---")
    
    # Company info header with action buttons
    header_col, financials_col, holders_col, analysts_col, insiders_col = st.columns([3, 1, 1, 1, 1])
    with header_col:
        st.header(f"{data['company_name']} ({st.session_state.ticker})")
    with financials_col:
        show_financials = st.button("📊 Financials", key="show_financials", use_container_width=True, help="View detailed financial statements")
    with holders_col:
        show_holders = st.button("🏛️ Holders", key="show_holders", use_container_width=True, help="View institutional and major holders")
    with analysts_col:
        show_analysts = st.button("🎯 Analysts", key="show_analysts", use_container_width=True, help="View analyst price targets and recommendations")
    with insiders_col:
        show_insiders = st.button("👤 Insiders", key="show_insiders", use_container_width=True, help="View insider trading activity")
    
    # Financials dialog/modal
    if show_financials:
        st.session_state.show_financials_dialog = True
    
    if st.session_state.get('show_financials_dialog', False):
        with st.expander(f"📊 {data['company_name']} Financial Statements", expanded=True):
            close_col1, close_col2 = st.columns([6, 1])
            with close_col2:
                if st.button("✕ Close", key="close_financials"):
                    st.session_state.show_financials_dialog = False
                    st.rerun()
            
            fin_tab1, fin_tab2, fin_tab3 = st.tabs(["📈 Income Statement", "📋 Balance Sheet", "💵 Cash Flow"])
            
            with fin_tab1:
                income_stmt = data.get('income_stmt')
                if income_stmt is not None and not income_stmt.empty:
                    # Format large numbers for display
                    formatted_income = income_stmt.copy()
                    for col in formatted_income.columns:
                        formatted_income[col] = formatted_income[col].apply(
                            lambda x: f"${x/1e9:.2f}B" if pd.notna(x) and abs(x) >= 1e9 else 
                                     (f"${x/1e6:.2f}M" if pd.notna(x) and abs(x) >= 1e6 else 
                                     (f"${x:,.0f}" if pd.notna(x) else "—"))
                        )
                    # Rename columns to years
                    formatted_income.columns = [col.strftime('%Y') if hasattr(col, 'strftime') else str(col) for col in formatted_income.columns]
                    st.dataframe(formatted_income, use_container_width=True, height=400)
                else:
                    st.info("Income statement data not available")
            
            with fin_tab2:
                balance_sheet = data.get('balance_sheet')
                if balance_sheet is not None and not balance_sheet.empty:
                    formatted_bs = balance_sheet.copy()
                    for col in formatted_bs.columns:
                        formatted_bs[col] = formatted_bs[col].apply(
                            lambda x: f"${x/1e9:.2f}B" if pd.notna(x) and abs(x) >= 1e9 else 
                                     (f"${x/1e6:.2f}M" if pd.notna(x) and abs(x) >= 1e6 else 
                                     (f"${x:,.0f}" if pd.notna(x) else "—"))
                        )
                    formatted_bs.columns = [col.strftime('%Y') if hasattr(col, 'strftime') else str(col) for col in formatted_bs.columns]
                    st.dataframe(formatted_bs, use_container_width=True, height=400)
                else:
                    st.info("Balance sheet data not available")
            
            with fin_tab3:
                cash_flow = data.get('cash_flow')
                if cash_flow is not None and not cash_flow.empty:
                    formatted_cf = cash_flow.copy()
                    for col in formatted_cf.columns:
                        formatted_cf[col] = formatted_cf[col].apply(
                            lambda x: f"${x/1e9:.2f}B" if pd.notna(x) and abs(x) >= 1e9 else 
                                     (f"${x/1e6:.2f}M" if pd.notna(x) and abs(x) >= 1e6 else 
                                     (f"${x:,.0f}" if pd.notna(x) else "—"))
                        )
                    formatted_cf.columns = [col.strftime('%Y') if hasattr(col, 'strftime') else str(col) for col in formatted_cf.columns]
                    st.dataframe(formatted_cf, use_container_width=True, height=400)
                else:
                    st.info("Cash flow data not available")
    
    # Holders dialog/modal
    if show_holders:
        st.session_state.show_holders_dialog = True
    
    if st.session_state.get('show_holders_dialog', False):
        with st.expander(f"🏛️ {data['company_name']} Shareholders", expanded=True):
            close_col1, close_col2 = st.columns([6, 1])
            with close_col2:
                if st.button("✕ Close", key="close_holders"):
                    st.session_state.show_holders_dialog = False
                    st.rerun()
            
            holders_tab1, holders_tab2, holders_tab3 = st.tabs(["🏛️ Institutional Holders", "📊 Major Holders", "💼 Mutual Fund Holders"])
            
            with holders_tab1:
                institutional_holders = data.get('institutional_holders')
                if institutional_holders is not None and not institutional_holders.empty:
                    # Format the dataframe for display
                    formatted_inst = institutional_holders.copy()
                    
                    # Format shares and value columns
                    if 'Shares' in formatted_inst.columns:
                        formatted_inst['Shares'] = formatted_inst['Shares'].apply(
                            lambda x: f"{x/1e6:.2f}M" if pd.notna(x) and x >= 1e6 else (f"{x:,.0f}" if pd.notna(x) else "—")
                        )
                    if 'Value' in formatted_inst.columns:
                        formatted_inst['Value'] = formatted_inst['Value'].apply(
                            lambda x: f"${x/1e9:.2f}B" if pd.notna(x) and x >= 1e9 else 
                                     (f"${x/1e6:.2f}M" if pd.notna(x) and x >= 1e6 else 
                                     (f"${x:,.0f}" if pd.notna(x) else "—"))
                        )
                    if '% Out' in formatted_inst.columns:
                        formatted_inst['% Out'] = formatted_inst['% Out'].apply(
                            lambda x: f"{x*100:.2f}%" if pd.notna(x) else "—"
                        )
                    if 'pctHeld' in formatted_inst.columns:
                        formatted_inst['pctHeld'] = formatted_inst['pctHeld'].apply(
                            lambda x: f"{x*100:.2f}%" if pd.notna(x) else "—"
                        )
                    
                    st.dataframe(formatted_inst, use_container_width=True, height=400)
                else:
                    st.info("Institutional holders data not available")
            
            with holders_tab2:
                major_holders = data.get('major_holders')
                if major_holders is not None and not major_holders.empty:
                    st.dataframe(major_holders, use_container_width=True)
                else:
                    st.info("Major holders data not available")
            
            with holders_tab3:
                mutualfund_holders = data.get('mutualfund_holders')
                if mutualfund_holders is not None and not mutualfund_holders.empty:
                    # Format the dataframe for display
                    formatted_mf = mutualfund_holders.copy()
                    
                    # Format shares and value columns
                    if 'Shares' in formatted_mf.columns:
                        formatted_mf['Shares'] = formatted_mf['Shares'].apply(
                            lambda x: f"{x/1e6:.2f}M" if pd.notna(x) and x >= 1e6 else (f"{x:,.0f}" if pd.notna(x) else "—")
                        )
                    if 'Value' in formatted_mf.columns:
                        formatted_mf['Value'] = formatted_mf['Value'].apply(
                            lambda x: f"${x/1e9:.2f}B" if pd.notna(x) and x >= 1e9 else 
                                     (f"${x/1e6:.2f}M" if pd.notna(x) and x >= 1e6 else 
                                     (f"${x:,.0f}" if pd.notna(x) else "—"))
                        )
                    if '% Out' in formatted_mf.columns:
                        formatted_mf['% Out'] = formatted_mf['% Out'].apply(
                            lambda x: f"{x*100:.2f}%" if pd.notna(x) else "—"
                        )
                    if 'pctHeld' in formatted_mf.columns:
                        formatted_mf['pctHeld'] = formatted_mf['pctHeld'].apply(
                            lambda x: f"{x*100:.2f}%" if pd.notna(x) else "—"
                        )
                    
                    st.dataframe(formatted_mf, use_container_width=True, height=400)
                else:
                    st.info("Mutual fund holders data not available")
    
    # Analysts dialog
    if show_analysts:
        st.session_state.show_analysts_dialog = True
    
    if st.session_state.get('show_analysts_dialog', False):
        with st.expander(f"🎯 {data['company_name']} Analyst Coverage", expanded=True):
            close_col1, close_col2 = st.columns([6, 1])
            with close_col2:
                if st.button("✕ Close", key="close_analysts"):
                    st.session_state.show_analysts_dialog = False
                    st.rerun()
            
            # Price Targets Summary
            st.markdown("### 📊 Price Targets")
            
            target_low = data.get('target_low')
            target_mean = data.get('target_mean')
            target_high = data.get('target_high')
            target_median = data.get('target_median')
            current_price = data.get('current_price')
            num_analysts = data.get('num_analysts')
            
            if target_mean or target_low or target_high:
                # Calculate upside/downside
                if current_price and target_mean:
                    upside = ((target_mean / current_price) - 1) * 100
                    upside_str = f"+{upside:.1f}%" if upside > 0 else f"{upside:.1f}%"
                else:
                    upside_str = "N/A"
                
                target_cols = st.columns(5)
                with target_cols[0]:
                    st.metric("Current Price", f"${current_price:.2f}" if current_price else "N/A")
                with target_cols[1]:
                    st.metric("Target Low", f"${target_low:.2f}" if target_low else "N/A")
                with target_cols[2]:
                    st.metric("Target Mean", f"${target_mean:.2f}" if target_mean else "N/A", delta=upside_str if target_mean else None)
                with target_cols[3]:
                    st.metric("Target High", f"${target_high:.2f}" if target_high else "N/A")
                with target_cols[4]:
                    st.metric("# Analysts", str(num_analysts) if num_analysts else "N/A")
                
                # Visual price target range
                if target_low and target_high and current_price:
                    st.markdown("#### Price Target Range")
                    range_width = target_high - target_low
                    if range_width > 0:
                        current_pos = ((current_price - target_low) / range_width) * 100
                        current_pos = max(0, min(100, current_pos))  # Clamp to 0-100
                        
                        st.markdown(f"""
                        <div style="background: linear-gradient(to right, #ff6b6b 0%, #ffd93d 50%, #6bcb77 100%); 
                                    height: 20px; border-radius: 10px; position: relative; margin: 10px 0;">
                            <div style="position: absolute; left: {current_pos}%; top: -5px; 
                                        width: 4px; height: 30px; background: white; border-radius: 2px;"></div>
                        </div>
                        <div style="display: flex; justify-content: space-between; font-size: 12px;">
                            <span>${target_low:.2f}</span>
                            <span style="font-weight: bold;">Current: ${current_price:.2f}</span>
                            <span>${target_high:.2f}</span>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("No analyst price targets available for this stock.")
            
            st.markdown("---")
            
            # Recommendation Summary
            st.markdown("### 📈 Analyst Recommendation")
            recommendation = data.get('recommendation')
            recommendation_mean = data.get('recommendation_mean')
            
            if recommendation:
                # Map recommendation to color and emoji
                rec_map = {
                    'strongBuy': ('🟢', 'Strong Buy', '#22c55e'),
                    'buy': ('🟢', 'Buy', '#84cc16'),
                    'hold': ('🟡', 'Hold', '#eab308'),
                    'sell': ('🔴', 'Sell', '#f97316'),
                    'strongSell': ('🔴', 'Strong Sell', '#ef4444'),
                }
                emoji, label, color = rec_map.get(recommendation, ('⚪', recommendation.title(), '#6b7280'))
                
                rec_cols = st.columns([1, 2])
                with rec_cols[0]:
                    st.markdown(f"<h2 style='color: {color};'>{emoji} {label}</h2>", unsafe_allow_html=True)
                with rec_cols[1]:
                    if recommendation_mean:
                        st.caption(f"Mean Score: {recommendation_mean:.2f} (1=Strong Buy, 5=Strong Sell)")
            else:
                st.info("No analyst recommendation available.")
            
            st.markdown("---")
            
            # Recommendations History
            st.markdown("### 📅 Recommendations History")
            recommendations = data.get('recommendations')
            if recommendations is not None and not recommendations.empty:
                # Get recent recommendations (last 10)
                recent_recs = recommendations.tail(10).copy()
                recent_recs = recent_recs.sort_index(ascending=False)
                
                # Format the index (dates)
                if hasattr(recent_recs.index, 'strftime'):
                    recent_recs.index = recent_recs.index.strftime('%Y-%m-%d')
                
                st.dataframe(recent_recs, use_container_width=True, height=300)
            else:
                st.info("No recommendations history available.")
    
    # Insiders dialog
    if show_insiders:
        st.session_state.show_insiders_dialog = True
    
    if st.session_state.get('show_insiders_dialog', False):
        with st.expander(f"👤 {data['company_name']} Insider Activity", expanded=True):
            close_col1, close_col2 = st.columns([6, 1])
            with close_col2:
                if st.button("✕ Close", key="close_insiders"):
                    st.session_state.show_insiders_dialog = False
                    st.rerun()
            
            insider_tab1, insider_tab2 = st.tabs(["📋 Recent Transactions", "📊 Purchase Summary"])
            
            with insider_tab1:
                insider_transactions = data.get('insider_transactions')
                if insider_transactions is not None and not insider_transactions.empty:
                    # Toggle to filter only buy transactions
                    show_buys_only = st.toggle("🟢 Show Buy Transactions Only", value=False, key="insider_buys_only")
                    
                    # Filter if toggle is on
                    if show_buys_only and 'Text' in insider_transactions.columns:
                        filtered_trans = insider_transactions[
                            insider_transactions['Text'].str.contains('Buy|Purchase|Acquisition', case=False, na=False)
                        ].copy()
                    else:
                        filtered_trans = insider_transactions.copy()
                    
                    # Format the dataframe
                    formatted_trans = filtered_trans.copy()
                    
                    # Format value/shares columns if they exist
                    if 'Value' in formatted_trans.columns:
                        formatted_trans['Value'] = formatted_trans['Value'].apply(
                            lambda x: f"${x/1e6:.2f}M" if pd.notna(x) and abs(x) >= 1e6 else 
                                     (f"${x:,.0f}" if pd.notna(x) else "—")
                        )
                    if 'Shares' in formatted_trans.columns:
                        formatted_trans['Shares'] = formatted_trans['Shares'].apply(
                            lambda x: f"{x:,.0f}" if pd.notna(x) else "—"
                        )
                    
                    # Show filtered count if filter is active
                    if show_buys_only:
                        st.caption(f"Showing {len(formatted_trans)} buy transaction(s)")
                    
                    # Color code buys vs sells
                    st.dataframe(formatted_trans, use_container_width=True, height=400)
                    
                    # Summary stats
                    st.markdown("---")
                    st.markdown("#### Transaction Summary")
                    
                    # Try to count buys vs sells
                    if 'Text' in insider_transactions.columns:
                        buys = insider_transactions['Text'].str.contains('Buy|Purchase|Acquisition', case=False, na=False).sum()
                        sells = insider_transactions['Text'].str.contains('Sale|Sell|Disposition', case=False, na=False).sum()
                        
                        sum_cols = st.columns(3)
                        with sum_cols[0]:
                            st.metric("🟢 Buy Transactions", buys)
                        with sum_cols[1]:
                            st.metric("🔴 Sell Transactions", sells)
                        with sum_cols[2]:
                            net = buys - sells
                            st.metric("Net Activity", f"+{net}" if net > 0 else str(net), 
                                     delta="Bullish" if net > 0 else ("Bearish" if net < 0 else "Neutral"))
                else:
                    st.info("No insider transaction data available.")
            
            with insider_tab2:
                insider_purchases = data.get('insider_purchases')
                if insider_purchases is not None and not insider_purchases.empty:
                    st.dataframe(insider_purchases, use_container_width=True, height=300)
                else:
                    st.info("No insider purchase summary available.")
    
    # SEC Filings Section
    if 'show_sec_filings' not in st.session_state:
        st.session_state.show_sec_filings = False
    
    sec_col1, sec_col2 = st.columns([1, 5])
    with sec_col1:
        if st.button("📄 SEC Filings", help="View recent SEC filings (10-K, 10-Q, 8-K)"):
            st.session_state.show_sec_filings = not st.session_state.show_sec_filings
    
    if st.session_state.show_sec_filings:
        with st.expander(f"📄 {data['company_name']} SEC Filings", expanded=True):
            close_col1, close_col2 = st.columns([6, 1])
            with close_col2:
                if st.button("✕ Close", key="close_sec"):
                    st.session_state.show_sec_filings = False
                    st.rerun()
            
            # Fetch SEC filings
            with st.spinner("Fetching SEC filings..."):
                filings = fetch_sec_filings(st.session_state.ticker, limit=15)
            
            if filings:
                st.markdown("**Recent Financial Filings** — Click to view on SEC.gov")
                
                # Create a clean display
                for filing in filings:
                    form_type = filing['form']
                    filing_date = filing['filingDate']
                    url = filing['url']
                    
                    # Color code by form type
                    if '10-K' in form_type:
                        icon = "📊"
                        desc = "Annual Report"
                    elif '10-Q' in form_type:
                        icon = "📈"
                        desc = "Quarterly Report"
                    elif '8-K' in form_type:
                        icon = "📰"
                        desc = "Current Report"
                    else:
                        icon = "📄"
                        desc = "Filing"
                    
                    # Display as clickable link
                    st.markdown(f"{icon} **[{form_type}]({url})** — {desc} — Filed: {filing_date}")
                
                st.markdown("---")
                st.caption("💡 **10-K**: Annual report with audited financials | **10-Q**: Quarterly report | **8-K**: Material events")
            else:
                st.info("No SEC filings found. This may be a non-US company or the ticker may not be registered with the SEC.")
    
    # Calculate Magic Formula metrics
    earnings_yield = None
    return_on_capital = None
    ebit = data.get('ebit')
    enterprise_value = data.get('enterprise_value')
    total_assets = data.get('total_assets')
    current_liabilities = data.get('current_liabilities')
    
    # Fallback: Calculate Enterprise Value if not provided (EV = Market Cap + Debt - Cash)
    if enterprise_value is None:
        market_cap = data.get('market_cap')
        total_debt = data.get('total_debt', 0) or 0
        total_cash = data.get('total_cash', 0) or 0
        if market_cap:
            enterprise_value = market_cap + total_debt - total_cash
    
    # Get cash from balance sheet (more accurate than info.totalCash which includes short-term investments)
    balance_sheet = data.get('balance_sheet')
    cash_for_roc = 0
    if balance_sheet is not None and not balance_sheet.empty:
        try:
            latest_col = balance_sheet.columns[0]
            if 'Cash And Cash Equivalents' in balance_sheet.index:
                cash_for_roc = balance_sheet.loc['Cash And Cash Equivalents', latest_col] or 0
        except:
            pass
    if cash_for_roc == 0:
        cash_for_roc = data.get('total_cash', 0) or 0
    
    # Earnings Yield = EBIT / Enterprise Value
    if ebit and enterprise_value and enterprise_value > 0:
        earnings_yield = (ebit / enterprise_value) * 100
    
    # Return on Capital = EBIT / (Total Assets - Current Liabilities - Cash)
    # Uses Greenblatt's definition of invested capital
    if ebit and total_assets and current_liabilities is not None:
        invested_capital = total_assets - (current_liabilities or 0) - cash_for_roc
        if invested_capital > 0:
            return_on_capital = (ebit / invested_capital) * 100
    
    # Current metrics
    m1, m2, m3, m4, m5, m6, m7, m8 = st.columns(8)
    with m1:
        st.metric("Current Price", format_currency(data['current_price']), help="The latest trading price of the stock. This is the baseline for calculating future returns.")
    with m2:
        st.metric("Current EPS", format_currency(data['current_eps']), help="Earnings Per Share (TTM). Net income divided by shares outstanding. Higher EPS indicates more profit per share.")
    with m3:
        st.metric("Revenue", format_large_number(data['total_revenue']), help="Total revenue from the most recent fiscal year. This is the top-line income before expenses.")
    with m4:
        st.metric("Net Income", format_large_number(data['net_income']), help="Net income (profit) after all expenses, taxes, and costs. This is the bottom-line earnings.")
    with m5:
        st.metric("Current Margin", format_percent(data['current_margin']), help="Net Income Margin = Net Income / Revenue. Shows what percentage of revenue becomes profit. Higher is better.")
    with m6:
        # Calculate insider buys in last quarter and last year
        insider_transactions = data.get('insider_transactions')
        buys_quarter = 0
        buys_year = 0
        if insider_transactions is not None and not insider_transactions.empty:
            from datetime import datetime, timedelta
            today = datetime.now()
            quarter_ago = today - timedelta(days=90)
            year_ago = today - timedelta(days=365)
            
            # Check if there's a date column and Text column
            if 'Text' in insider_transactions.columns:
                # Filter for buy transactions
                buy_mask = insider_transactions['Text'].str.contains('Buy|Purchase|Acquisition', case=False, na=False)
                buy_trans = insider_transactions[buy_mask]
                
                # Try to find date column
                date_col = None
                for col in ['Start Date', 'Date', 'Filing Date', 'startDate']:
                    if col in buy_trans.columns:
                        date_col = col
                        break
                
                if date_col:
                    try:
                        buy_trans_dated = buy_trans.copy()
                        buy_trans_dated[date_col] = pd.to_datetime(buy_trans_dated[date_col], errors='coerce')
                        buys_quarter = len(buy_trans_dated[buy_trans_dated[date_col] >= quarter_ago])
                        buys_year = len(buy_trans_dated[buy_trans_dated[date_col] >= year_ago])
                    except:
                        # If date parsing fails, just count total buys
                        buys_year = len(buy_trans)
                        buys_quarter = min(buys_year, len(buy_trans) // 4) if buys_year > 0 else 0
                else:
                    # No date column, estimate from total
                    buys_year = len(buy_trans)
                    buys_quarter = min(buys_year, len(buy_trans) // 4) if buys_year > 0 else 0
        
        # Display metric
        if buys_quarter > 0:
            delta_text = f"{buys_year} in last year"
            delta_color = "normal"
        elif buys_year > 0:
            delta_text = f"{buys_year} in last year"
            delta_color = "off"
        else:
            delta_text = "None in last year"
            delta_color = "off"
        
        st.metric(
            "Insider Buys (90d)",
            str(buys_quarter),
            delta=delta_text,
            delta_color=delta_color,
            help="Number of insider buy transactions in the last 90 days. Sub-metric shows buys in the last year. More insider buying can be a bullish signal."
        )
    with m7:
        st.metric(
            "Earnings Yield",
            f"{earnings_yield:.1f}%" if earnings_yield is not None else "N/A",
            help="Magic Formula metric: EBIT / Enterprise Value. Measures how cheap the stock is relative to its operating earnings. Higher is better (cheaper). Part of Joel Greenblatt's Magic Formula."
        )
    with m8:
        st.metric(
            "Return on Capital",
            f"{return_on_capital:.1f}%" if return_on_capital is not None else "N/A",
            help="Magic Formula metric: EBIT / Invested Capital (Total Assets - Current Liabilities - Cash). Measures how efficiently the company generates profits from its capital. Higher is better. Part of Joel Greenblatt's Magic Formula."
        )
    
    # ============================================
    # VALUATION SUMMARY SECTION (above DCF)
    # ============================================
    st.markdown("---")
    st.subheader("📊 Valuation Summary")
    
    # Pre-calculate all valuations for the summary
    # Fixed parameters matching reference model
    discount_rate_summary = 0.08  # Fixed 8% discount rate
    projection_years_summary = 10  # 10 year projection
    base_growth_summary = 0.15  # Fixed 15% FCF growth rate (matching reference)
    
    # 1. DCF Value (using 15% growth rate)
    dcf_summary = calculate_dcf(
        free_cash_flow=data.get('free_cash_flow'),
        fcf_growth_rate=base_growth_summary,
        discount_rate=discount_rate_summary,
        terminal_growth_rate=0.03,
        projection_years=projection_years_summary,
        total_debt=data.get('total_debt', 0),
        total_cash=data.get('total_cash', 0),
        shares_outstanding=data.get('shares_outstanding', 1)
    )
    dcf_value = dcf_summary.get('intrinsic_value')
    
    # 2. DDM Value (handles non-dividend stocks gracefully)
    # Use 6% dividend growth for established dividend payers, 2% spread
    ddm_result = calculate_ddm(
        dividend_rate=data.get('dividend_rate'),
        current_price=data.get('current_price'),
        dividend_growth_rate=0.06,  # 6% dividend growth
        required_return=discount_rate_summary  # 8% required return (2% spread)
    )
    ddm_value = ddm_result.get('intrinsic_value')
    
    # 3. Multiples Value - use sector average P/E for fair value comparison
    eps = data.get('current_eps')
    multiples_result = calculate_multiples_value(
        eps=eps,
        current_price=data.get('current_price'),
        sector=data.get('sector', 'Unknown')
    )
    multiples_value = multiples_result.get('intrinsic_value')
    
    # 4. Reverse DCF
    reverse_dcf_result = calculate_reverse_dcf(
        current_price=data.get('current_price'),
        shares_outstanding=data.get('shares_outstanding'),
        free_cash_flow=data.get('free_cash_flow'),
        discount_rate=discount_rate_summary,
        terminal_growth_rate=0.03,
        projection_years=projection_years_summary,
        total_debt=data.get('total_debt', 0),
        total_cash=data.get('total_cash', 0)
    )
    implied_growth = reverse_dcf_result.get('implied_growth_pct')
    
    # 5. Combined Intrinsic Value
    combined_result = calculate_combined_intrinsic(dcf_value, ddm_value, multiples_value)
    combined_value = combined_result.get('combined_value')
    
    current_price = data.get('current_price', 0)
    
    # Calculate overall margin of safety
    if combined_value and current_price and current_price > 0:
        overall_margin = ((combined_value - current_price) / current_price) * 100
        overall_upside = ((combined_value / current_price) - 1) * 100
    else:
        overall_margin = None
        overall_upside = None
    
    # Display Combined Intrinsic Value prominently
    if combined_value:
        summary_cols = st.columns([2, 1, 1, 1])
        
        with summary_cols[0]:
            # Large combined value display
            if overall_upside is not None:
                if overall_upside > 0:
                    delta_text = f"+{overall_upside:.1f}% upside"
                    delta_color = "normal"
                else:
                    delta_text = f"{overall_upside:.1f}% downside"
                    delta_color = "inverse"
            else:
                delta_text = None
                delta_color = "off"
            
            st.metric(
                "🎯 Combined Intrinsic Value",
                f"${combined_value:.2f}",
                delta=delta_text,
                delta_color=delta_color,
                help=f"Average of {combined_result.get('count', 0)} valuation methods: {', '.join(combined_result.get('methods_used', []))}"
            )
        
        with summary_cols[1]:
            st.metric(
                "Current Price",
                f"${current_price:.2f}" if current_price else "N/A"
            )
        
        with summary_cols[2]:
            # Margin of Safety
            if overall_margin is not None:
                mos_display = f"{overall_margin:.1f}%"
                if overall_margin > 15:
                    mos_status = "✅ Buy Zone"
                elif overall_margin > 0:
                    mos_status = "🟡 Fair"
                else:
                    mos_status = "🔴 Overvalued"
            else:
                mos_display = "N/A"
                mos_status = None
            
            st.metric(
                "Margin of Safety",
                mos_display,
                delta=mos_status,
                delta_color="off",
                help="Positive = undervalued. >15% typically considered a buy opportunity."
            )
        
        with summary_cols[3]:
            # Wall Street Target comparison
            target_mean = data.get('target_mean')
            if target_mean and current_price:
                ws_upside = ((target_mean / current_price) - 1) * 100
                st.metric(
                    "Wall St. Target",
                    f"${target_mean:.2f}",
                    delta=f"{ws_upside:+.1f}%",
                    delta_color="normal" if ws_upside > 0 else "inverse",
                    help="Average analyst price target"
                )
            else:
                st.metric("Wall St. Target", "N/A")
        
        # Individual valuation methods row
        st.markdown("##### Individual Valuations")
        val_cols = st.columns(4)
        
        with val_cols[0]:
            if dcf_value:
                dcf_upside = ((dcf_value / current_price) - 1) * 100 if current_price else 0
                st.metric(
                    "DCF Value",
                    f"${dcf_value:.2f}",
                    delta=f"{dcf_upside:+.1f}%",
                    delta_color="normal" if dcf_upside > 0 else "inverse"
                )
            else:
                st.metric("DCF Value", "N/A", help="Requires positive free cash flow")
        
        with val_cols[1]:
            if ddm_value:
                ddm_upside = ((ddm_value / current_price) - 1) * 100 if current_price else 0
                st.metric(
                    "DDM Value",
                    f"${ddm_value:.2f}",
                    delta=f"{ddm_upside:+.1f}%",
                    delta_color="normal" if ddm_upside > 0 else "inverse"
                )
            else:
                st.metric("DDM Value", "—", help="Stock does not pay dividends")
        
        with val_cols[2]:
            if multiples_value:
                mult_upside = ((multiples_value / current_price) - 1) * 100 if current_price else 0
                sector_pe = multiples_result.get('sector_pe', 'N/A')
                st.metric(
                    "Multiples Value",
                    f"${multiples_value:.2f}",
                    delta=f"{mult_upside:+.1f}%",
                    delta_color="normal" if mult_upside > 0 else "inverse",
                    help=f"Based on sector avg P/E of {sector_pe}x"
                )
            else:
                st.metric("Multiples Value", "N/A", help="Requires positive EPS")
        
        with val_cols[3]:
            if implied_growth is not None:
                # Interpret implied growth
                if implied_growth > 30:
                    growth_status = "🔴 Very High"
                elif implied_growth > 20:
                    growth_status = "🟠 High"
                elif implied_growth > 10:
                    growth_status = "🟢 Moderate"
                else:
                    growth_status = "🟢 Low"
                
                st.metric(
                    "Reverse DCF",
                    f"{implied_growth:.1f}%",
                    delta=growth_status,
                    delta_color="off",
                    help="Implied FCF growth rate priced into current stock price. Lower = more margin of safety."
                )
            else:
                st.metric("Reverse DCF", "N/A", help="Requires positive free cash flow")
        
        # Sensitivity Analysis Table
        st.markdown("##### DCF Sensitivity Analysis")
        sensitivity = calculate_dcf_sensitivity(
            free_cash_flow=data.get('free_cash_flow'),
            base_growth_rate=base_growth_summary,
            discount_rate=discount_rate_summary,
            terminal_growth_rate=0.03,
            projection_years=projection_years_summary,
            total_debt=data.get('total_debt', 0),
            total_cash=data.get('total_cash', 0),
            shares_outstanding=data.get('shares_outstanding', 1),
            current_price=current_price
        )
        
        if sensitivity:
            # Build table using pandas DataFrame for reliable rendering
            import pandas as pd
            
            sens_data = []
            for scenario in sensitivity:
                sens_data.append({
                    'Scenario': scenario['label'],
                    'Growth Rate': f"{scenario['growth_rate']:.0f}%",
                    'Value': f"${scenario['value']:.0f}",
                    'Upside': f"{scenario['upside']:+.0f}%"
                })
            
            sens_df = pd.DataFrame(sens_data)
            sens_df = sens_df.set_index('Scenario')
            
            # Use st.dataframe with custom styling - muted, darker colors
            def color_scenario(val):
                if val.name == 'Low':
                    return ['background-color: rgba(34, 120, 74, 0.6); color: white; font-weight: bold'] * len(val)
                elif val.name == 'Medium':
                    return ['background-color: rgba(180, 130, 20, 0.6); color: white; font-weight: bold'] * len(val)
                elif val.name == 'High':
                    return ['background-color: rgba(180, 60, 60, 0.6); color: white; font-weight: bold'] * len(val)
                return [''] * len(val)
            
            styled_df = sens_df.style.apply(color_scenario, axis=1)
            st.dataframe(styled_df, use_container_width=True)
        
        # Buy/Sell Signal
        if overall_margin is not None:
            if overall_margin > 25:
                st.success("🟢 **Strong Buy Signal**: Stock appears significantly undervalued across multiple valuation methods.")
            elif overall_margin > 15:
                st.info("🟢 **Buy Signal**: Stock appears undervalued with adequate margin of safety.")
            elif overall_margin > -10:
                st.warning("🟡 **Hold**: Stock appears fairly valued. Consider waiting for a better entry point.")
            else:
                st.error("🔴 **Overvalued**: Stock trades above intrinsic value estimates. Exercise caution.")
    else:
        st.info("📊 Insufficient data to calculate combined intrinsic value. Individual metrics shown below.")
    
    # DCF Intrinsic Value Section
    st.markdown("---")
    st.subheader("💰 DCF Intrinsic Value")
    
    # Fixed parameters matching reference model
    discount_rate = 0.08  # Fixed 8% discount rate
    projection_years = 10  # 10 year projection
    fcf_growth_rate = 0.15  # Fixed 15% FCF growth rate (matching reference)
    
    # Calculate DCF
    dcf_result = calculate_dcf(
        free_cash_flow=data.get('free_cash_flow'),
        fcf_growth_rate=fcf_growth_rate,
        discount_rate=discount_rate,
        terminal_growth_rate=0.03,  # 3% terminal growth
        projection_years=projection_years,
        total_debt=data.get('total_debt', 0),
        total_cash=data.get('total_cash', 0),
        shares_outstanding=data.get('shares_outstanding', 1)
    )
    
    if dcf_result.get('intrinsic_value') is not None:
        intrinsic_value = dcf_result['intrinsic_value']
        current_price = data['current_price']
        
        # Calculate upside/downside and margin of safety
        if current_price and current_price > 0:
            upside_pct = ((intrinsic_value / current_price) - 1) * 100
            margin_of_safety = ((intrinsic_value - current_price) / intrinsic_value) * 100 if intrinsic_value > 0 else 0
        else:
            upside_pct = 0
            margin_of_safety = 0
        
        dcf_cols = st.columns(4)
        with dcf_cols[0]:
            # Show warning icon if FCF is negative
            label = "⚠️ DCF Fair Value" if dcf_result.get('fcf_negative') else "DCF Fair Value"
            st.metric(
                label,
                f"${intrinsic_value:.2f}" if intrinsic_value > 0 else f"${intrinsic_value:.2f}",
                delta=f"{upside_pct:+.1f}% vs current",
                # Use "normal" so positive upside = green, negative upside = red
                delta_color="normal",
                help="Intrinsic value per share based on Discounted Cash Flow analysis. Compares projected future cash flows discounted to present value."
            )
        with dcf_cols[1]:
            # Margin of safety - positive means stock is undervalued
            if margin_of_safety > 15:
                safety_status = "Undervalued"
                safety_color = "normal"
            elif margin_of_safety > -15:
                safety_status = "Fair Value"
                safety_color = "off"
            else:
                safety_status = "Overvalued"
                safety_color = "inverse"
            st.metric(
                "Margin of Safety",
                f"{margin_of_safety:.1f}%",
                delta=safety_status,
                delta_color=safety_color,
                help="Margin of Safety = (Intrinsic Value - Price) / Intrinsic Value. Positive means stock trades below fair value. >15% is typically considered a good margin."
            )
        with dcf_cols[2]:
            st.metric(
                "Discount Rate",
                f"{discount_rate*100:.1f}%",
                help="Fixed 8% discount rate used to discount future cash flows to present value."
            )
        with dcf_cols[3]:
            fcf = data.get('free_cash_flow')
            fcf_display = format_large_number(fcf) if fcf else "N/A"
            st.metric(
                "Free Cash Flow",
                fcf_display,
                help="Current annual Free Cash Flow (Operating Cash Flow - CapEx). This is the cash available to shareholders after all expenses and investments."
            )
        
        # Warning for negative FCF
        if dcf_result.get('fcf_negative'):
            st.warning("⚠️ **Negative Free Cash Flow**: This company is currently burning cash. DCF valuation shows what the stock would be worth if cash burn continues at the projected growth rate. Use with caution.")
        
        # Expandable DCF details
        with st.expander("📊 DCF Calculation Details"):
            detail_cols = st.columns(3)
            with detail_cols[0]:
                st.markdown("**Assumptions**")
                st.write(f"- FCF Growth Rate: {fcf_growth_rate*100:.1f}% (Dynamic)")
                st.write(f"- Terminal Growth: 3.0%")
                st.write(f"- Discount Rate: {discount_rate*100:.1f}%")
                st.write(f"- Projection Years: {projection_years}")
            with detail_cols[1]:
                st.markdown("**Valuation Components**")
                st.write(f"- PV of FCFs: {format_large_number(dcf_result.get('pv_fcfs', 0))}")
                st.write(f"- PV of Terminal: {format_large_number(dcf_result.get('pv_terminal', 0))}")
                st.write(f"- Enterprise Value: {format_large_number(dcf_result.get('enterprise_value', 0))}")
            with detail_cols[2]:
                st.markdown("**Equity Calculation**")
                st.write(f"- Total Debt: {format_large_number(data.get('total_debt', 0))}")
                st.write(f"- Total Cash: {format_large_number(data.get('total_cash', 0))}")
                st.write(f"- Equity Value: {format_large_number(dcf_result.get('equity_value', 0))}")
            
            # DCF Sensitivity Analysis
            st.markdown("---")
            st.markdown("**📈 Sensitivity Analysis**")
            sensitivity = calculate_dcf_sensitivity(
                free_cash_flow=data.get('free_cash_flow'),
                base_growth_rate=fcf_growth_rate,
                discount_rate=discount_rate,
                terminal_growth_rate=0.03,
                projection_years=projection_years,
                total_debt=data.get('total_debt', 0),
                total_cash=data.get('total_cash', 0),
                shares_outstanding=data.get('shares_outstanding', 1),
                current_price=data.get('current_price')
            )
            
            if sensitivity:
                # Create table matching reference screenshot format
                st.markdown("""
                <style>
                .sens-table { width: 100%; border-collapse: collapse; margin-top: 10px; }
                .sens-table th, .sens-table td { padding: 8px 12px; text-align: center; border: 1px solid #ddd; }
                .sens-table th { background-color: #f8f9fa; font-weight: bold; }
                .sens-row-low { background-color: #22c55e; color: white; font-weight: bold; }
                .sens-row-med { background-color: #f59e0b; color: white; font-weight: bold; }
                .sens-row-high { background-color: #ef4444; color: white; font-weight: bold; }
                .upside-cell { background-color: #fef3c7; }
                </style>
                """, unsafe_allow_html=True)
                
                table_html = """
                <table class="sens-table">
                <tr>
                    <th></th>
                    <th>Growth Rate</th>
                    <th>Value</th>
                    <th>Upside</th>
                </tr>
                """
                
                for scenario in sensitivity:
                    label = scenario['label']
                    if label == 'Low':
                        row_class = 'sens-row-low'
                    elif label == 'Medium':
                        row_class = 'sens-row-med'
                    else:
                        row_class = 'sens-row-high'
                    
                    table_html += f"""
                    <tr>
                        <td class="{row_class}">{label}</td>
                        <td>{scenario['growth_rate']:.0f}%</td>
                        <td>${scenario['value']:.2f}</td>
                        <td class="upside-cell">{scenario['upside']:+.1f}%</td>
                    </tr>
                    """
                
                table_html += "</table>"
                st.markdown(table_html, unsafe_allow_html=True)
                
                # Add Reverse DCF below the table
                if implied_growth is not None:
                    st.markdown(f"""
                    <div style="margin-top: 15px; display: inline-block; border: 2px solid #000;">
                        <span style="background-color: #fef08a; padding: 8px 12px; font-weight: bold; display: inline-block;">Reverse DCF</span>
                        <span style="background-color: #fef08a; padding: 8px 12px; display: inline-block;">{implied_growth:.1f}%</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.caption("Sensitivity analysis not available.")
    else:
        st.info("💡 DCF valuation not available. Free Cash Flow data may be missing for this stock.")
    
    # Warning for negative earnings
    if data['current_eps'] and data['current_eps'] < 0:
        st.warning("⚠️ This stock has negative EPS. P/E-based valuation may not be applicable.")
    
    # Financial Health Score Section
    st.markdown("---")
    st.subheader("🏥 Financial Health Score")
    
    health = calculate_financial_health(data)
    
    if health.get('letter_grade'):
        # Main health score display
        health_cols = st.columns([1, 1, 1, 1, 1])
        
        # Color code the letter grade
        grade = health['letter_grade']
        if grade in ['A', 'B+']:
            grade_color = "🟢"
        elif grade in ['B', 'C+']:
            grade_color = "🟡"
        elif grade in ['C', 'D']:
            grade_color = "🟠"
        else:
            grade_color = "🔴"
        
        with health_cols[0]:
            st.metric(
                "Health Grade",
                f"{grade_color} {grade}",
                delta=health['health_status'],
                delta_color="off",
                help="Overall financial health grade based on debt levels, liquidity, and bankruptcy risk indicators."
            )
        
        with health_cols[1]:
            de = health.get('debt_to_equity')
            de_display = f"{de:.2f}" if de is not None else "N/A"
            de_status = "Low" if de and de < 0.5 else ("Moderate" if de and de < 1.5 else "High") if de else ""
            st.metric(
                "Debt/Equity",
                de_display,
                delta=de_status,
                delta_color="normal" if de and de < 1.0 else "inverse",
                help="Total Debt / Shareholder Equity. Lower is better. <0.5 is excellent, <1.0 is good, >2.0 is concerning."
            )
        
        with health_cols[2]:
            cr = health.get('current_ratio')
            cr_display = f"{cr:.2f}" if cr is not None else "N/A"
            cr_status = "Strong" if cr and cr >= 1.5 else ("Adequate" if cr and cr >= 1.0 else "Weak") if cr else ""
            st.metric(
                "Current Ratio",
                cr_display,
                delta=cr_status,
                delta_color="normal" if cr and cr >= 1.0 else "inverse",
                help="Current Assets / Current Liabilities. Measures short-term liquidity. >1.5 is healthy, <1.0 may indicate liquidity issues."
            )
        
        with health_cols[3]:
            ic = health.get('interest_coverage')
            if ic and ic >= 999:
                ic_display = "No Debt"
                ic_status = "Excellent"
            elif ic is not None:
                ic_display = f"{ic:.1f}x"
                ic_status = "Strong" if ic >= 4 else ("Adequate" if ic >= 2 else "Weak")
            else:
                ic_display = "N/A"
                ic_status = ""
            st.metric(
                "Interest Coverage",
                ic_display,
                delta=ic_status,
                delta_color="normal" if ic and ic >= 2 else ("inverse" if ic else "off"),
                help="EBITDA / Interest Expense. Measures ability to pay interest. >4x is strong, <2x is concerning."
            )
        
        with health_cols[4]:
            az = health.get('altman_z')
            if az is not None:
                az_display = f"{az:.2f}"
                if az > 3.0:
                    az_status = "Safe Zone"
                    az_color = "normal"
                elif az > 1.8:
                    az_status = "Grey Zone"
                    az_color = "off"
                else:
                    az_status = "Distress Zone"
                    az_color = "inverse"
            else:
                az_display = "N/A"
                az_status = ""
                az_color = "off"
            st.metric(
                "Altman Z-Score",
                az_display,
                delta=az_status,
                delta_color=az_color,
                help="Bankruptcy risk predictor. >3.0 = Safe, 1.8-3.0 = Grey Zone (caution), <1.8 = Distress Zone (high risk)."
            )
        
        # Expandable details
        with st.expander("📊 Financial Health Details"):
            detail_cols = st.columns(2)
            
            with detail_cols[0]:
                st.markdown("**Component Scores** (0-100)")
                scores = health.get('component_scores', {})
                for component, score in scores.items():
                    label = component.replace('_', ' ').title()
                    bar_color = "🟢" if score >= 70 else ("🟡" if score >= 50 else "🔴")
                    st.write(f"{bar_color} **{label}**: {score}/100")
                
                st.markdown(f"**Overall Score**: {health.get('overall_score', 0):.1f}/100")
            
            with detail_cols[1]:
                st.markdown("**Interpretation Guide**")
                st.markdown("""
                | Grade | Status | Meaning |
                |-------|--------|---------|
                | A | Excellent | Very strong balance sheet |
                | B+ | Very Good | Solid financial position |
                | B | Good | Healthy with minor concerns |
                | C+ | Fair | Some financial stress |
                | C | Below Avg | Notable weaknesses |
                | D | Poor | Significant concerns |
                | F | Distressed | High bankruptcy risk |
                """)
    else:
        st.info("💡 Financial health data not available for this stock.")
    
    # Graham Value Analysis Section (compact, integrated with existing flow)
    graham = calculate_graham_value(data)
    
    if graham.get('graham_number') or graham.get('ncav_per_share') or graham.get('pe_times_pb'):
        st.markdown("---")
        st.subheader("📖 Graham Value Analysis")
        
        # Show sector context
        sector_pe = graham.get('sector_pe', 20)
        sector_pb = graham.get('sector_pb', 3.0)
        sector_threshold = graham.get('sector_threshold', 60)
        graham_sector = graham.get('sector', 'Unknown')
        st.caption(f"**Sector-Adjusted** — Using {graham_sector} averages: P/E={sector_pe}, P/B={sector_pb}")
        
        graham_cols = st.columns(4)
        
        with graham_cols[0]:
            gn = graham.get('graham_number')
            gn_margin = graham.get('graham_margin')
            if gn:
                gn_display = f"${gn:.2f}"
                if gn_margin is not None:
                    margin_text = f"{gn_margin:+.1f}% vs price"
                    # Use "normal" so positive margin = green, negative margin = red
                    delta_color = "normal"
                else:
                    margin_text = ""
                    delta_color = "off"
            else:
                gn_display = "N/A"
                margin_text = "Requires positive EPS & book value"
                delta_color = "off"
            st.metric(
                "Graham Number",
                gn_display,
                delta=margin_text,
                delta_color=delta_color,
                help=f"Sector-adjusted fair value: √({sector_pe}×{sector_pb} × EPS × Book Value). If price < Graham Number, stock may be undervalued relative to sector."
            )
        
        with graham_cols[1]:
            ncav = graham.get('ncav_per_share')
            ncav_margin = graham.get('ncav_margin')
            if ncav is not None:
                if ncav > 0:
                    ncav_display = f"${ncav:.2f}"
                    if ncav_margin is not None:
                        margin_text = f"{ncav_margin:+.1f}% vs price"
                        # Use "normal" so positive margin = green, negative margin = red
                        delta_color = "normal"
                    else:
                        margin_text = ""
                        delta_color = "off"
                else:
                    ncav_display = f"${ncav:.2f}"
                    margin_text = "Negative (liabilities > current assets)"
                    delta_color = "off"
            else:
                ncav_display = "N/A"
                margin_text = ""
                delta_color = "off"
            st.metric(
                "NCAV/Share",
                ncav_display,
                delta=margin_text,
                delta_color=delta_color,
                help="Net Current Asset Value per share: (Current Assets - Total Liabilities) / Shares. If price < NCAV, it's a 'net-net' bargain."
            )
        
        with graham_cols[2]:
            pe_pb = graham.get('pe_times_pb')
            if pe_pb is not None:
                pe_pb_display = f"{pe_pb:.1f}"
                if pe_pb < sector_threshold:
                    status = f"Below sector avg ({sector_threshold:.0f})"
                    delta_color = "normal"
                else:
                    status = f"Above sector avg ({sector_threshold:.0f})"
                    delta_color = "inverse"
            else:
                pe_pb_display = "N/A"
                status = ""
                delta_color = "off"
            st.metric(
                "P/E × P/B",
                pe_pb_display,
                delta=status,
                delta_color=delta_color,
                help=f"Combined valuation ratio. Sector threshold ({graham_sector}): {sector_threshold:.0f}. Below threshold suggests relative value."
            )
        
        with graham_cols[3]:
            ey = graham.get('earnings_yield')
            if ey is not None:
                ey_display = f"{ey:.1f}%"
                # Compare to approximate bond yield (use 4.5% as reference)
                if ey > 6:
                    status = "Attractive vs bonds"
                    delta_color = "normal"
                elif ey > 4:
                    status = "Competitive"
                    delta_color = "off"
                else:
                    status = "Low yield"
                    delta_color = "inverse"
            else:
                ey_display = "N/A"
                status = ""
                delta_color = "off"
            st.metric(
                "Earnings Yield",
                ey_display,
                delta=status,
                delta_color=delta_color,
                help="EPS / Price (inverse of P/E). Compare to bond yields to assess relative value. Higher is better."
            )
        
        # Special callouts for significant findings
        if graham.get('is_net_net'):
            st.success("🎯 **Net-Net Bargain**: This stock trades below its Net Current Asset Value — a rare Graham-style deep value opportunity.")
        elif graham.get('is_undervalued') and graham.get('pe_times_pb') and graham['pe_times_pb'] < sector_threshold:
            st.info(f"✅ **Graham Criteria Met**: Price below Graham Number and P/E × P/B below sector threshold ({sector_threshold:.0f})")
    
    st.markdown("---")
    
    # Valuation Ratios Section
    st.subheader("📊 Valuation Ratios")
    
    # Sector-based typical ranges (approximate industry averages)
    sector = data.get('sector', 'Unknown')
    sector_benchmarks = {
        'Technology': {'p_sales': (3, 8), 'p_book': (3, 10), 'ev_ebitda': (12, 25), 'ev_revenue': (3, 8), 'trailing_pe': (20, 35), 'forward_pe': (15, 30)},
        'Healthcare': {'p_sales': (2, 6), 'p_book': (2, 6), 'ev_ebitda': (10, 20), 'ev_revenue': (2, 6), 'trailing_pe': (15, 30), 'forward_pe': (12, 25)},
        'Financial Services': {'p_sales': (2, 5), 'p_book': (1, 2), 'ev_ebitda': (8, 15), 'ev_revenue': (2, 5), 'trailing_pe': (10, 18), 'forward_pe': (8, 15)},
        'Consumer Cyclical': {'p_sales': (1, 3), 'p_book': (2, 6), 'ev_ebitda': (8, 15), 'ev_revenue': (1, 3), 'trailing_pe': (15, 25), 'forward_pe': (12, 22)},
        'Consumer Defensive': {'p_sales': (1, 3), 'p_book': (3, 8), 'ev_ebitda': (10, 18), 'ev_revenue': (1, 3), 'trailing_pe': (18, 28), 'forward_pe': (15, 25)},
        'Industrials': {'p_sales': (1, 3), 'p_book': (2, 5), 'ev_ebitda': (8, 15), 'ev_revenue': (1, 3), 'trailing_pe': (15, 25), 'forward_pe': (12, 22)},
        'Energy': {'p_sales': (0.5, 2), 'p_book': (1, 3), 'ev_ebitda': (4, 10), 'ev_revenue': (0.5, 2), 'trailing_pe': (8, 18), 'forward_pe': (6, 15)},
        'Communication Services': {'p_sales': (2, 5), 'p_book': (2, 6), 'ev_ebitda': (8, 18), 'ev_revenue': (2, 5), 'trailing_pe': (15, 30), 'forward_pe': (12, 25)},
        'Real Estate': {'p_sales': (3, 8), 'p_book': (1, 3), 'ev_ebitda': (12, 22), 'ev_revenue': (5, 12), 'trailing_pe': (25, 50), 'forward_pe': (20, 40)},
        'Basic Materials': {'p_sales': (1, 3), 'p_book': (1, 3), 'ev_ebitda': (5, 12), 'ev_revenue': (1, 3), 'trailing_pe': (10, 20), 'forward_pe': (8, 18)},
        'Utilities': {'p_sales': (1, 3), 'p_book': (1, 2), 'ev_ebitda': (8, 14), 'ev_revenue': (2, 4), 'trailing_pe': (15, 22), 'forward_pe': (12, 20)},
    }
    
    # Get benchmarks for this sector (default to Technology if unknown)
    benchmarks = sector_benchmarks.get(sector, sector_benchmarks['Technology'])
    
    st.caption(f"**Sector:** {sector} | **Industry:** {data.get('industry', 'Unknown')}")
    
    # Helper functions for formatting
    def format_ratio(value, suffix=""):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "N/A"
        return f"{value:.2f}{suffix}"
    
    def get_comparison(value, benchmark_range):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return ""
        low, high = benchmark_range
        if value < low:
            return "📉 Below"
        elif value > high:
            return "📈 Above"
        else:
            return "✅ In range"
    
    def format_range(benchmark_range, suffix=""):
        return f"{benchmark_range[0]}{suffix} - {benchmark_range[1]}{suffix}"
    
    # Create valuation ratios table as DataFrame
    valuation_data = {
        'Ratio': ['P/Sales', 'P/Book', 'EV/EBITDA', 'EV/Revenue', 'Enterprise Value', 'Trailing P/E', 'Forward P/E'],
        'Value': [
            format_ratio(data.get('p_sales')),
            format_ratio(data.get('p_book')),
            format_ratio(data.get('ev_ebitda')),
            format_ratio(data.get('ev_revenue')),
            format_large_number(data.get('enterprise_value')),
            format_ratio(data.get('trailing_pe'), "x"),
            format_ratio(data.get('forward_pe'), "x")
        ],
        f'Sector Range ({sector})': [
            format_range(benchmarks['p_sales']),
            format_range(benchmarks['p_book']),
            format_range(benchmarks['ev_ebitda']),
            format_range(benchmarks['ev_revenue']),
            "—",
            format_range(benchmarks['trailing_pe'], "x"),
            format_range(benchmarks['forward_pe'], "x")
        ],
        'vs Sector': [
            get_comparison(data.get('p_sales'), benchmarks['p_sales']),
            get_comparison(data.get('p_book'), benchmarks['p_book']),
            get_comparison(data.get('ev_ebitda'), benchmarks['ev_ebitda']),
            get_comparison(data.get('ev_revenue'), benchmarks['ev_revenue']),
            "—",
            get_comparison(data.get('trailing_pe'), benchmarks['trailing_pe']),
            get_comparison(data.get('forward_pe'), benchmarks['forward_pe'])
        ]
    }
    
    valuation_df = pd.DataFrame(valuation_data)
    valuation_df = valuation_df.set_index('Ratio')
    st.table(valuation_df)
    
    # Valuation ratio definitions
    with st.expander("ℹ️ Valuation Ratio Definitions (click to expand)"):
        st.markdown("""
        | Ratio | Description | Interpretation |
        |-------|-------------|----------------|
        | **P/Sales** | Price-to-Sales = Market Cap ÷ Revenue | Lower may indicate undervaluation; compare within sector |
        | **P/Book** | Price-to-Book = Price ÷ Book Value per Share | <1 may be undervalued; >3 often growth stocks |
        | **EV/EBITDA** | Enterprise Value ÷ EBITDA | Lower = cheaper; useful for comparing companies with different capital structures |
        | **EV/Revenue** | Enterprise Value ÷ Revenue | Similar to P/Sales but accounts for debt |
        | **Enterprise Value** | Market Cap + Debt - Cash | Total takeover value of the company |
        | **Trailing P/E** | Price ÷ Last 12 Months EPS | Based on actual earnings; higher = more expensive |
        | **Forward P/E** | Price ÷ Expected Next Year EPS | Based on analyst estimates; lower than trailing = expected growth |
        """)
    
    st.markdown("---")
    
    # Scenario Settings
    st.subheader("⚙️ Scenario Settings")
    
    # Get defaults from current data - ensure positive values for margins
    raw_margin = data['current_margin'] if data['current_margin'] else 20.0
    default_margin = max(raw_margin, 5.0)  # Ensure at least 5% margin for loss-making companies
    current_rev_growth = max(data.get('revenue_growth', 10.0), 1.0)  # Ensure positive growth default
    
    # Calculate margin values ensuring they stay within bounds
    bear_margin_val = max(1.0, min(60.0, float(default_margin) * 0.9))
    base_margin_val = max(1.0, min(60.0, float(default_margin)))
    bull_margin_val = max(1.0, min(60.0, float(default_margin) * 1.1))
    
    # Calculate growth values ensuring they stay within bounds
    bear_growth_val = max(-20.0, min(100.0, float(current_rev_growth) * 0.5))
    base_growth_val = max(-20.0, min(100.0, float(current_rev_growth)))
    bull_growth_val = max(-20.0, min(100.0, float(current_rev_growth) * 1.5))
    
    settings_col1, settings_col2, settings_col3, settings_col4 = st.columns(4)
    
    # Use ticker in key to force reset when ticker changes
    ticker_key = st.session_state.ticker
    
    with settings_col1:
        st.markdown("**🐻 Bear Case**")
        bear_growth = st.number_input("Revenue Growth %", min_value=-20.0, max_value=100.0, value=bear_growth_val, step=1.0, key=f"bear_growth_{ticker_key}", help="Annual revenue growth rate for pessimistic scenario. Default is 0.5x current growth.")
        bear_margin = st.number_input("Net Margin %", min_value=1.0, max_value=60.0, value=bear_margin_val, step=0.5, key=f"bear_margin_{ticker_key}", help="Net income as % of revenue for bear case. Default is 0.9x current margin.")
    
    with settings_col2:
        st.markdown("**📊 Base Case**")
        base_growth = st.number_input("Revenue Growth %", min_value=-20.0, max_value=100.0, value=base_growth_val, step=1.0, key=f"base_growth_{ticker_key}", help="Annual revenue growth rate for base scenario. Default is current growth rate.")
        base_margin = st.number_input("Net Margin %", min_value=1.0, max_value=60.0, value=base_margin_val, step=0.5, key=f"base_margin_{ticker_key}", help="Net income as % of revenue for base case. Default is current margin.")
    
    with settings_col3:
        st.markdown("**🐂 Bull Case**")
        bull_growth = st.number_input("Revenue Growth %", min_value=-20.0, max_value=100.0, value=bull_growth_val, step=1.0, key=f"bull_growth_{ticker_key}", help="Annual revenue growth rate for optimistic scenario. Default is 1.5x current growth.")
        bull_margin = st.number_input("Net Margin %", min_value=1.0, max_value=60.0, value=bull_margin_val, step=0.5, key=f"bull_margin_{ticker_key}", help="Net income as % of revenue for bull case. Default is 1.1x current margin.")
    
    with settings_col4:
        st.markdown("**📐 P/E Multiples**")
        _, pe_low, pe_high, _, _, using_fallback = calculate_projections(data, base_growth, base_margin)
        if using_fallback:
            st.caption("Using historical P/E ±20%")
        st.metric("P/E Low", f"{pe_low:.1f}x" if pe_low else "N/A", label_visibility="visible", help="Lower P/E multiple derived from analyst high EPS estimate. Used to calculate conservative price targets.")
        st.metric("P/E High", f"{pe_high:.1f}x" if pe_high else "N/A", label_visibility="visible", help="Higher P/E multiple derived from analyst low EPS estimate. Used to calculate optimistic price targets.")
    
    st.markdown("---")
    
    # --- THREE SCENARIO TABLES ---
    st.subheader("📊 5-Year Projection Tables")
    
    # Metric explanations expander
    with st.expander("ℹ️ Metric Definitions (click to expand)"):
        st.markdown("""
        | Metric | Description |
        |--------|-------------|
        | **Revenue** | Total sales/income before any expenses |
        | **Rev Growth** | Year-over-year percentage increase in revenue |
        | **Net Income** | Profit after all expenses, taxes, and costs |
        | **Net Inc. Growth** | Year-over-year percentage increase in net income |
        | **Net Inc. Margins** | Net Income ÷ Revenue. Profitability ratio |
        | **EPS** | Earnings Per Share = Net Income ÷ Shares Outstanding |
        | **P/E Low Est** | Conservative valuation multiple (Price ÷ High EPS Estimate) |
        | **P/E High Est** | Optimistic valuation multiple (Price ÷ Low EPS Estimate) |
        | **Share Price Low** | EPS × P/E Low. Conservative price target |
        | **Share Price High** | EPS × P/E High. Optimistic price target |
        | **CAGR Low/High** | Compound Annual Growth Rate from current price to target |
        """)
    
    # Toggle for historical data
    show_history = st.checkbox("📅 Show Historical Data (3 previous years)", value=False, key="show_history_toggle")
    
    st.markdown("")
    
    # Bear Case Table
    st.markdown("### 🐻 Bear Case")
    bear_df, bear_proj, bear_cagr_low, bear_cagr_high = build_scenario_table(data, bear_growth, bear_margin, show_history)
    st.table(bear_df)
    
    st.markdown("")
    
    # Base Case Table
    st.markdown("### 📊 Base Case")
    base_df, base_proj, base_cagr_low, base_cagr_high = build_scenario_table(data, base_growth, base_margin, show_history)
    st.table(base_df)
    
    st.markdown("")
    
    # Bull Case Table
    st.markdown("### 🐂 Bull Case")
    bull_df, bull_proj, bull_cagr_low, bull_cagr_high = build_scenario_table(data, bull_growth, bull_margin, show_history)
    st.table(bull_df)

elif 'stock_data' in st.session_state and not st.session_state.stock_data.get('success'):
    st.error(f"❌ Error fetching data: {st.session_state.stock_data.get('error')}")
else:
    st.info("Enter a ticker symbol above and press Enter to begin.")

# Footer
st.markdown("---")

# Help/Documentation section
footer_col1, footer_col2 = st.columns([1, 5])
with footer_col1:
    show_help = st.button("ℹ️ Help & Documentation", key="show_help_docs", use_container_width=True)

if show_help:
    st.session_state.show_help_dialog = True

if st.session_state.get('show_help_dialog', False):
    @st.dialog("📖 Stock Analysis - Documentation", width="large")
    def show_help_dialog():
        # Read the markdown file
        try:
            with open("STOCK_ANALYSIS.md", "r") as f:
                help_content = f.read()
            st.markdown(help_content)
        except FileNotFoundError:
            st.error("Documentation file not found.")
        
        if st.button("Close", key="close_help"):
            st.session_state.show_help_dialog = False
            st.rerun()
    
    show_help_dialog()

st.caption("Data provided by Yahoo Finance via yfinance. This tool is for educational purposes only and does not constitute financial advice.")
