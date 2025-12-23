import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(page_title="Stock Valuation Projection Engine", layout="wide")

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
</style>
""", unsafe_allow_html=True)

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
        if income_stmt is not None and not income_stmt.empty:
            total_revenue = income_stmt.loc['Total Revenue'].iloc[0] if 'Total Revenue' in income_stmt.index else None
            net_income = income_stmt.loc['Net Income'].iloc[0] if 'Net Income' in income_stmt.index else None
            # Get the fiscal year from the most recent column
            fiscal_year = income_stmt.columns[0].year if hasattr(income_stmt.columns[0], 'year') else None
            # Calculate revenue growth from last 2 years
            if 'Total Revenue' in income_stmt.index and len(income_stmt.columns) >= 2:
                rev_current = income_stmt.loc['Total Revenue'].iloc[0]
                rev_prior = income_stmt.loc['Total Revenue'].iloc[1]
                if rev_prior and rev_prior > 0:
                    revenue_growth = ((rev_current / rev_prior) - 1) * 100
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
            'success': True,
            'error': None
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def calculate_projections(data: dict, revenue_growth: float, net_margin: float) -> pd.DataFrame:
    """Calculate 3-year projections."""
    current_price = data['current_price']
    revenue = data['total_revenue']
    shares = data['shares_outstanding']
    eps_low = data['eps_low']
    eps_high = data['eps_high']
    historical_pe = data['historical_pe']
    fiscal_year = data.get('fiscal_year', 2024)
    
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
    
    # Build projection table - Years 0-3 (current + 3 future years)
    num_years = 4
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
    
    for idx in range(num_years):
        if idx == 0:
            # Current year (Year 0)
            proj_revenue = revenue
            proj_net_income = data['net_income']
            proj_eps = data['current_eps']
            rev_growth = None
            ni_growth = None
            margin = data['current_margin']
            price_low = current_price
            price_high = current_price
        else:
            # Future years
            proj_revenue = revenue * ((1 + revenue_growth / 100) ** idx)
            proj_net_income = proj_revenue * (net_margin / 100)
            proj_eps = proj_net_income / shares if shares else None
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
    
    # Calculate CAGR for final year (Year 3)
    cagr_low = None
    cagr_high = None
    final_idx = num_years - 1
    if projections['Share Price Low'][final_idx] and current_price and current_price > 0:
        cagr_low = ((projections['Share Price Low'][final_idx] / current_price) ** (1/final_idx) - 1) * 100
    if projections['Share Price High'][final_idx] and current_price and current_price > 0:
        cagr_high = ((projections['Share Price High'][final_idx] / current_price) ** (1/final_idx) - 1) * 100
    
    return projections, pe_low, pe_high, cagr_low, cagr_high, using_fallback

def build_scenario_table(data: dict, revenue_growth: float, net_margin: float) -> pd.DataFrame:
    """Build a formatted DataFrame for a scenario."""
    projections, pe_low, pe_high, cagr_low, cagr_high, _ = calculate_projections(
        data, revenue_growth, net_margin
    )
    
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
    
    for metric_key, metric_name, formatter in metrics:
        row = {'Metric': metric_name}
        for i, year in enumerate(projections['Year']):
            row[f'{year}'] = formatter(projections[metric_key][i])
        display_data.append(row)
    
    # Add CAGR rows
    num_years = len(projections['Year'])
    cagr_low_row = {'Metric': 'CAGR Low'}
    cagr_high_row = {'Metric': 'CAGR High'}
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

# --- TOP HEADER WITH SEARCH ---
st.title("📈 Stock Valuation Projection Engine")
st.markdown("*3-Year Forward-Estimate Model with Bull, Base & Bear Scenarios*")

# Search bar at the top with Fetch Data button
col_search, col_fetch = st.columns([4, 1])
with col_search:
    ticker = st.text_input("🔍 Enter Stock Ticker", value="AVGO", placeholder="e.g., AAPL, MSFT, AVGO", label_visibility="collapsed")
with col_fetch:
    fetch_clicked = st.button("Fetch Data", type="primary", use_container_width=True)

# Auto-fetch when ticker changes, on first load, or when button clicked
if 'last_ticker' not in st.session_state:
    st.session_state.last_ticker = None

if fetch_clicked or ticker.upper() != st.session_state.last_ticker or 'stock_data' not in st.session_state:
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
if 'stock_data' in st.session_state and st.session_state.stock_data.get('success'):
    news_data = st.session_state.stock_data.get('news', [])
    calendar_data = st.session_state.stock_data.get('calendar')
    earnings_dates = st.session_state.stock_data.get('earnings_dates')
    
    with st.sidebar:
        # Calendar / Upcoming Events Section
        st.header(f"📅 {st.session_state.ticker} Upcoming Events")
        
        has_events = False
        
        # Display calendar events (earnings, dividends, etc.)
        if calendar_data is not None:
            if isinstance(calendar_data, dict):
                # Earnings Date
                if 'Earnings Date' in calendar_data:
                    earnings = calendar_data['Earnings Date']
                    if earnings:
                        has_events = True
                        if isinstance(earnings, list) and len(earnings) >= 2:
                            st.markdown(f"📊 **Earnings:** {earnings[0].strftime('%b %d, %Y')} - {earnings[1].strftime('%b %d, %Y')}")
                        elif isinstance(earnings, list) and len(earnings) == 1:
                            st.markdown(f"📊 **Earnings:** {earnings[0].strftime('%b %d, %Y')}")
                
                # Dividend Date
                if 'Dividend Date' in calendar_data and calendar_data['Dividend Date']:
                    has_events = True
                    div_date = calendar_data['Dividend Date']
                    if hasattr(div_date, 'strftime'):
                        st.markdown(f"💰 **Dividend Date:** {div_date.strftime('%b %d, %Y')}")
                
                # Ex-Dividend Date
                if 'Ex-Dividend Date' in calendar_data and calendar_data['Ex-Dividend Date']:
                    has_events = True
                    ex_div = calendar_data['Ex-Dividend Date']
                    if hasattr(ex_div, 'strftime'):
                        st.markdown(f"📆 **Ex-Dividend:** {ex_div.strftime('%b %d, %Y')}")
        
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
                pub_date = content.get('pubDate', '')
                
                st.markdown(f"📄 **[{title}]({link})**")
                if provider:
                    st.caption(f"🔗 {provider}")
                st.markdown("---")
        else:
            st.info("No recent news available.")

# --- Settings Row ---
if 'stock_data' in st.session_state and st.session_state.stock_data.get('success'):
    data = st.session_state.stock_data
    
    st.markdown("---")
    
    # Company info header
    st.header(f"{data['company_name']} ({st.session_state.ticker})")
    
    # Current metrics
    m1, m2, m3, m4, m5 = st.columns(5)
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
    
    # Warning for negative earnings
    if data['current_eps'] and data['current_eps'] < 0:
        st.warning("⚠️ This stock has negative EPS. P/E-based valuation may not be applicable.")
    
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
    
    with settings_col1:
        st.markdown("**🐻 Bear Case**")
        bear_growth = st.number_input("Revenue Growth %", min_value=-20.0, max_value=100.0, value=bear_growth_val, step=1.0, key="bear_growth", help="Annual revenue growth rate for pessimistic scenario. Default is 0.5x current growth.")
        bear_margin = st.number_input("Net Margin %", min_value=1.0, max_value=60.0, value=bear_margin_val, step=0.5, key="bear_margin", help="Net income as % of revenue for bear case. Default is 0.9x current margin.")
    
    with settings_col2:
        st.markdown("**📊 Base Case**")
        base_growth = st.number_input("Revenue Growth %", min_value=-20.0, max_value=100.0, value=base_growth_val, step=1.0, key="base_growth", help="Annual revenue growth rate for base scenario. Default is current growth rate.")
        base_margin = st.number_input("Net Margin %", min_value=1.0, max_value=60.0, value=base_margin_val, step=0.5, key="base_margin", help="Net income as % of revenue for base case. Default is current margin.")
    
    with settings_col3:
        st.markdown("**🐂 Bull Case**")
        bull_growth = st.number_input("Revenue Growth %", min_value=-20.0, max_value=100.0, value=bull_growth_val, step=1.0, key="bull_growth", help="Annual revenue growth rate for optimistic scenario. Default is 1.5x current growth.")
        bull_margin = st.number_input("Net Margin %", min_value=1.0, max_value=60.0, value=bull_margin_val, step=0.5, key="bull_margin", help="Net income as % of revenue for bull case. Default is 1.1x current margin.")
    
    with settings_col4:
        st.markdown("**📐 P/E Multiples**")
        _, pe_low, pe_high, _, _, using_fallback = calculate_projections(data, base_growth, base_margin)
        if using_fallback:
            st.caption("Using historical P/E ±20%")
        st.metric("P/E Low", f"{pe_low:.1f}x" if pe_low else "N/A", label_visibility="visible", help="Lower P/E multiple derived from analyst high EPS estimate. Used to calculate conservative price targets.")
        st.metric("P/E High", f"{pe_high:.1f}x" if pe_high else "N/A", label_visibility="visible", help="Higher P/E multiple derived from analyst low EPS estimate. Used to calculate optimistic price targets.")
    
    st.markdown("---")
    
    # --- THREE SCENARIO TABLES ---
    st.subheader("📊 3-Year Projection Tables")
    
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
    
    # Bear Case Table
    st.markdown("### 🐻 Bear Case")
    bear_df, bear_proj, bear_cagr_low, bear_cagr_high = build_scenario_table(data, bear_growth, bear_margin)
    st.table(bear_df)
    
    st.markdown("")
    
    # Base Case Table
    st.markdown("### 📊 Base Case")
    base_df, base_proj, base_cagr_low, base_cagr_high = build_scenario_table(data, base_growth, base_margin)
    st.table(base_df)
    
    st.markdown("")
    
    # Bull Case Table
    st.markdown("### 🐂 Bull Case")
    bull_df, bull_proj, bull_cagr_low, bull_cagr_high = build_scenario_table(data, bull_growth, bull_margin)
    st.table(bull_df)

elif 'stock_data' in st.session_state and not st.session_state.stock_data.get('success'):
    st.error(f"❌ Error fetching data: {st.session_state.stock_data.get('error')}")
else:
    st.info("Enter a ticker symbol above and press Enter to begin.")

# Footer
st.markdown("---")
st.caption("Data provided by Yahoo Finance via yfinance. This tool is for educational purposes only and does not constitute financial advice.")
