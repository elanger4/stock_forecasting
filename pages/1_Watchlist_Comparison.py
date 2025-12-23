import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import sys
import os

# Add parent directory to path to import from app.py and watchlists
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import centralized watchlist configuration
from watchlists import WATCHLISTS, get_watchlist_names, get_watchlist_stocks, get_default_watchlist

st.set_page_config(page_title="Watchlist Comparison", layout="wide")

# Initialize watchlist selection in session state
if 'selected_watchlist' not in st.session_state:
    st.session_state.selected_watchlist = get_default_watchlist()

# Hide Streamlit default header but keep sidebar
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    [data-testid="stSidebarNav"] {display: none;}
</style>
""", unsafe_allow_html=True)

# --- Page Navigation & Watchlist Selector ---
nav_col1, nav_col2, nav_col3, nav_col4 = st.columns([1, 1, 1, 3])
with nav_col1:
    st.markdown("👉 [Stock Analysis](/)")
with nav_col2:
    st.markdown("**📋 Watchlist Comparison** *(current)*")
with nav_col3:
    # Watchlist selector
    watchlist_names = get_watchlist_names()
    current_idx = watchlist_names.index(st.session_state.selected_watchlist) if st.session_state.selected_watchlist in watchlist_names else 0
    new_watchlist = st.selectbox(
        "📋 Watchlist",
        options=watchlist_names,
        index=current_idx,
        key="comparison_watchlist_selector",
        label_visibility="collapsed"
    )
    if new_watchlist != st.session_state.selected_watchlist:
        st.session_state.selected_watchlist = new_watchlist
        # Clear comparison data when watchlist changes
        if 'comparison_loaded' in st.session_state:
            st.session_state.comparison_loaded = False
        st.rerun()

# Get current watchlist stocks
WATCHLIST_STOCKS = get_watchlist_stocks(st.session_state.selected_watchlist)

st.markdown("---")

# --- Custom CSS (same as main app) ---
st.markdown("""
<style>
    html, body, [class*="css"] { font-size: 18px; }
    h1 { font-size: 2.5rem !important; }
    h2 { font-size: 2rem !important; }
    h3 { font-size: 1.6rem !important; }
    [data-testid="stMetricValue"] { font-size: 1.8rem !important; }
    .stTable table { font-size: 1.1rem !important; }
    .stTable th { font-size: 1.15rem !important; padding: 12px 16px !important; }
    .stTable td { font-size: 1.1rem !important; padding: 10px 16px !important; }
</style>
""", unsafe_allow_html=True)

# --- Helper Functions ---
def format_currency(value: float) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    return f"${value:,.2f}"

def format_large_number(value: float) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
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

def fetch_stock_data(ticker: str) -> dict:
    """Fetch stock data from yfinance."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        current_price = info.get('currentPrice') or info.get('regularMarketPrice') or info.get('previousClose')
        shares_outstanding = info.get('sharesOutstanding')
        
        income_stmt = stock.income_stmt
        fiscal_year = None
        revenue_growth = None
        
        if income_stmt is not None and not income_stmt.empty:
            total_revenue = income_stmt.loc['Total Revenue'].iloc[0] if 'Total Revenue' in income_stmt.index else None
            net_income = income_stmt.loc['Net Income'].iloc[0] if 'Net Income' in income_stmt.index else None
            fiscal_year = income_stmt.columns[0].year if hasattr(income_stmt.columns[0], 'year') else None
            
            if 'Total Revenue' in income_stmt.index and len(income_stmt.columns) >= 2:
                rev_current = income_stmt.loc['Total Revenue'].iloc[0]
                rev_prior = income_stmt.loc['Total Revenue'].iloc[1]
                if rev_prior and rev_prior > 0:
                    revenue_growth = ((rev_current / rev_prior) - 1) * 100
        else:
            total_revenue = info.get('totalRevenue')
            net_income = info.get('netIncomeToCommon')
        
        if revenue_growth is None:
            revenue_growth = info.get('revenueGrowth')
            if revenue_growth:
                revenue_growth = revenue_growth * 100
        
        current_eps = info.get('trailingEps')
        
        # Analyst estimates
        eps_low = None
        eps_high = None
        try:
            earnings_estimate = stock.earnings_estimate
            if earnings_estimate is not None and not earnings_estimate.empty:
                if '+1y' in earnings_estimate.columns:
                    eps_low = earnings_estimate.loc['low', '+1y'] if 'low' in earnings_estimate.index else None
                    eps_high = earnings_estimate.loc['high', '+1y'] if 'high' in earnings_estimate.index else None
        except:
            pass
        
        if eps_low is None:
            eps_low = info.get('earningsLow')
        if eps_high is None:
            eps_high = info.get('earningsHigh')
        
        current_margin = None
        if total_revenue and net_income and total_revenue > 0:
            current_margin = (net_income / total_revenue) * 100
        
        historical_pe = info.get('trailingPE')
        
        if fiscal_year is None:
            from datetime import datetime
            fiscal_year = datetime.now().year
        
        fifty_two_week_low = info.get('fiftyTwoWeekLow')
        fifty_two_week_high = info.get('fiftyTwoWeekHigh')
        market_cap = info.get('marketCap')
        
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
            'fifty_two_week_low': fifty_two_week_low,
            'fifty_two_week_high': fifty_two_week_high,
            'market_cap': market_cap,
            'success': True,
            'error': None
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}

def calculate_projections(data: dict, revenue_growth: float, net_margin: float):
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
    market_cap = data.get('market_cap')
    
    pe_high = None
    pe_low = None
    use_ps_ratio = False  # Flag for Price/Sales valuation fallback
    ps_ratio = None
    
    # Try P/E based valuation first
    if eps_low and eps_high and eps_low > 0 and eps_high > 0:
        pe_high = current_price / eps_low
        pe_low = current_price / eps_high
    elif historical_pe and historical_pe > 0:
        pe_low = historical_pe * 0.8
        pe_high = historical_pe * 1.2
    
    # Fallback to Price/Sales ratio for negative EPS stocks
    if pe_low is None or pe_high is None:
        if market_cap and revenue and revenue > 0:
            ps_ratio = market_cap / revenue
            use_ps_ratio = True
        elif current_price and shares and revenue and revenue > 0:
            # Calculate P/S from price and revenue per share
            revenue_per_share = revenue / shares
            ps_ratio = current_price / revenue_per_share
            use_ps_ratio = True
    
    num_years = 6
    calendar_years = [fiscal_year + i for i in range(num_years)]
    projections = {
        'Year': calendar_years,
        'Revenue': [],
        'Net Income': [],
        'EPS': [],
        'Share Price Low': [],
        'Share Price High': [],
    }
    
    prev_net_income = data['net_income']
    
    for i in range(num_years):
        if i == 0:
            proj_revenue = revenue
            proj_net_income = data['net_income']
            proj_eps = data['current_eps']
            price_low = fifty_two_week_low
            price_high = fifty_two_week_high
        else:
            proj_revenue = revenue * ((1 + revenue_growth / 100) ** i)
            proj_net_income = proj_revenue * (net_margin / 100)
            proj_eps = proj_net_income / shares if shares else None
            
            if pe_low and pe_high and proj_eps and proj_eps > 0:
                # P/E based valuation
                price_low = proj_eps * pe_low
                price_high = proj_eps * pe_high
            elif use_ps_ratio and ps_ratio and shares:
                # Price/Sales fallback for negative EPS stocks
                # Use P/S ratio with a range (0.8x to 1.2x current P/S)
                revenue_per_share = proj_revenue / shares
                price_low = revenue_per_share * (ps_ratio * 0.7)  # Bear case: P/S compression
                price_high = revenue_per_share * (ps_ratio * 1.3)  # Bull case: P/S expansion
            else:
                price_low = None
                price_high = None
        
        projections['Revenue'].append(proj_revenue)
        projections['Net Income'].append(proj_net_income)
        projections['EPS'].append(proj_eps)
        projections['Share Price Low'].append(price_low)
        projections['Share Price High'].append(price_high)
    
    # Calculate CAGR for Year 5
    cagr_low = None
    cagr_high = None
    final_idx = num_years - 1
    if projections['Share Price Low'][final_idx] and current_price and current_price > 0:
        cagr_low = ((projections['Share Price Low'][final_idx] / current_price) ** (1/final_idx) - 1) * 100
    if projections['Share Price High'][final_idx] and current_price and current_price > 0:
        cagr_high = ((projections['Share Price High'][final_idx] / current_price) ** (1/final_idx) - 1) * 100
    
    return projections, pe_low, pe_high, cagr_low, cagr_high

@st.cache_data(ttl=600, show_spinner=False)
def get_stock_comparison_data(ticker: str):
    """Fetch stock data and calculate projections for comparison."""
    try:
        data = fetch_stock_data(ticker)
        if not data.get('success'):
            return None
        
        current_price = data.get('current_price')
        revenue_growth = data.get('revenue_growth', 10.0)
        current_margin = data.get('current_margin')
        
        if current_margin and current_margin > 0:
            base_margin = current_margin
        else:
            base_margin = 10.0
        
        scenarios = {
            'bear': {'growth': max(0, revenue_growth * 0.5), 'margin': max(1, base_margin * 0.8)},
            'base': {'growth': revenue_growth, 'margin': base_margin},
            'bull': {'growth': revenue_growth * 1.5, 'margin': min(40, base_margin * 1.2)},
        }
        
        results = {
            'ticker': ticker,
            'current_price': current_price,
            'company_name': data.get('company_name', ticker),
        }
        
        for scenario_name, params in scenarios.items():
            try:
                projections, pe_low, pe_high, cagr_low, cagr_high = calculate_projections(
                    data, params['growth'], params['margin']
                )
                # 5-year values (index 5)
                results[f'{scenario_name}_price_low'] = projections['Share Price Low'][5] if projections['Share Price Low'][5] else None
                results[f'{scenario_name}_price_high'] = projections['Share Price High'][5] if projections['Share Price High'][5] else None
                results[f'{scenario_name}_cagr_low'] = cagr_low
                results[f'{scenario_name}_cagr_high'] = cagr_high
                
                # 1-year values (index 1) for time-alignment check
                price_1y_low = projections['Share Price Low'][1] if projections['Share Price Low'][1] else None
                price_1y_high = projections['Share Price High'][1] if projections['Share Price High'][1] else None
                if price_1y_low and current_price and current_price > 0:
                    results[f'{scenario_name}_return_1y_low'] = ((price_1y_low / current_price) - 1) * 100
                else:
                    results[f'{scenario_name}_return_1y_low'] = None
                if price_1y_high and current_price and current_price > 0:
                    results[f'{scenario_name}_return_1y_high'] = ((price_1y_high / current_price) - 1) * 100
                else:
                    results[f'{scenario_name}_return_1y_high'] = None
            except:
                results[f'{scenario_name}_price_low'] = None
                results[f'{scenario_name}_price_high'] = None
                results[f'{scenario_name}_cagr_low'] = None
                results[f'{scenario_name}_cagr_high'] = None
                results[f'{scenario_name}_return_1y_low'] = None
                results[f'{scenario_name}_return_1y_high'] = None
        
        # Add market cap for optimism bias check
        results['market_cap'] = data.get('market_cap')
        
        return results
    except:
        return None

def calculate_conviction_score(stock_data: dict, hurdle_rate: float = 10.0, 
                                scenario_probs: dict = None) -> dict:
    """
    Calculate Conviction Score based on weighted CAGR, time-alignment, and variance.
    
    Args:
        stock_data: Dict with bear/base/bull CAGR and 1-year returns
        hurdle_rate: Minimum acceptable CAGR (default 10% = index fund benchmark)
        scenario_probs: Probability weights for each scenario (default: bear=0.25, base=0.50, bull=0.25)
    
    Returns:
        ConvictionSummary dict with expected_value_5y, conviction_rating, risk_profile, warnings
    """
    if scenario_probs is None:
        scenario_probs = {'bear': 0.25, 'base': 0.50, 'bull': 0.25}
    
    # Extract CAGR values (use average of low/high for each scenario)
    bear_cagr = None
    base_cagr = None
    bull_cagr = None
    
    if stock_data.get('bear_cagr_low') is not None and stock_data.get('bear_cagr_high') is not None:
        bear_cagr = (stock_data['bear_cagr_low'] + stock_data['bear_cagr_high']) / 2
    elif stock_data.get('bear_cagr_high') is not None:
        bear_cagr = stock_data['bear_cagr_high']
    elif stock_data.get('bear_cagr_low') is not None:
        bear_cagr = stock_data['bear_cagr_low']
    
    if stock_data.get('base_cagr_low') is not None and stock_data.get('base_cagr_high') is not None:
        base_cagr = (stock_data['base_cagr_low'] + stock_data['base_cagr_high']) / 2
    elif stock_data.get('base_cagr_high') is not None:
        base_cagr = stock_data['base_cagr_high']
    elif stock_data.get('base_cagr_low') is not None:
        base_cagr = stock_data['base_cagr_low']
    
    if stock_data.get('bull_cagr_low') is not None and stock_data.get('bull_cagr_high') is not None:
        bull_cagr = (stock_data['bull_cagr_low'] + stock_data['bull_cagr_high']) / 2
    elif stock_data.get('bull_cagr_high') is not None:
        bull_cagr = stock_data['bull_cagr_high']
    elif stock_data.get('bull_cagr_low') is not None:
        bull_cagr = stock_data['bull_cagr_low']
    
    # Initialize result
    result = {
        'ticker': stock_data.get('ticker'),
        'expected_cagr': None,
        'expected_value_5y': None,
        'conviction_rating': 'N/A',
        'risk_profile': 'N/A',
        'time_alignment': 'N/A',
        'warnings': []
    }
    
    # Check if we have enough data
    if bear_cagr is None or base_cagr is None or bull_cagr is None:
        result['warnings'].append("Insufficient CAGR data")
        return result
    
    # Step A: Calculate Weighted CAGR (Expected CAGR)
    expected_cagr = (
        bear_cagr * scenario_probs['bear'] +
        base_cagr * scenario_probs['base'] +
        bull_cagr * scenario_probs['bull']
    )
    result['expected_cagr'] = expected_cagr
    
    # Calculate expected 5-year price
    current_price = stock_data.get('current_price')
    if current_price and current_price > 0:
        result['expected_value_5y'] = current_price * ((1 + expected_cagr / 100) ** 5)
    
    # Determine conviction rating based on hurdle rate
    if expected_cagr >= hurdle_rate + 10:
        result['conviction_rating'] = '🟢 Strong Buy'
    elif expected_cagr >= hurdle_rate:
        result['conviction_rating'] = '🟡 Buy'
    elif expected_cagr >= hurdle_rate - 5:
        result['conviction_rating'] = '🟠 Hold'
    elif expected_cagr >= 0:
        result['conviction_rating'] = '🔴 Avoid'
    else:
        result['conviction_rating'] = '⛔ Strong Avoid'
    
    # Step B: Time-Alignment Sanity Check (compare 1-year to 5-year)
    # Use base case for 1-year return
    return_1y = None
    if stock_data.get('base_return_1y_low') is not None and stock_data.get('base_return_1y_high') is not None:
        return_1y = (stock_data['base_return_1y_low'] + stock_data['base_return_1y_high']) / 2
    elif stock_data.get('base_return_1y_high') is not None:
        return_1y = stock_data['base_return_1y_high']
    elif stock_data.get('base_return_1y_low') is not None:
        return_1y = stock_data['base_return_1y_low']
    
    if return_1y is not None:
        if return_1y > 10 and expected_cagr < 0:
            result['time_alignment'] = '⚠️ Value Trap'
            result['warnings'].append("Short-term positive but long-term negative - potential value trap")
        elif return_1y < -10 and expected_cagr > 15:
            result['time_alignment'] = '💎 Accumulation'
            result['warnings'].append("Short-term pain, long-term gain - accumulation opportunity")
        elif return_1y > 0 and expected_cagr > 0:
            result['time_alignment'] = '✅ Aligned'
        elif return_1y < 0 and expected_cagr < 0:
            result['time_alignment'] = '⚠️ Bearish'
        else:
            result['time_alignment'] = '➖ Mixed'
    
    # Step C: Variance Scoring (Predictability Metric)
    delta_cagr = bull_cagr - bear_cagr
    result['delta_cagr'] = delta_cagr
    
    if delta_cagr > 30:
        result['risk_profile'] = '🎰 Speculative'
    elif delta_cagr > 20:
        result['risk_profile'] = '📈 High Vol'
    elif delta_cagr > 10:
        result['risk_profile'] = '📊 Medium Vol'
    else:
        result['risk_profile'] = '🏦 Predictable'
    
    # Edge Case: Optimism Bias Check
    market_cap = stock_data.get('market_cap')
    if market_cap and bull_cagr:
        # For trillion-dollar companies, flag if CAGR > 30%
        if market_cap >= 1e12 and bull_cagr > 30:
            result['warnings'].append(f"⚠️ Optimism Bias: {bull_cagr:.0f}% CAGR unlikely for ${market_cap/1e12:.1f}T company")
        # For $100B+ companies, flag if CAGR > 50%
        elif market_cap >= 1e11 and bull_cagr > 50:
            result['warnings'].append(f"⚠️ Optimism Bias: {bull_cagr:.0f}% CAGR unlikely for ${market_cap/1e9:.0f}B company")
        # For any company, flag if CAGR > 80%
        elif bull_cagr > 80:
            result['warnings'].append(f"⚠️ Optimism Bias: {bull_cagr:.0f}% CAGR is mathematically extreme")
    
    return result

# --- Page Content ---
st.title("📋 Watchlist Comparison")
st.markdown("*Compare Bear, Base & Bull scenarios across all watchlist stocks*")

# Load comparison data button
if st.button("🔄 Load/Refresh Comparison Data", type="primary"):
    st.session_state.comparison_loaded = True
    get_stock_comparison_data.clear()

if st.session_state.get('comparison_loaded', False):
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    comparison_data = []
    
    for i, ticker in enumerate(WATCHLIST_STOCKS):
        status_text.text(f"Loading {ticker}...")
        progress_bar.progress((i + 1) / len(WATCHLIST_STOCKS))
        
        result = get_stock_comparison_data(ticker)
        if result:
            comparison_data.append(result)
    
    progress_bar.empty()
    status_text.empty()
    
    if comparison_data:
        # Calculate conviction scores for all stocks
        conviction_data = []
        for stock in comparison_data:
            conviction = calculate_conviction_score(stock)
            conviction['current_price'] = stock.get('current_price')
            conviction_data.append(conviction)
        
        # --- Conviction Engine Summary Table ---
        st.markdown("### 🎯 Conviction Engine Summary")
        st.caption("Weighted scenario analysis with time-alignment and risk profiling")
        
        # Hurdle rate input
        hurdle_col1, hurdle_col2 = st.columns([1, 5])
        with hurdle_col1:
            hurdle_rate = st.number_input("Hurdle Rate %", value=10.0, min_value=0.0, max_value=50.0, step=1.0, 
                                          help="Minimum acceptable CAGR (default 10% = index fund benchmark)")
        
        # Recalculate with custom hurdle rate if changed
        if hurdle_rate != 10.0:
            conviction_data = []
            for stock in comparison_data:
                conviction = calculate_conviction_score(stock, hurdle_rate=hurdle_rate)
                conviction['current_price'] = stock.get('current_price')
                conviction_data.append(conviction)
        
        # Build conviction summary table
        conviction_table = []
        for conv in conviction_data:
            row = {
                'Ticker': conv['ticker'],
                'Current': conv.get('current_price'),
                'Exp. 5Y Price': conv.get('expected_value_5y'),
                'Exp. CAGR': conv.get('expected_cagr'),
                'Conviction': conv.get('conviction_rating', 'N/A'),
                'Risk': conv.get('risk_profile', 'N/A'),
                'Alignment': conv.get('time_alignment', 'N/A'),
            }
            conviction_table.append(row)
        
        conv_df = pd.DataFrame(conviction_table)
        conv_df = conv_df.set_index('Ticker')
        
        # Column config for conviction table
        conv_column_config = {
            'Current': st.column_config.NumberColumn(format="$%.2f"),
            'Exp. 5Y Price': st.column_config.NumberColumn(format="$%.2f"),
            'Exp. CAGR': st.column_config.NumberColumn(format="%.1f%%"),
        }
        
        st.dataframe(conv_df, use_container_width=True, height=400, column_config=conv_column_config)
        
        # Show warnings for any stocks with issues
        warnings_to_show = []
        for conv in conviction_data:
            if conv.get('warnings'):
                for warning in conv['warnings']:
                    warnings_to_show.append(f"**{conv['ticker']}**: {warning}")
        
        if warnings_to_show:
            with st.expander("⚠️ Warnings & Flags", expanded=False):
                for warning in warnings_to_show:
                    st.markdown(f"- {warning}")
        
        st.markdown("---")
        
        # Legend for conviction table
        st.caption("**Conviction Ratings:** 🟢 Strong Buy (CAGR ≥ hurdle+10%) | 🟡 Buy (≥ hurdle) | 🟠 Hold (≥ hurdle-5%) | 🔴 Avoid (≥ 0%) | ⛔ Strong Avoid (< 0%)")
        st.caption("**Risk Profiles:** 🏦 Predictable (<10% spread) | 📊 Medium Vol (10-20%) | 📈 High Vol (20-30%) | 🎰 Speculative (>30%)")
        st.caption("**Time Alignment:** ✅ Aligned | 💎 Accumulation Zone | ⚠️ Value Trap/Bearish | ➖ Mixed")
        
        st.markdown("---")
        
        # --- Original Price Targets Table ---
        st.markdown("### 📊 5-Year Price Targets & CAGR by Scenario")
        st.caption("Share prices and CAGR shown are for Year 5 projections")
        
        # Keep raw numeric values for proper sorting
        table_data = []
        for stock in comparison_data:
            row = {
                'Ticker': stock['ticker'],
                'Current Price': stock['current_price'],
                '🐻 Price Low': stock.get('bear_price_low'),
                '🐻 Price High': stock.get('bear_price_high'),
                '🐻 CAGR Low': stock.get('bear_cagr_low'),
                '🐻 CAGR High': stock.get('bear_cagr_high'),
                '📊 Price Low': stock.get('base_price_low'),
                '📊 Price High': stock.get('base_price_high'),
                '📊 CAGR Low': stock.get('base_cagr_low'),
                '📊 CAGR High': stock.get('base_cagr_high'),
                '🐂 Price Low': stock.get('bull_price_low'),
                '🐂 Price High': stock.get('bull_price_high'),
                '🐂 CAGR Low': stock.get('bull_cagr_low'),
                '🐂 CAGR High': stock.get('bull_cagr_high'),
            }
            table_data.append(row)
        
        df = pd.DataFrame(table_data)
        df = df.set_index('Ticker')
        
        # Configure column formatting for display while keeping numeric values for sorting
        column_config = {
            'Current Price': st.column_config.NumberColumn(format="$%.2f"),
            '🐻 Price Low': st.column_config.NumberColumn(format="$%.2f"),
            '🐻 Price High': st.column_config.NumberColumn(format="$%.2f"),
            '🐻 CAGR Low': st.column_config.NumberColumn(format="%.0f%%"),
            '🐻 CAGR High': st.column_config.NumberColumn(format="%.0f%%"),
            '📊 Price Low': st.column_config.NumberColumn(format="$%.2f"),
            '📊 Price High': st.column_config.NumberColumn(format="$%.2f"),
            '📊 CAGR Low': st.column_config.NumberColumn(format="%.0f%%"),
            '📊 CAGR High': st.column_config.NumberColumn(format="%.0f%%"),
            '🐂 Price Low': st.column_config.NumberColumn(format="$%.2f"),
            '🐂 Price High': st.column_config.NumberColumn(format="$%.2f"),
            '🐂 CAGR Low': st.column_config.NumberColumn(format="%.0f%%"),
            '🐂 CAGR High': st.column_config.NumberColumn(format="%.0f%%"),
        }
        
        st.dataframe(df, use_container_width=True, height=600, column_config=column_config)
        
        st.markdown("---")
        st.caption("**Legend:** 🐻 Bear Case | 📊 Base Case | 🐂 Bull Case")
        st.caption("Projections use default growth/margin assumptions based on each stock's current financials.")
    else:
        st.warning("No data could be loaded for watchlist stocks.")
else:
    st.info("Click 'Load/Refresh Comparison Data' to generate the comparison table.")

# Footer
st.markdown("---")
st.caption("Data provided by Yahoo Finance via yfinance. This tool is for educational purposes only and does not constitute financial advice.")
