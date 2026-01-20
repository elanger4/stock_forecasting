import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import sys
import os

# Add parent directory to path to import from watchlists
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from watchlists import get_watchlist_names, get_watchlist_stocks, get_default_watchlist

st.set_page_config(page_title="Monte Carlo Simulation", layout="wide")

# Initialize watchlist selection in session state
if 'selected_watchlist' not in st.session_state:
    st.session_state.selected_watchlist = get_default_watchlist()

# Custom CSS for larger UI elements
st.markdown("""
<style>
    .stMetric > div {
        font-size: 1.2rem;
    }
    .stMetric label {
        font-size: 1rem !important;
    }
    .stMetric > div > div {
        font-size: 1.8rem !important;
    }
    
    /* ========== MOBILE RESPONSIVE STYLES ========== */
    @media (max-width: 768px) {
        .block-container { padding: 1rem 1rem !important; margin-left: 0 !important; }
        [data-testid="stAppViewContainer"] { padding-left: 0.5rem !important; }
        [data-testid="stMetric"] { padding-left: 0.25rem !important; }
        [data-testid="column"] { width: 100% !important; flex: 1 1 100% !important; min-width: 100% !important; padding-left: 0.5rem !important; }
        [data-testid="stMetricValue"] { font-size: 1.4rem !important; }
        [data-testid="stMetricLabel"] { font-size: 0.9rem !important; }
        h1 { font-size: 1.8rem !important; }
        h2 { font-size: 1.4rem !important; }
        h3 { font-size: 1.2rem !important; }
        [data-testid="stDataFrame"], [data-testid="stTable"], .stDataFrame { overflow-x: auto !important; -webkit-overflow-scrolling: touch; }
        .stTable table, [data-testid="stDataFrame"] table { font-size: 0.85rem !important; }
        .stTable th, .stTable td { padding: 6px 8px !important; font-size: 0.85rem !important; }
        .stButton button { min-height: 48px !important; padding: 12px 16px !important; font-size: 1rem !important; width: 100% !important; }
        .stNumberInput input { min-height: 44px !important; font-size: 1rem !important; }
        .stSelectbox > div > div { min-height: 44px !important; }
        .row-widget.stHorizontalBlock { flex-wrap: wrap !important; }
        [data-testid="stSidebar"] { min-width: 280px !important; }
        .streamlit-expanderHeader { font-size: 1rem !important; padding: 12px 8px !important; }
    }
    @media (max-width: 480px) {
        .block-container { padding: 0.5rem 0.25rem !important; }
        [data-testid="stMetricValue"] { font-size: 1.2rem !important; }
        h1 { font-size: 1.5rem !important; }
        h2 { font-size: 1.2rem !important; }
    }
</style>
""", unsafe_allow_html=True)

# --- Page Navigation ---
nav_col1, nav_col2, nav_col3, nav_col4, nav_col5 = st.columns([1, 1, 1, 1, 1])
with nav_col1:
    st.page_link("app.py", label="👉 Stock Analysis")
with nav_col2:
    st.page_link("pages/1_Watchlist_Comparison.py", label="👉 Watchlist")
with nav_col3:
    st.markdown("**🎲 Monte Carlo** *(current)*")
with nav_col4:
    st.page_link("pages/3_Cross_Asset_Dashboard.py", label="👉 Cross-Asset")
with nav_col5:
    st.page_link("pages/4_Magic_Screener.py", label="👉 Magic Screener")

st.markdown("---")

st.title("🎲 Monte Carlo Simulation")
st.caption("Run probabilistic simulations to understand the range of possible outcomes for a stock")

# --- Helper Functions ---

def fetch_stock_data(ticker: str) -> dict:
    """Fetch stock data for simulation."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        current_price = info.get('currentPrice') or info.get('regularMarketPrice')
        total_revenue = info.get('totalRevenue')
        net_income = info.get('netIncomeToCommon')
        shares_outstanding = info.get('sharesOutstanding')
        
        # Get income statement for historical data
        income_stmt = stock.income_stmt
        
        # Calculate current EPS
        current_eps = None
        if income_stmt is not None and not income_stmt.empty:
            if 'Basic EPS' in income_stmt.index:
                current_eps = income_stmt.loc['Basic EPS'].iloc[0]
            elif 'Diluted EPS' in income_stmt.index:
                current_eps = income_stmt.loc['Diluted EPS'].iloc[0]
        
        if current_eps is None and net_income and shares_outstanding:
            current_eps = net_income / shares_outstanding
        
        # Current margin
        current_margin = None
        if total_revenue and net_income and total_revenue > 0:
            current_margin = (net_income / total_revenue) * 100
        
        # Revenue growth
        revenue_growth = info.get('revenueGrowth')
        if revenue_growth:
            revenue_growth = revenue_growth * 100
        
        # P/E ratios for estimating P/E range
        trailing_pe = info.get('trailingPE')
        forward_pe = info.get('forwardPE')
        
        # Historical volatility from price data
        hist = stock.history(period="2y")
        if not hist.empty:
            returns = hist['Close'].pct_change().dropna()
            annual_volatility = returns.std() * np.sqrt(252) * 100  # Annualized volatility %
        else:
            annual_volatility = 30.0  # Default assumption
        
        # Calculate historical revenue growth volatility
        rev_growth_std = 5.0  # Default
        margin_std = 2.0  # Default
        
        if income_stmt is not None and not income_stmt.empty:
            if 'Total Revenue' in income_stmt.index:
                revenues = income_stmt.loc['Total Revenue'].dropna()
                if len(revenues) >= 2:
                    rev_changes = revenues.pct_change().dropna() * 100
                    if len(rev_changes) > 0:
                        rev_growth_std = max(abs(rev_changes.std()), 3.0)  # Min 3%
            
            if 'Net Income' in income_stmt.index and 'Total Revenue' in income_stmt.index:
                net_incomes = income_stmt.loc['Net Income'].dropna()
                revenues = income_stmt.loc['Total Revenue'].dropna()
                if len(net_incomes) >= 2 and len(revenues) >= 2:
                    margins = (net_incomes / revenues * 100).dropna()
                    if len(margins) > 1:
                        margin_std = max(abs(margins.std()), 1.0)  # Min 1%
        
        return {
            'success': True,
            'current_price': current_price,
            'total_revenue': total_revenue,
            'net_income': net_income,
            'shares_outstanding': shares_outstanding,
            'current_eps': current_eps,
            'current_margin': current_margin,
            'revenue_growth': revenue_growth if revenue_growth else 10.0,
            'trailing_pe': trailing_pe,
            'forward_pe': forward_pe,
            'annual_volatility': annual_volatility,
            'rev_growth_std': rev_growth_std,
            'margin_std': margin_std,
            'company_name': info.get('shortName', ticker),
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}


def run_monte_carlo(
    current_price: float,
    current_revenue: float,
    current_margin: float,
    current_eps: float,
    shares: float,
    # Growth parameters
    growth_mean: float,
    growth_std: float,
    # Margin parameters
    margin_mean: float,
    margin_std: float,
    # P/E parameters
    pe_mean: float,
    pe_std: float,
    # Simulation parameters
    num_simulations: int,
    num_years: int,
    distribution: str,
    # Correlation
    growth_margin_corr: float = 0.3,
    # Growth decay and mean reversion
    growth_decay_rate: float = 0.15,  # Growth decays 15% per year toward terminal
    terminal_growth_rate: float = 5.0,  # Long-term sustainable growth rate
    pe_mean_reversion: float = 0.1,  # P/E reverts 10% per year toward market average
    market_pe: float = 20.0,  # Long-term market average P/E
) -> dict:
    """
    Run Monte Carlo simulation with growth decay and mean reversion.
    
    Growth Decay: High growth rates decay toward a terminal rate over time.
    Mean Reversion: P/E multiples gradually revert toward market average.
    
    Returns dict with:
    - final_prices: array of final year prices
    - yearly_prices: 2D array (simulations x years)
    - yearly_eps: 2D array
    - statistics: dict of summary stats
    """
    np.random.seed(42)  # For reproducibility
    
    # Initialize arrays
    yearly_prices = np.zeros((num_simulations, num_years + 1))
    yearly_eps = np.zeros((num_simulations, num_years + 1))
    yearly_revenue = np.zeros((num_simulations, num_years + 1))
    
    # Year 0 is current state
    yearly_prices[:, 0] = current_price
    yearly_eps[:, 0] = current_eps if current_eps else 0
    yearly_revenue[:, 0] = current_revenue
    
    for sim in range(num_simulations):
        revenue = current_revenue
        
        # Track decaying growth rate for this simulation
        current_growth_mean = growth_mean
        current_pe_mean = pe_mean
        
        for year in range(1, num_years + 1):
            # Apply growth decay: growth rate decays toward terminal rate each year
            # Formula: new_growth = terminal + (current - terminal) * (1 - decay_rate)
            current_growth_mean = terminal_growth_rate + (current_growth_mean - terminal_growth_rate) * (1 - growth_decay_rate)
            
            # Apply P/E mean reversion: P/E reverts toward market average each year
            current_pe_mean = market_pe + (current_pe_mean - market_pe) * (1 - pe_mean_reversion)
            
            # Adjust std dev proportionally as growth decays
            adjusted_growth_std = growth_std * (current_growth_mean / max(growth_mean, 1)) if growth_mean > 0 else growth_std * 0.5
            adjusted_growth_std = max(adjusted_growth_std, 2.0)  # Minimum 2% std dev
            
            # Generate random values based on distribution
            if distribution == "Normal":
                growth = np.random.normal(current_growth_mean, adjusted_growth_std)
                margin = np.random.normal(margin_mean, margin_std)
                pe = np.random.normal(current_pe_mean, pe_std)
            elif distribution == "Log-Normal":
                # Convert to log-normal parameters
                safe_growth_mean = max(current_growth_mean + 100, 1)
                growth = np.random.lognormal(
                    np.log(safe_growth_mean) - 0.5 * (adjusted_growth_std/100)**2,
                    adjusted_growth_std / 100
                ) - 100
                margin = max(-50, np.random.normal(margin_mean, margin_std))
                safe_pe_mean = max(current_pe_mean, 1)
                pe = np.random.lognormal(
                    np.log(safe_pe_mean) - 0.5 * (pe_std/safe_pe_mean)**2,
                    pe_std / safe_pe_mean
                )
            elif distribution == "Triangular":
                # Use mean as mode, std to define range
                growth = np.random.triangular(
                    current_growth_mean - 2*adjusted_growth_std,
                    current_growth_mean,
                    current_growth_mean + 2*adjusted_growth_std
                )
                margin = np.random.triangular(
                    max(-50, margin_mean - 2*margin_std),
                    margin_mean,
                    min(50, margin_mean + 2*margin_std)
                )
                pe = np.random.triangular(
                    max(1, current_pe_mean - 2*pe_std),
                    current_pe_mean,
                    current_pe_mean + 2*pe_std
                )
            else:  # Uniform
                growth = np.random.uniform(current_growth_mean - 2*adjusted_growth_std, current_growth_mean + 2*adjusted_growth_std)
                margin = np.random.uniform(max(-50, margin_mean - 2*margin_std), min(50, margin_mean + 2*margin_std))
                pe = np.random.uniform(max(1, current_pe_mean - 2*pe_std), current_pe_mean + 2*pe_std)
            
            # Apply correlation between growth and margin (higher growth often = higher margins)
            if growth_margin_corr > 0 and adjusted_growth_std > 0:
                margin_adjustment = (growth - current_growth_mean) / adjusted_growth_std * margin_std * growth_margin_corr
                margin = margin + margin_adjustment
            
            # Ensure reasonable bounds
            margin = max(-50, min(50, margin))  # Cap margin at -50% to 50%
            pe = max(1, min(60, pe))  # Cap P/E at 1 to 60 (more realistic)
            
            # Calculate projections
            revenue = revenue * (1 + growth / 100)
            net_income = revenue * (margin / 100)
            eps = net_income / shares if shares else 0
            
            # Price based on EPS and P/E
            if eps > 0:
                price = eps * pe
            else:
                # For negative EPS, use P/S ratio approach
                ps_ratio = current_price / (current_revenue / shares) if current_revenue and shares else 2
                price = (revenue / shares) * ps_ratio * 0.9  # Slight discount for negative earnings
            
            price = max(0.01, price)  # Floor at $0.01
            
            yearly_prices[sim, year] = price
            yearly_eps[sim, year] = eps
            yearly_revenue[sim, year] = revenue
    
    # Calculate statistics
    final_prices = yearly_prices[:, -1]
    
    # CAGR for each simulation
    cagrs = ((final_prices / current_price) ** (1 / num_years) - 1) * 100
    
    statistics = {
        'mean_price': np.mean(final_prices),
        'median_price': np.median(final_prices),
        'std_price': np.std(final_prices),
        'min_price': np.min(final_prices),
        'max_price': np.max(final_prices),
        'percentile_10': np.percentile(final_prices, 10),
        'percentile_25': np.percentile(final_prices, 25),
        'percentile_75': np.percentile(final_prices, 75),
        'percentile_90': np.percentile(final_prices, 90),
        'prob_profit': np.mean(final_prices > current_price) * 100,
        'prob_double': np.mean(final_prices > current_price * 2) * 100,
        'prob_loss_20': np.mean(final_prices < current_price * 0.8) * 100,
        'prob_loss_50': np.mean(final_prices < current_price * 0.5) * 100,
        'mean_cagr': np.mean(cagrs),
        'median_cagr': np.median(cagrs),
        'var_95': np.percentile(final_prices, 5),  # 5th percentile = 95% VaR
        'var_99': np.percentile(final_prices, 1),  # 1st percentile = 99% VaR
    }
    
    return {
        'final_prices': final_prices,
        'yearly_prices': yearly_prices,
        'yearly_eps': yearly_eps,
        'yearly_revenue': yearly_revenue,
        'cagrs': cagrs,
        'statistics': statistics,
    }


# --- Stock Selection ---
st.subheader("📈 Select Stock")

stock_col1, stock_col2, stock_col3 = st.columns([1, 1, 2])

with stock_col1:
    # Watchlist selector
    watchlist_names = get_watchlist_names()
    current_idx = watchlist_names.index(st.session_state.selected_watchlist) if st.session_state.selected_watchlist in watchlist_names else 0
    selected_watchlist = st.selectbox(
        "Watchlist",
        options=watchlist_names,
        index=current_idx,
        key="mc_watchlist_selector"
    )
    if selected_watchlist != st.session_state.selected_watchlist:
        st.session_state.selected_watchlist = selected_watchlist

with stock_col2:
    # Stock dropdown from watchlist
    watchlist_stocks = get_watchlist_stocks(st.session_state.selected_watchlist)
    selected_stock = st.selectbox(
        "Stock",
        options=[""] + watchlist_stocks,
        key="mc_stock_selector",
        help="Select a stock from your watchlist or enter a custom ticker"
    )

with stock_col3:
    # Custom ticker input
    custom_ticker = st.text_input(
        "Or enter custom ticker",
        value="",
        key="mc_custom_ticker",
        help="Enter any ticker symbol"
    )

# Determine which ticker to use
ticker = custom_ticker.upper().strip() if custom_ticker else selected_stock

# Fetch data button
fetch_col1, fetch_col2 = st.columns([1, 5])
with fetch_col1:
    fetch_data = st.button("📥 Fetch Data", key="mc_fetch_data", use_container_width=True)

if fetch_data and ticker:
    with st.spinner(f"Fetching data for {ticker}..."):
        st.session_state.mc_stock_data = fetch_stock_data(ticker)
        st.session_state.mc_ticker = ticker

# --- Display stock data and simulation parameters ---
if 'mc_stock_data' in st.session_state and st.session_state.mc_stock_data.get('success'):
    data = st.session_state.mc_stock_data
    ticker = st.session_state.get('mc_ticker', 'Unknown')
    
    st.markdown("---")
    st.subheader(f"📊 {data['company_name']} ({ticker})")
    
    # Current metrics
    metric_cols = st.columns(5)
    with metric_cols[0]:
        st.metric("Current Price", f"${data['current_price']:.2f}" if data['current_price'] else "N/A")
    with metric_cols[1]:
        st.metric("Current EPS", f"${data['current_eps']:.2f}" if data['current_eps'] else "N/A")
    with metric_cols[2]:
        st.metric("Revenue Growth", f"{data['revenue_growth']:.1f}%" if data['revenue_growth'] else "N/A")
    with metric_cols[3]:
        st.metric("Net Margin", f"{data['current_margin']:.1f}%" if data['current_margin'] else "N/A")
    with metric_cols[4]:
        st.metric("Volatility", f"{data['annual_volatility']:.1f}%" if data['annual_volatility'] else "N/A",
                 help="Annualized price volatility based on 2-year historical data")
    
    st.markdown("---")
    
    # --- Simulation Parameters ---
    st.subheader("⚙️ Simulation Parameters")
    
    # Simulation settings
    sim_col1, sim_col2, sim_col3 = st.columns(3)
    
    with sim_col1:
        num_simulations = st.number_input(
            "Number of Simulations",
            min_value=100,
            max_value=50000,
            value=1000,
            step=100,
            help="More simulations = more accurate results but slower. 1,000-10,000 recommended."
        )
    
    with sim_col2:
        num_years = st.number_input(
            "Projection Years",
            min_value=1,
            max_value=10,
            value=5,
            help="Number of years to project forward"
        )
    
    with sim_col3:
        distribution = st.selectbox(
            "Distribution Type",
            options=["Normal", "Log-Normal", "Triangular", "Uniform"],
            index=0,
            help="""
            **Normal**: Bell curve, symmetric outcomes (default)
            **Log-Normal**: Skewed, prevents negative values, models growth well
            **Triangular**: Uses min/mode/max, intuitive for scenario planning
            **Uniform**: Equal probability across range
            """
        )
    
    st.markdown("---")
    
    # Variable parameters in expandable sections
    st.markdown("### 📈 Variable Parameters")
    st.caption("Adjust the mean and standard deviation for each variable. Defaults are based on current/historical data.")
    
    # Revenue Growth Parameters
    with st.expander("📊 Revenue Growth Parameters", expanded=True):
        growth_col1, growth_col2 = st.columns(2)
        # Cap the default value to a more realistic range for simulation
        # Very high growth rates (>50%) are unsustainable and will decay anyway
        raw_growth = float(data['revenue_growth']) if data['revenue_growth'] else 10.0
        default_growth = min(max(raw_growth, -50.0), 50.0)  # Cap at 50% for more realistic defaults
        
        # Show warning if we capped the growth rate
        if raw_growth > 50:
            st.warning(f"⚠️ Current growth rate ({raw_growth:.1f}%) is very high. Defaulting to 50% with decay toward 5% terminal rate. Adjust in Advanced Settings if needed.")
        
        with growth_col1:
            growth_mean = st.number_input(
                "Mean Revenue Growth (%)",
                min_value=-50.0,
                max_value=200.0,
                value=default_growth,
                step=1.0,
                help="Expected average annual revenue growth rate. High rates will decay toward terminal rate over time."
            )
        with growth_col2:
            default_growth_std = float(data['rev_growth_std']) if data['rev_growth_std'] else 5.0
            default_growth_std = min(max(default_growth_std, 0.1), 100.0)  # Clamp to valid range
            
            growth_std = st.number_input(
                "Std Dev Revenue Growth (%)",
                min_value=0.1,
                max_value=100.0,
                value=default_growth_std,
                step=0.5,
                help="Uncertainty in growth rate. Higher = more variable outcomes."
            )
    
    # Net Margin Parameters
    with st.expander("💰 Net Margin Parameters", expanded=True):
        margin_col1, margin_col2 = st.columns(2)
        # Clamp default margin to valid range
        default_margin = float(data['current_margin']) if data['current_margin'] else 10.0
        default_margin = min(max(default_margin, -100.0), 100.0)
        
        with margin_col1:
            margin_mean = st.number_input(
                "Mean Net Margin (%)",
                min_value=-100.0,
                max_value=100.0,
                value=default_margin,
                step=1.0,
                help="Expected average net income margin"
            )
        with margin_col2:
            default_margin_std = float(data['margin_std']) if data['margin_std'] else 2.0
            default_margin_std = min(max(default_margin_std, 0.1), 50.0)  # Clamp to valid range
            
            margin_std = st.number_input(
                "Std Dev Net Margin (%)",
                min_value=0.1,
                max_value=50.0,
                value=default_margin_std,
                step=0.5,
                help="Uncertainty in margin. Higher = more variable outcomes."
            )
    
    # P/E Multiple Parameters
    with st.expander("📉 P/E Multiple Parameters", expanded=True):
        # Calculate default P/E with clamping
        default_pe = data.get('trailing_pe') or data.get('forward_pe') or 20.0
        if default_pe and default_pe > 0:
            default_pe = min(max(float(default_pe), 1.0), 200.0)  # Clamp to valid range
        else:
            default_pe = 20.0
        
        pe_col1, pe_col2 = st.columns(2)
        with pe_col1:
            pe_mean = st.number_input(
                "Mean P/E Multiple",
                min_value=1.0,
                max_value=200.0,
                value=float(default_pe),
                step=1.0,
                help="Expected average P/E ratio for valuation"
            )
        with pe_col2:
            default_pe_std = min(float(default_pe) * 0.2, 50.0)  # 20% of mean, max 50
            default_pe_std = max(default_pe_std, 0.1)  # Ensure minimum
            
            pe_std = st.number_input(
                "Std Dev P/E Multiple",
                min_value=0.1,
                max_value=50.0,
                value=default_pe_std,
                step=0.5,
                help="Uncertainty in P/E multiple. Higher = more valuation variability."
            )
    
    # Advanced settings
    with st.expander("🔧 Advanced Settings (Growth Decay & Mean Reversion)", expanded=False):
        st.markdown("#### Growth Decay")
        st.caption("High growth rates naturally decay toward a sustainable terminal rate over time.")
        
        decay_col1, decay_col2 = st.columns(2)
        with decay_col1:
            growth_decay_rate = st.slider(
                "Growth Decay Rate (%/year)",
                min_value=0.0,
                max_value=50.0,
                value=15.0,
                step=5.0,
                help="How fast growth decays toward terminal rate. 15% means growth loses 15% of its excess over terminal each year."
            ) / 100
        with decay_col2:
            terminal_growth_rate = st.number_input(
                "Terminal Growth Rate (%)",
                min_value=0.0,
                max_value=20.0,
                value=5.0,
                step=1.0,
                help="Long-term sustainable growth rate that high growth decays toward. Typically 3-7% for mature companies."
            )
        
        st.markdown("#### P/E Mean Reversion")
        st.caption("P/E multiples tend to revert toward market averages over time.")
        
        pe_col1, pe_col2 = st.columns(2)
        with pe_col1:
            pe_mean_reversion = st.slider(
                "P/E Reversion Rate (%/year)",
                min_value=0.0,
                max_value=50.0,
                value=10.0,
                step=5.0,
                help="How fast P/E reverts toward market average. 10% means P/E loses 10% of its premium/discount each year."
            ) / 100
        with pe_col2:
            market_pe = st.number_input(
                "Market Average P/E",
                min_value=5.0,
                max_value=40.0,
                value=20.0,
                step=1.0,
                help="Long-term market average P/E that multiples revert toward. S&P 500 historical average is ~15-20."
            )
        
        st.markdown("#### Correlation")
        growth_margin_corr = st.slider(
            "Growth-Margin Correlation",
            min_value=-1.0,
            max_value=1.0,
            value=0.3,
            step=0.1,
            help="Correlation between revenue growth and margin. Positive means high growth tends to come with higher margins."
        )
    
    st.markdown("---")
    
    # Run Simulation Button
    run_col1, run_col2 = st.columns([1, 5])
    with run_col1:
        run_simulation = st.button("🎲 Run Simulation", key="run_mc_simulation", use_container_width=True, type="primary")
    
    if run_simulation:
        with st.spinner(f"Running {num_simulations:,} simulations..."):
            results = run_monte_carlo(
                current_price=data['current_price'],
                current_revenue=data['total_revenue'],
                current_margin=data['current_margin'] or 10.0,
                current_eps=data['current_eps'] or 1.0,
                shares=data['shares_outstanding'],
                growth_mean=growth_mean,
                growth_std=growth_std,
                margin_mean=margin_mean,
                margin_std=margin_std,
                pe_mean=pe_mean,
                pe_std=pe_std,
                num_simulations=int(num_simulations),
                num_years=int(num_years),
                distribution=distribution,
                growth_margin_corr=growth_margin_corr,
                growth_decay_rate=growth_decay_rate,
                terminal_growth_rate=terminal_growth_rate,
                pe_mean_reversion=pe_mean_reversion,
                market_pe=market_pe,
            )
            st.session_state.mc_results = results
            st.session_state.mc_params = {
                'num_simulations': num_simulations,
                'num_years': num_years,
                'distribution': distribution,
                'current_price': data['current_price'],
            }
    
    # --- Display Results ---
    if 'mc_results' in st.session_state:
        results = st.session_state.mc_results
        params = st.session_state.mc_params
        stats = results['statistics']
        current_price = params['current_price']
        
        st.markdown("---")
        st.header("📊 Simulation Results")
        st.caption(f"Based on {params['num_simulations']:,} simulations over {params['num_years']} years using {params['distribution']} distribution")
        
        # --- Key Metrics ---
        st.subheader("🎯 Key Metrics")
        
        key_cols = st.columns(4)
        with key_cols[0]:
            st.metric(
                "Expected Price (Mean)",
                f"${stats['mean_price']:.2f}",
                delta=f"{((stats['mean_price']/current_price)-1)*100:+.1f}%",
                help="Average of all simulated final prices"
            )
        with key_cols[1]:
            st.metric(
                "Median Price",
                f"${stats['median_price']:.2f}",
                delta=f"{((stats['median_price']/current_price)-1)*100:+.1f}%",
                help="Middle value of all simulated prices. Less affected by outliers than mean."
            )
        with key_cols[2]:
            st.metric(
                "Expected CAGR",
                f"{stats['mean_cagr']:.1f}%",
                help="Average compound annual growth rate across all simulations"
            )
        with key_cols[3]:
            st.metric(
                "Median CAGR",
                f"{stats['median_cagr']:.1f}%",
                help="Middle CAGR value. More robust than mean for skewed distributions."
            )
        
        st.markdown("---")
        
        # --- Probability Metrics ---
        st.subheader("📈 Probability Analysis")
        
        prob_cols = st.columns(4)
        with prob_cols[0]:
            st.metric(
                "Probability of Profit",
                f"{stats['prob_profit']:.1f}%",
                help="Chance that the stock price will be higher than today"
            )
        with prob_cols[1]:
            st.metric(
                "Probability of 2x",
                f"{stats['prob_double']:.1f}%",
                help="Chance that the stock will double in value"
            )
        with prob_cols[2]:
            st.metric(
                "Probability of -20%",
                f"{stats['prob_loss_20']:.1f}%",
                help="Chance of losing 20% or more"
            )
        with prob_cols[3]:
            st.metric(
                "Probability of -50%",
                f"{stats['prob_loss_50']:.1f}%",
                help="Chance of losing 50% or more"
            )
        
        st.markdown("---")
        
        # --- Confidence Intervals ---
        st.subheader("📊 Confidence Intervals")
        
        ci_cols = st.columns(4)
        with ci_cols[0]:
            st.metric(
                "10th Percentile",
                f"${stats['percentile_10']:.2f}",
                delta=f"{((stats['percentile_10']/current_price)-1)*100:+.1f}%",
                help="90% of outcomes are above this price (bearish scenario)"
            )
        with ci_cols[1]:
            st.metric(
                "25th Percentile",
                f"${stats['percentile_25']:.2f}",
                delta=f"{((stats['percentile_25']/current_price)-1)*100:+.1f}%",
                help="75% of outcomes are above this price"
            )
        with ci_cols[2]:
            st.metric(
                "75th Percentile",
                f"${stats['percentile_75']:.2f}",
                delta=f"{((stats['percentile_75']/current_price)-1)*100:+.1f}%",
                help="25% of outcomes are above this price"
            )
        with ci_cols[3]:
            st.metric(
                "90th Percentile",
                f"${stats['percentile_90']:.2f}",
                delta=f"{((stats['percentile_90']/current_price)-1)*100:+.1f}%",
                help="10% of outcomes are above this price (bullish scenario)"
            )
        
        # 80% confidence interval callout
        st.info(f"📊 **80% Confidence Interval**: There's an 80% probability the price will be between **${stats['percentile_10']:.2f}** and **${stats['percentile_90']:.2f}** in {params['num_years']} years.")
        
        st.markdown("---")
        
        # --- Risk Metrics ---
        st.subheader("⚠️ Risk Metrics (Value at Risk)")
        
        var_cols = st.columns(3)
        with var_cols[0]:
            st.metric(
                "VaR 95%",
                f"${stats['var_95']:.2f}",
                delta=f"{((stats['var_95']/current_price)-1)*100:+.1f}%",
                help="Value at Risk: 95% of the time, the price will be above this level. Only 5% chance of being lower."
            )
        with var_cols[1]:
            st.metric(
                "VaR 99%",
                f"${stats['var_99']:.2f}",
                delta=f"{((stats['var_99']/current_price)-1)*100:+.1f}%",
                help="Value at Risk: 99% of the time, the price will be above this level. Only 1% chance of being lower."
            )
        with var_cols[2]:
            max_loss = ((stats['var_95'] / current_price) - 1) * 100
            st.metric(
                "Max Expected Loss (95%)",
                f"{max_loss:.1f}%",
                help="The maximum loss you should expect 95% of the time"
            )
        
        st.markdown("---")
        
        # --- Visualizations ---
        st.subheader("📈 Visualizations")
        
        viz_tab1, viz_tab2, viz_tab3 = st.tabs(["📊 Price Distribution", "📈 Price Paths (Fan Chart)", "📉 Cumulative Probability"])
        
        with viz_tab1:
            st.markdown("#### Final Price Distribution")
            st.caption("Histogram showing the distribution of simulated final prices")
            
            import plotly.express as px
            import plotly.graph_objects as go
            
            fig = px.histogram(
                x=results['final_prices'],
                nbins=50,
                labels={'x': f'Price after {params["num_years"]} Years', 'y': 'Frequency'},
                title=f'Distribution of {params["num_simulations"]:,} Simulated Prices'
            )
            
            # Add vertical lines for key values
            fig.add_vline(x=current_price, line_dash="dash", line_color="red", 
                         annotation_text=f"Current: ${current_price:.2f}")
            fig.add_vline(x=stats['median_price'], line_dash="dash", line_color="green",
                         annotation_text=f"Median: ${stats['median_price']:.2f}")
            fig.add_vline(x=stats['percentile_10'], line_dash="dot", line_color="orange",
                         annotation_text=f"10th %ile: ${stats['percentile_10']:.2f}")
            fig.add_vline(x=stats['percentile_90'], line_dash="dot", line_color="blue",
                         annotation_text=f"90th %ile: ${stats['percentile_90']:.2f}")
            
            fig.update_layout(height=500, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with viz_tab2:
            st.markdown("#### Price Paths Over Time (Fan Chart)")
            st.caption("Shows the range of possible price trajectories with confidence bands")
            
            import plotly.graph_objects as go
            
            yearly_prices = results['yearly_prices']
            years = list(range(params['num_years'] + 1))
            current_year = datetime.now().year
            year_labels = [str(current_year + y) for y in years]
            
            # Calculate percentiles for each year
            p10 = np.percentile(yearly_prices, 10, axis=0)
            p25 = np.percentile(yearly_prices, 25, axis=0)
            p50 = np.percentile(yearly_prices, 50, axis=0)
            p75 = np.percentile(yearly_prices, 75, axis=0)
            p90 = np.percentile(yearly_prices, 90, axis=0)
            
            fig = go.Figure()
            
            # Add confidence bands
            fig.add_trace(go.Scatter(
                x=year_labels + year_labels[::-1],
                y=list(p90) + list(p10)[::-1],
                fill='toself',
                fillcolor='rgba(0, 100, 255, 0.1)',
                line=dict(color='rgba(255,255,255,0)'),
                name='10-90% Range',
                showlegend=True
            ))
            
            fig.add_trace(go.Scatter(
                x=year_labels + year_labels[::-1],
                y=list(p75) + list(p25)[::-1],
                fill='toself',
                fillcolor='rgba(0, 100, 255, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='25-75% Range',
                showlegend=True
            ))
            
            # Add median line
            fig.add_trace(go.Scatter(
                x=year_labels,
                y=p50,
                mode='lines+markers',
                name='Median',
                line=dict(color='blue', width=3)
            ))
            
            # Add current price line
            fig.add_trace(go.Scatter(
                x=year_labels,
                y=[current_price] * len(year_labels),
                mode='lines',
                name='Current Price',
                line=dict(color='red', width=2, dash='dash')
            ))
            
            fig.update_layout(
                title='Projected Price Range Over Time',
                xaxis_title='Year',
                yaxis_title='Price ($)',
                height=500,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with viz_tab3:
            st.markdown("#### Cumulative Probability Distribution")
            st.caption("Shows the probability of achieving at least a given price")
            
            import plotly.graph_objects as go
            
            sorted_prices = np.sort(results['final_prices'])
            cumulative_prob = np.arange(1, len(sorted_prices) + 1) / len(sorted_prices) * 100
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=sorted_prices,
                y=100 - cumulative_prob,  # Probability of exceeding
                mode='lines',
                name='P(Price > X)',
                line=dict(color='blue', width=2)
            ))
            
            # Add reference lines
            fig.add_hline(y=50, line_dash="dash", line_color="gray", 
                         annotation_text="50% probability")
            fig.add_vline(x=current_price, line_dash="dash", line_color="red",
                         annotation_text=f"Current: ${current_price:.2f}")
            fig.add_vline(x=current_price * 2, line_dash="dot", line_color="green",
                         annotation_text=f"2x: ${current_price*2:.2f}")
            
            fig.update_layout(
                title='Probability of Exceeding Price Target',
                xaxis_title=f'Price after {params["num_years"]} Years ($)',
                yaxis_title='Probability (%)',
                height=500,
                yaxis=dict(range=[0, 100])
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # --- Year-by-Year Table ---
        st.subheader("📅 Year-by-Year Projections")
        
        yearly_prices = results['yearly_prices']
        current_year = datetime.now().year
        
        yearly_data = []
        for year in range(params['num_years'] + 1):
            year_prices = yearly_prices[:, year]
            yearly_data.append({
                'Year': current_year + year,
                'Median Price': f"${np.median(year_prices):.2f}",
                'Mean Price': f"${np.mean(year_prices):.2f}",
                '10th %ile': f"${np.percentile(year_prices, 10):.2f}",
                '90th %ile': f"${np.percentile(year_prices, 90):.2f}",
                'Prob > Current': f"{np.mean(year_prices > current_price) * 100:.1f}%",
            })
        
        yearly_df = pd.DataFrame(yearly_data)
        yearly_df = yearly_df.set_index('Year')
        st.dataframe(yearly_df, use_container_width=True)

elif 'mc_stock_data' in st.session_state and not st.session_state.mc_stock_data.get('success'):
    st.error(f"❌ Error fetching data: {st.session_state.mc_stock_data.get('error')}")
else:
    st.info("👆 Select a stock from your watchlist or enter a ticker, then click 'Fetch Data' to begin.")

# --- Footer ---
st.markdown("---")

# Help/Documentation section
footer_col1, footer_col2 = st.columns([1, 5])
with footer_col1:
    show_help = st.button("ℹ️ Help & Documentation", key="show_mc_help_docs", use_container_width=True)

if show_help:
    st.session_state.show_mc_help_dialog = True

if st.session_state.get('show_mc_help_dialog', False):
    @st.dialog("📖 Monte Carlo Simulation - Documentation", width="large")
    def show_mc_help_dialog():
        doc_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "MONTE_CARLO.md")
        try:
            with open(doc_path, "r") as f:
                help_content = f.read()
            st.markdown(help_content)
        except FileNotFoundError:
            st.markdown("""
# Monte Carlo Simulation Help

## What is Monte Carlo Simulation?

Monte Carlo simulation is a technique that uses random sampling to understand the range of possible outcomes for an investment. Instead of producing a single forecast, it generates thousands of possible scenarios based on the uncertainty in your assumptions.

## How to Use

1. **Select a stock** from your watchlist or enter a custom ticker
2. **Adjust parameters** for revenue growth, margins, and P/E multiples
3. **Run the simulation** to see the distribution of possible outcomes

## Interpreting Results

- **Mean/Median Price**: Expected outcomes
- **Confidence Intervals**: Range where prices are likely to fall
- **Probability Metrics**: Chances of profit, doubling, or loss
- **VaR (Value at Risk)**: Worst-case scenarios

## Distribution Types

- **Normal**: Symmetric bell curve (default)
- **Log-Normal**: Skewed, prevents negative values
- **Triangular**: Uses min/mode/max
- **Uniform**: Equal probability across range
            """)
        
        if st.button("Close", key="close_mc_help"):
            st.session_state.show_mc_help_dialog = False
            st.rerun()
    
    show_mc_help_dialog()

st.caption("Monte Carlo simulation is for educational purposes only. Past performance and simulated results do not guarantee future returns.")
