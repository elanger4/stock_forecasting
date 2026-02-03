import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(page_title="Cross-Asset Market Dashboard", layout="wide")

# Hide Streamlit default elements
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    [data-testid="stSidebarNav"] {display: none;}
</style>
""", unsafe_allow_html=True)

# --- Asset Category Configuration ---
ASSET_CATEGORIES = {
    "Debt": {
        "TLT": "20+ Year Treasury",
        "IEF": "7-10 Year Treasury",
        "SHY": "1-3 Year Treasury",
        "TIP": "TIPS (Inflation Protected)",
        "LQD": "Investment Grade Corp",
        "HYG": "High Yield Corp",
        "EMB": "Emerging Market Bonds",
        "MUB": "Municipal Bonds",
        "AGG": "US Aggregate Bond",
        "BND": "Total Bond Market",
    },
    "Region": {
        "SPY": "United States",
        "EFA": "Developed Markets (ex-US)",
        "EEM": "Emerging Markets",
        "VGK": "Europe",
        "EWJ": "Japan",
        "FXI": "China",
        "EWZ": "Brazil",
        "EWY": "South Korea",
        "EWT": "Taiwan",
        "INDA": "India",
        "EWG": "Germany",
        "EWU": "United Kingdom",
        "EWC": "Canada",
        "EWA": "Australia",
    },
    "Indices": {
        "SPY": "S&P 500",
        "QQQ": "Nasdaq 100",
        "DIA": "Dow Jones",
        "IWM": "Russell 2000",
        "MDY": "S&P MidCap 400",
        "XLK": "Technology",
        "XLF": "Financials",
        "XLE": "Energy",
        "XLV": "Healthcare",
        "XLI": "Industrials",
        "XLP": "Consumer Staples",
        "XLY": "Consumer Discretionary",
        "XLU": "Utilities",
        "XLB": "Materials",
        "XLRE": "Real Estate",
        "XLC": "Communication Services",
    },
    "Commodities": {
        "GLD": "Gold",
        "SLV": "Silver",
        "USO": "Oil (WTI)",
        "UNG": "Natural Gas",
        "PDBC": "Broad Commodities",
        "DBA": "Agriculture",
        "WEAT": "Wheat",
        "CORN": "Corn",
        "CPER": "Copper",
        "URA": "Uranium",
        "PALL": "Palladium",
        "PPLT": "Platinum",
    },
    "Industries": {
        "XBI": "Biotech",
        "SMH": "Semiconductors",
        "XHB": "Homebuilders",
        "XRT": "Retail",
        "KRE": "Regional Banks",
        "XOP": "Oil & Gas E&P",
        "ITB": "Home Construction",
        "HACK": "Cybersecurity",
        "ARKK": "Innovation",
        "IBB": "Biotech (Nasdaq)",
        "IYT": "Transportation",
        "XME": "Metals & Mining",
    },
    "Style": {
        "IWF": "Large Growth",
        "IWD": "Large Value",
        "IWO": "Small Growth",
        "IWN": "Small Value",
        "MTUM": "Momentum Factor",
        "QUAL": "Quality Factor",
        "VLUE": "Value Factor",
        "SIZE": "Size Factor",
        "USMV": "Min Volatility",
        "DVY": "Dividend Select",
        "VIG": "Dividend Growth",
        "SCHD": "Dividend Equity",
    },
}

# Custom CSS for the dashboard
st.markdown("""
<style>
    /* Compact layout */
    .block-container {
        padding: 1rem 1rem !important;
        max-width: 100% !important;
    }
    
    /* Table styling */
    .dataframe {
        font-size: 0.85rem !important;
    }
    
    /* Category headers */
    .category-header {
        font-size: 1.1rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
        padding: 0.3rem 0.5rem;
        background: linear-gradient(90deg, #1f77b4 0%, transparent 100%);
        border-radius: 4px;
    }
    
    /* Momentum colors */
    .momentum-high { color: #00ff00 !important; }
    .momentum-mid { color: #ffff00 !important; }
    .momentum-low { color: #ff4444 !important; }
    
    /* Mobile responsive */
    @media (max-width: 768px) {
        .block-container { padding: 0.5rem !important; }
        [data-testid="column"] { width: 100% !important; flex: 1 1 100% !important; }
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
    st.page_link("pages/2_Monte_Carlo.py", label="👉 Monte Carlo")
with nav_col4:
    st.markdown("**🌍 Cross-Asset Dashboard** *(current)*")
with nav_col5:
    st.page_link("pages/4_Personalities.py", label="👉 Personalities")

st.markdown("---")

st.title("🌍 Cross-Asset Market Dashboard")
st.caption("Multi-asset performance matrix with 10-year momentum rankings")


@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_asset_data(ticker: str, name: str) -> dict:
    """Fetch price data and calculate metrics for a single asset."""
    try:
        stock = yf.Ticker(ticker)
        
        # Get current price and 1-day change
        hist_1d = stock.history(period="5d")
        if hist_1d.empty or len(hist_1d) < 2:
            return None
        
        current_price = hist_1d['Close'].iloc[-1]
        prev_close = hist_1d['Close'].iloc[-2]
        ch_1d_pct = ((current_price / prev_close) - 1) * 100
        
        # Get 10-year data for momentum
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * 10)
        hist_10y = stock.history(start=start_date, end=end_date)
        
        # Calculate 10-year momentum (annualized return)
        if len(hist_10y) >= 252:  # At least 1 year of data
            start_price = hist_10y['Close'].iloc[0]
            end_price = hist_10y['Close'].iloc[-1]
            years = len(hist_10y) / 252  # Trading days per year
            total_return = (end_price / start_price) - 1
            annualized_return = ((1 + total_return) ** (1 / years)) - 1
            momentum_raw = annualized_return * 100  # As percentage
        else:
            momentum_raw = None
        
        return {
            'name': name,
            'ticker': ticker,
            'price': current_price,
            'ch_1d_pct': ch_1d_pct,
            'momentum_raw': momentum_raw,
        }
    except Exception as e:
        return None


def normalize_momentum(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize momentum within category to 0-100 percentile rank."""
    if 'momentum_raw' in df.columns and df['momentum_raw'].notna().any():
        # Percentile rank (0-100)
        df['momentum_10y'] = df['momentum_raw'].rank(pct=True) * 100
    else:
        df['momentum_10y'] = None
    return df


def color_momentum(val):
    """Color code momentum values."""
    if pd.isna(val):
        return ''
    if val >= 70:
        return 'background-color: #1a472a; color: #00ff00'  # Green
    elif val >= 40:
        return 'background-color: #4a4a00; color: #ffff00'  # Yellow
    else:
        return 'background-color: #4a1a1a; color: #ff6666'  # Red


def color_change(val):
    """Color code daily change."""
    if pd.isna(val):
        return ''
    if val > 0:
        return 'color: #00ff00'
    elif val < 0:
        return 'color: #ff6666'
    return ''


def format_table_with_styling(df: pd.DataFrame):
    """Format dataframe for display with row color coding based on 1-day change."""
    display_df = df.copy()
    
    # Keep raw change values for styling
    raw_change = display_df['ch_1d_pct'].copy()
    
    # Format columns
    display_df['price'] = display_df['price'].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "—")
    display_df['ch_1d_pct'] = display_df['ch_1d_pct'].apply(lambda x: f"{x:+.1f}%" if pd.notna(x) else "—")
    display_df['momentum_10y'] = display_df['momentum_10y'].apply(lambda x: f"{x:.0f}" if pd.notna(x) else "—")
    
    # Rename columns for display
    display_df = display_df.rename(columns={
        'name': 'Name',
        'ticker': 'Ticker',
        'price': 'Price',
        'ch_1d_pct': '1D %',
        'momentum_10y': '10Y Mom',
    })
    
    # Select and order columns
    display_df = display_df[['Name', 'Ticker', 'Price', '1D %', '10Y Mom']]
    
    # Create row styling function based on raw change values
    def style_row(row_idx):
        change_val = raw_change.iloc[row_idx] if row_idx < len(raw_change) else 0
        if pd.isna(change_val):
            return [''] * len(display_df.columns)
        elif change_val > 1.5:
            return ['background-color: #1a3d1a'] * len(display_df.columns)  # Strong green
        elif change_val > 0:
            return ['background-color: #0d2d0d'] * len(display_df.columns)  # Light green
        elif change_val < -1.5:
            return ['background-color: #3d1a1a'] * len(display_df.columns)  # Strong red
        elif change_val < 0:
            return ['background-color: #2d0d0d'] * len(display_df.columns)  # Light red
        else:
            return [''] * len(display_df.columns)
    
    # Apply row-wise styling
    def apply_row_colors(df_to_style):
        styles = pd.DataFrame('', index=df_to_style.index, columns=df_to_style.columns)
        for i, idx in enumerate(df_to_style.index):
            change_val = raw_change.iloc[i] if i < len(raw_change) else 0
            if pd.isna(change_val):
                continue
            elif change_val > 1.5:
                styles.loc[idx] = 'background-color: #1a4d1a; color: #90EE90'  # Strong green
            elif change_val > 0:
                styles.loc[idx] = 'background-color: #0d2d0d; color: #98FB98'  # Light green
            elif change_val < -1.5:
                styles.loc[idx] = 'background-color: #4d1a1a; color: #FFA07A'  # Strong red
            elif change_val < 0:
                styles.loc[idx] = 'background-color: #2d0d0d; color: #FFB6C1'  # Light red
        return styles
    
    styled_df = display_df.style.apply(lambda _: apply_row_colors(display_df), axis=None)
    
    return styled_df


@st.cache_data(ttl=300)
def fetch_category_data(category_name: str, tickers: dict) -> pd.DataFrame:
    """Fetch data for all tickers in a category."""
    results = []
    for ticker, name in tickers.items():
        data = fetch_asset_data(ticker, name)
        if data:
            results.append(data)
    
    if not results:
        return pd.DataFrame()
    
    df = pd.DataFrame(results)
    df = normalize_momentum(df)
    return df


# --- Main Dashboard ---

# Sorting options
sort_col1, sort_col2, sort_col3 = st.columns([1, 1, 4])
with sort_col1:
    sort_by = st.selectbox("Sort by", ["momentum_10y", "ch_1d_pct", "name"], index=0, 
                           format_func=lambda x: {"momentum_10y": "10Y Momentum", "ch_1d_pct": "1D Change", "name": "Name"}[x])
with sort_col2:
    sort_order = st.selectbox("Order", ["Descending", "Ascending"], index=0)

ascending = sort_order == "Ascending"

# Load all data with progress
with st.spinner("Loading cross-asset data..."):
    category_data = {}
    progress_bar = st.progress(0)
    categories = list(ASSET_CATEGORIES.keys())
    
    for i, category in enumerate(categories):
        category_data[category] = fetch_category_data(category, ASSET_CATEGORIES[category])
        progress_bar.progress((i + 1) / len(categories))
    
    progress_bar.empty()

# Display tables in a 3x2 grid
st.markdown("---")

# Row 1: Debt, Region, Indices
row1_cols = st.columns(3)

for idx, category in enumerate(["Debt", "Region", "Indices"]):
    with row1_cols[idx]:
        st.markdown(f"### {category}")
        df = category_data.get(category, pd.DataFrame())
        
        if not df.empty:
            # Sort
            if sort_by in df.columns:
                df = df.sort_values(sort_by, ascending=ascending, na_position='last')
                df = df.reset_index(drop=True)
            
            # Format for display with row coloring
            styled_df = format_table_with_styling(df)
            
            st.dataframe(
                styled_df,
                use_container_width=True,
                height=450,
                hide_index=True,
            )
        else:
            st.info(f"No data available for {category}")

# Row 2: Commodities, Industries, Style
row2_cols = st.columns(3)

for idx, category in enumerate(["Commodities", "Industries", "Style"]):
    with row2_cols[idx]:
        st.markdown(f"### {category}")
        df = category_data.get(category, pd.DataFrame())
        
        if not df.empty:
            # Sort
            if sort_by in df.columns:
                df = df.sort_values(sort_by, ascending=ascending, na_position='last')
                df = df.reset_index(drop=True)
            
            # Format for display with row coloring
            styled_df = format_table_with_styling(df)
            
            st.dataframe(
                styled_df,
                use_container_width=True,
                height=450,
                hide_index=True,
            )
        else:
            st.info(f"No data available for {category}")

# Footer with timestamp
st.markdown("---")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Data cached for 5 minutes")
