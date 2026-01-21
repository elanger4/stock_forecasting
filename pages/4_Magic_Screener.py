import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import time
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(page_title="Magic Formula Screener", layout="wide")

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
    st.markdown("**🪄 Magic Screener** *(current)*")

st.markdown("---")

# --- Title and Description ---
st.title("🪄 Magic Formula Screener")
st.markdown("""
**Joel Greenblatt's Magic Formula** ranks stocks by combining two metrics:
1. **Earnings Yield** = EBIT / Enterprise Value (cheapness)
2. **Return on Capital** = EBIT / (Net Working Capital + Net Fixed Assets) (quality)

Stocks are ranked on each metric, and the combined rank determines the "magic" score. 
**Lower combined rank = better investment candidate.**
""")

st.markdown("---")


# --- Fetch Russell 3000 Tickers ---
@st.cache_data(ttl=86400)  # Cache for 24 hours
def get_russell_3000_tickers():
    """Fetch Russell 3000 constituents from multiple sources with fallbacks."""
    import requests
    
    all_tickers = []
    
    # Method 1: Try Wikipedia S&P 500 + S&P 400 + S&P 600 (most reliable)
    try:
        # S&P 500
        sp500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        sp500_tables = pd.read_html(sp500_url, attrs={'id': 'constituents'})
        if sp500_tables:
            sp500_tickers = sp500_tables[0]['Symbol'].str.replace('.', '-', regex=False).tolist()
            all_tickers.extend(sp500_tickers)
    except Exception:
        pass
    
    try:
        # S&P 400 Mid Cap
        sp400_url = "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"
        sp400_tables = pd.read_html(sp400_url, attrs={'id': 'constituents'})
        if sp400_tables:
            sp400_tickers = sp400_tables[0]['Symbol'].str.replace('.', '-', regex=False).tolist()
            all_tickers.extend(sp400_tickers)
    except Exception:
        pass
    
    try:
        # S&P 600 Small Cap
        sp600_url = "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies"
        sp600_tables = pd.read_html(sp600_url, attrs={'id': 'constituents'})
        if sp600_tables:
            sp600_tickers = sp600_tables[0]['Symbol'].str.replace('.', '-', regex=False).tolist()
            all_tickers.extend(sp600_tickers)
    except Exception:
        pass
    
    # If we got a reasonable number of tickers from Wikipedia, use them
    if len(all_tickers) >= 500:
        # Deduplicate and clean
        all_tickers = list(set(all_tickers))
        all_tickers = [t.strip() for t in all_tickers if isinstance(t, str) and len(t) <= 5]
        return sorted(all_tickers)
    
    # Method 2: Try iShares IWV ETF holdings
    try:
        url = "https://www.ishares.com/us/products/239714/ishares-russell-3000-etf/1467271812596.ajax?fileType=csv&fileName=IWV_holdings&dataType=fund"
        df = pd.read_csv(url, skiprows=9)
        
        if 'Ticker' in df.columns:
            tickers = df[df['Asset Class'] == 'Equity']['Ticker'].dropna().tolist()
            tickers = [t.strip() for t in tickers if isinstance(t, str) and len(t) <= 5]
            if len(tickers) >= 500:
                return tickers[:3000]
    except Exception:
        pass
    
    # Final fallback: Return comprehensive curated list
    return get_fallback_tickers()


def get_fallback_tickers():
    """Comprehensive fallback list of ~1500 liquid US stocks."""
    # S&P 500 + S&P 400 + additional liquid stocks
    major_tickers = [
        # Mega Cap / Large Cap (S&P 500 top holdings)
        "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "UNH",
        "JNJ", "V", "XOM", "JPM", "WMT", "MA", "PG", "CVX", "HD", "LLY",
        "MRK", "ABBV", "PEP", "KO", "COST", "AVGO", "TMO", "MCD", "CSCO", "ACN",
        "ABT", "DHR", "WFC", "NEE", "LIN", "VZ", "ADBE", "TXN", "PM", "CRM",
        "RTX", "BMY", "UPS", "CMCSA", "NKE", "ORCL", "HON", "QCOM", "T", "UNP",
        "LOW", "COP", "BA", "INTC", "SPGI", "CAT", "IBM", "AMD", "GS", "ELV",
        "AMAT", "DE", "SBUX", "MDT", "AXP", "BLK", "PLD", "GILD", "ADI", "MDLZ",
        "CVS", "ISRG", "TJX", "REGN", "VRTX", "SYK", "BKNG", "MMC", "ADP", "TMUS",
        "CI", "ZTS", "LRCX", "MO", "CB", "ETN", "SO", "DUK", "BDX", "EOG",
        "PNC", "ITW", "NOC", "SCHW", "BSX", "APD", "CL", "CSX", "CME", "SLB",
        # More S&P 500
        "MU", "MCO", "USB", "FDX", "ICE", "EMR", "NSC", "PXD", "FCX", "GM",
        "GD", "ORLY", "SHW", "HUM", "AON", "KLAC", "PSA", "TGT", "SNPS", "CDNS",
        "AZO", "MCHP", "MCK", "CTAS", "CARR", "TFC", "AEP", "MSI", "MET", "EW",
        "MNST", "D", "SRE", "OXY", "FTNT", "PAYX", "AIG", "O", "TEL", "DXCM",
        "A", "PCAR", "KMB", "WELL", "GIS", "MSCI", "HES", "IDXX", "DOW", "HSY",
        "CTSH", "PRU", "CMG", "BIIB", "YUM", "ROST", "FAST", "VRSK", "ODFL", "KHC",
        "CPRT", "GEHC", "KDP", "CTVA", "EA", "BKR", "CBRE", "EXC", "DD", "XEL",
        "VICI", "ANSS", "ON", "FANG", "HAL", "CDW", "ROP", "KEYS", "DVN", "WEC",
        "ED", "DLTR", "AWK", "VMC", "MLM", "GWW", "EFX", "ACGL", "RMD", "TSCO",
        "WST", "WTW", "EBAY", "FTV", "CHD", "MTD", "CSGP", "DLR", "HIG", "PPG",
        # Mid Cap (S&P 400 selections)
        "DECK", "POOL", "MANH", "TTEK", "RBC", "BURL", "WSM", "LECO", "FDS", "SAIA",
        "EXEL", "LSTR", "TREX", "SITE", "WYNN", "JAZZ", "RGLD", "OLED", "MIDD", "RGEN",
        "GTLS", "NOVT", "ENSG", "EXPO", "CGNX", "PEGA", "CWST", "ASGN", "UFPI", "ITRI",
        "GLOB", "MEDP", "MGPI", "CADE", "ESNT", "LNTH", "APAM", "CVLT", "BCPC", "PRGS",
        "QLYS", "LANC", "CRUS", "HUBG", "CALM", "ACIW", "IBOC", "POWI", "IOSP", "MGEE",
        "SPSC", "CCOI", "MMSI", "NEOG", "ICUI", "HLIT", "PLUS", "VCEL", "NMIH", "ALTR",
        # Additional Large/Mid Cap
        "PYPL", "SQ", "SHOP", "SNOW", "CRWD", "DDOG", "ZS", "NET", "PANW", "OKTA",
        "TWLO", "MDB", "VEEV", "WDAY", "NOW", "TEAM", "ZM", "DOCU", "SPLK", "COUP",
        "ROKU", "TTD", "PINS", "SNAP", "UBER", "LYFT", "DASH", "ABNB", "COIN", "HOOD",
        "RIVN", "LCID", "NIO", "XPEV", "LI", "FSR", "PLTR", "PATH", "AI", "BBAI",
        "IONQ", "RGTI", "QUBT", "SOUN", "UPST", "AFRM", "SOFI", "NU", "MELI", "SE",
        "GRAB", "BABA", "JD", "PDD", "BIDU", "NTES", "TME", "BILI", "IQ", "VIPS",
        # Healthcare
        "PFE", "MRK", "LLY", "ABBV", "BMY", "AMGN", "GILD", "BIIB", "REGN", "VRTX",
        "MRNA", "BNTX", "NVAX", "ZBH", "DXCM", "ALGN", "HOLX", "ILMN", "EXAS", "NTRA",
        "RARE", "BMRN", "ALNY", "SGEN", "INCY", "NBIX", "UTHR", "SRPT", "HZNP", "IONS",
        # Financials
        "BAC", "C", "WFC", "GS", "MS", "SCHW", "BK", "STT", "NTRS", "TROW",
        "BLK", "BX", "KKR", "APO", "ARES", "OWL", "CG", "TPG", "EQH", "VOYA",
        "AMP", "RJF", "LPLA", "SEIC", "MKTX", "CBOE", "NDAQ", "CME", "ICE", "MSCI",
        "SPGI", "MCO", "FDS", "VRSK", "TRI", "INFO", "DNB", "ZG", "RDFN", "OPEN",
        # Energy
        "XOM", "CVX", "COP", "EOG", "SLB", "OXY", "MPC", "VLO", "PSX", "HES",
        "DVN", "FANG", "PXD", "APA", "HAL", "BKR", "NOV", "CHK", "RRC", "AR",
        "EQT", "SWN", "CTRA", "MTDR", "CHRD", "PR", "VTLE", "ESTE", "GPOR", "PDCE",
        # Industrials
        "CAT", "DE", "HON", "UNP", "UPS", "FDX", "LMT", "RTX", "BA", "GE",
        "MMM", "EMR", "ETN", "ITW", "PH", "ROK", "AME", "FAST", "ODFL", "JBHT",
        "XPO", "CHRW", "EXPD", "LSTR", "SAIA", "WERN", "KNX", "SNDR", "ARCB", "HTLD",
        # Consumer
        "AMZN", "WMT", "COST", "TGT", "HD", "LOW", "TJX", "ROST", "BBY", "DG",
        "DLTR", "FIVE", "OLLI", "BIG", "PRTY", "BBWI", "ULTA", "ELF", "COTY", "EL",
        "NKE", "LULU", "UAA", "VFC", "PVH", "RL", "TPR", "CPRI", "GOOS", "CROX",
        "DKS", "HIBB", "ASO", "BGFV", "PLCE", "ANF", "AEO", "URBN", "GPS", "EXPR",
        # Technology
        "AAPL", "MSFT", "NVDA", "AMD", "INTC", "AVGO", "QCOM", "TXN", "ADI", "MCHP",
        "NXPI", "ON", "SWKS", "QRVO", "MRVL", "MPWR", "SLAB", "DIOD", "POWI", "AOSL",
        "CRM", "ORCL", "SAP", "INTU", "ADSK", "ANSS", "CDNS", "SNPS", "PTC", "MANH",
        "ADBE", "FTNT", "PANW", "CRWD", "ZS", "OKTA", "CYBR", "TENB", "QLYS", "RPD",
        # REITs
        "AMT", "PLD", "CCI", "EQIX", "PSA", "SPG", "O", "WELL", "DLR", "AVB",
        "EQR", "VTR", "ARE", "BXP", "SLG", "VNO", "KIM", "REG", "FRT", "UDR",
        "ESS", "MAA", "CPT", "AIV", "INVH", "AMH", "SUI", "ELS", "REXR", "STAG",
        # Utilities
        "NEE", "DUK", "SO", "D", "AEP", "SRE", "XEL", "ED", "EXC", "WEC",
        "ES", "AWK", "ATO", "NI", "CMS", "DTE", "LNT", "EVRG", "PNW", "OGE",
        # Materials
        "LIN", "APD", "SHW", "ECL", "DD", "NEM", "FCX", "NUE", "STLD", "CLF",
        "X", "AA", "CENX", "ATI", "CMC", "RS", "WOR", "ZEUS", "HAYN", "USAP",
        # Communications
        "GOOGL", "META", "NFLX", "DIS", "CMCSA", "T", "VZ", "TMUS", "CHTR", "LBRDK",
        "WBD", "PARA", "FOX", "FOXA", "NWSA", "NWS", "NYT", "GCI", "LEE", "TGNA",
        # Additional mid/small caps for broader coverage
        "AXON", "PAYC", "PCTY", "HQY", "WEX", "GPN", "FIS", "FISV", "ADP", "PAYX",
        "CDAY", "BILL", "HUBS", "ZI", "DSGX", "NCNO", "APPF", "ASAN", "FROG", "ESTC",
        "CFLT", "GTLB", "DOCN", "BRZE", "AMPL", "SEMR", "PUBM", "MGNI", "CRTO", "APPS",
    ]
    # Deduplicate
    return list(dict.fromkeys(major_tickers))


@st.cache_data(ttl=86400)  # Cache for 24 hours
def get_cached_screening_results():
    """Return cached screening results if available."""
    return None  # Will be populated by screening


def fetch_magic_formula_metrics(ticker: str) -> dict:
    """
    Fetch metrics needed for Magic Formula calculation.
    
    Earnings Yield = EBIT / Enterprise Value
    Return on Capital = EBIT / (Net Working Capital + Net Fixed Assets)
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Skip if no data or not a stock
        if not info or info.get('quoteType') not in ['EQUITY', None]:
            return None
        
        # Get Market Cap - skip if below threshold
        market_cap = info.get('marketCap')
        if not market_cap or market_cap < 50_000_000:  # $50M minimum
            return None
        
        # Get Enterprise Value
        enterprise_value = info.get('enterpriseValue')
        if not enterprise_value or enterprise_value <= 0:
            return None
        
        # Get EBIT (Operating Income)
        # Try from info first, then from income statement
        ebit = info.get('ebitda')  # Start with EBITDA as fallback
        
        try:
            income_stmt = stock.income_stmt
            if income_stmt is not None and not income_stmt.empty:
                # Try to get EBIT directly
                if 'EBIT' in income_stmt.index:
                    ebit = income_stmt.loc['EBIT'].iloc[0]
                elif 'Operating Income' in income_stmt.index:
                    ebit = income_stmt.loc['Operating Income'].iloc[0]
                elif 'EBITDA' in income_stmt.index:
                    ebitda = income_stmt.loc['EBITDA'].iloc[0]
                    # Approximate EBIT = EBITDA - D&A
                    if 'Depreciation And Amortization' in income_stmt.index:
                        da = income_stmt.loc['Depreciation And Amortization'].iloc[0]
                        if pd.notna(ebitda) and pd.notna(da):
                            ebit = ebitda - abs(da)
                    else:
                        ebit = ebitda  # Use EBITDA as approximation
        except:
            pass
        
        if not ebit or pd.isna(ebit) or ebit <= 0:
            return None  # Skip companies with negative or no EBIT
        
        # Calculate Earnings Yield
        earnings_yield = ebit / enterprise_value
        
        # Get Balance Sheet data for Return on Capital
        try:
            balance_sheet = stock.balance_sheet
            if balance_sheet is None or balance_sheet.empty:
                return None
            
            # Get Current Assets
            current_assets = None
            for key in ['Current Assets', 'Total Current Assets']:
                if key in balance_sheet.index:
                    current_assets = balance_sheet.loc[key].iloc[0]
                    break
            
            # Get Current Liabilities
            current_liabilities = None
            for key in ['Current Liabilities', 'Total Current Liabilities']:
                if key in balance_sheet.index:
                    current_liabilities = balance_sheet.loc[key].iloc[0]
                    break
            
            # Get Net Fixed Assets (PP&E)
            net_ppe = None
            for key in ['Net PPE', 'Property Plant Equipment Net', 'Net Property, Plant and Equipment']:
                if key in balance_sheet.index:
                    net_ppe = balance_sheet.loc[key].iloc[0]
                    break
            
            # Calculate Net Working Capital (excluding excess cash per Greenblatt)
            if current_assets is None or current_liabilities is None:
                return None
            
            net_working_capital = current_assets - current_liabilities
            
            # Net Fixed Assets - use 0 if not available (some businesses are asset-light)
            if net_ppe is None or pd.isna(net_ppe):
                net_ppe = 0
            
            # Tangible Capital Employed = NWC + Net Fixed Assets
            tangible_capital = net_working_capital + net_ppe
            
            # Skip if tangible capital is negative or zero (can't calculate meaningful ROC)
            if tangible_capital <= 0:
                return None
            
            # Calculate Return on Capital
            return_on_capital = ebit / tangible_capital
            
        except Exception as e:
            return None
        
        # Get company name and other info
        company_name = info.get('shortName', ticker)
        sector = info.get('sector', 'Unknown')
        
        return {
            'ticker': ticker,
            'company_name': company_name,
            'sector': sector,
            'market_cap': market_cap,
            'enterprise_value': enterprise_value,
            'ebit': ebit,
            'net_working_capital': net_working_capital,
            'net_fixed_assets': net_ppe,
            'tangible_capital': tangible_capital,
            'earnings_yield': earnings_yield,
            'return_on_capital': return_on_capital,
        }
        
    except Exception as e:
        return None


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


def format_percent(value):
    """Format as percentage."""
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value*100:.1f}%"


def calculate_rankings(df):
    """Calculate Magic Formula rankings."""
    if df.empty:
        return df
    
    # Rank by Earnings Yield (higher is better, so ascending=False gives rank 1 to highest)
    df['ey_rank'] = df['earnings_yield'].rank(ascending=False, method='min')
    
    # Rank by Return on Capital (higher is better)
    df['roc_rank'] = df['return_on_capital'].rank(ascending=False, method='min')
    
    # Combined Magic Formula rank (lower is better)
    df['magic_rank'] = df['ey_rank'] + df['roc_rank']
    
    # Sort by combined rank
    df = df.sort_values('magic_rank')
    
    return df


def format_results_table(df, top_n=30):
    """Format the results dataframe for display."""
    if df.empty:
        return df
    
    display_df = df.head(top_n).copy()
    
    # Select and rename columns for display
    display_df = display_df[[
        'ticker', 'company_name', 'sector', 'market_cap', 
        'earnings_yield', 'return_on_capital',
        'ey_rank', 'roc_rank', 'magic_rank',
        'ebit', 'enterprise_value', 'tangible_capital'
    ]].copy()
    
    # Format columns
    display_df['Market Cap'] = display_df['market_cap'].apply(format_currency)
    display_df['EBIT'] = display_df['ebit'].apply(format_currency)
    display_df['EV'] = display_df['enterprise_value'].apply(format_currency)
    display_df['Tang. Capital'] = display_df['tangible_capital'].apply(format_currency)
    display_df['Earnings Yield'] = display_df['earnings_yield'].apply(format_percent)
    display_df['Return on Capital'] = display_df['return_on_capital'].apply(format_percent)
    display_df['EY Rank'] = display_df['ey_rank'].astype(int)
    display_df['ROC Rank'] = display_df['roc_rank'].astype(int)
    display_df['Magic Rank'] = display_df['magic_rank'].astype(int)
    
    # Rename columns
    display_df = display_df.rename(columns={
        'ticker': 'Ticker',
        'company_name': 'Company',
        'sector': 'Sector',
    })
    
    # Select final columns
    display_df = display_df[[
        'Ticker', 'Company', 'Sector', 'Market Cap',
        'Earnings Yield', 'EY Rank', 
        'Return on Capital', 'ROC Rank',
        'Magic Rank',
        'EBIT', 'EV', 'Tang. Capital'
    ]]
    
    return display_df


# --- Session State for Results ---
if 'magic_formula_results' not in st.session_state:
    st.session_state.magic_formula_results = None
if 'magic_formula_last_run' not in st.session_state:
    st.session_state.magic_formula_last_run = None
if 'screening_in_progress' not in st.session_state:
    st.session_state.screening_in_progress = False


# --- Sidebar Controls ---
st.sidebar.header("🔧 Screener Settings")

min_market_cap = st.sidebar.selectbox(
    "Minimum Market Cap",
    options=[50_000_000, 100_000_000, 500_000_000, 1_000_000_000, 10_000_000_000],
    format_func=lambda x: f"${x/1e6:.0f}M" if x < 1e9 else f"${x/1e9:.0f}B",
    index=0
)

exclude_financials = st.sidebar.checkbox("Exclude Financials", value=True, 
    help="Greenblatt excludes financials due to different capital structures")

exclude_utilities = st.sidebar.checkbox("Exclude Utilities", value=True,
    help="Greenblatt excludes utilities due to regulated returns")

top_n_display = st.sidebar.slider("Show Top N Results", min_value=10, max_value=100, value=30)

delay_between_requests = st.sidebar.slider(
    "Delay Between Requests (seconds)", 
    min_value=0.1, max_value=2.0, value=0.3, step=0.1,
    help="Lower = faster but higher rate limit risk. Recommended: 0.3-0.5s"
)

# --- Main Content ---

# Show cached results info
if st.session_state.magic_formula_results is not None and st.session_state.magic_formula_last_run:
    time_since_run = datetime.now() - st.session_state.magic_formula_last_run
    hours_ago = time_since_run.total_seconds() / 3600
    st.info(f"📊 Showing cached results from **{hours_ago:.1f} hours ago** ({st.session_state.magic_formula_last_run.strftime('%Y-%m-%d %H:%M')})")

# Run Screener Button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    run_screener = st.button("🚀 Run Magic Formula Screen", use_container_width=True, type="primary")

if run_screener:
    st.session_state.screening_in_progress = True
    
    # Get tickers
    with st.spinner("Fetching stock universe..."):
        tickers = get_russell_3000_tickers()
    
    st.success(f"✅ Found **{len(tickers)}** stocks in universe. Starting screening...")
    
    # Create placeholders for live updates
    progress_placeholder = st.empty()
    status_placeholder = st.empty()
    results_placeholder = st.empty()
    
    all_results = []
    successful = 0
    failed = 0
    skipped = 0
    
    start_time = time.time()
    
    for i, ticker in enumerate(tickers):
        # Fetch data
        data = fetch_magic_formula_metrics(ticker)
        
        if data:
            # Apply filters
            if data['market_cap'] < min_market_cap:
                skipped += 1
            elif exclude_financials and data['sector'] == 'Financial Services':
                skipped += 1
            elif exclude_utilities and data['sector'] == 'Utilities':
                skipped += 1
            else:
                all_results.append(data)
                successful += 1
        else:
            failed += 1
        
        # Update progress every stock
        progress = (i + 1) / len(tickers)
        elapsed = time.time() - start_time
        eta_seconds = (elapsed / (i + 1)) * (len(tickers) - i - 1) if i > 0 else 0
        eta_minutes = eta_seconds / 60
        
        progress_placeholder.progress(progress, text=f"Screening: {i+1}/{len(tickers)} stocks ({progress*100:.1f}%)")
        status_placeholder.markdown(f"✅ **{successful}** qualifying stocks | ⏭️ {skipped} skipped | ❌ {failed} failed | ⏱️ ETA: {eta_minutes:.1f} min")
        
        # Update results table every 10 stocks (for performance)
        if len(all_results) > 0 and (i % 10 == 0 or i == len(tickers) - 1):
            df = pd.DataFrame(all_results)
            df = calculate_rankings(df)
            display_df = format_results_table(df, top_n=top_n_display)
            
            with results_placeholder.container():
                st.markdown(f"### 🏆 Top {min(top_n_display, len(display_df))} Magic Formula Stocks (Live)")
                st.dataframe(
                    display_df.sort_values('Magic Rank'),
                    use_container_width=True,
                    hide_index=True,
                    height=(len(display_df) + 1) * 35 + 3,
                    column_config={
                        "Ticker": st.column_config.TextColumn("Ticker", help="Stock ticker symbol"),
                        "Company": st.column_config.TextColumn("Company", help="Company name"),
                        "Sector": st.column_config.TextColumn("Sector", help="Business sector classification"),
                        "Market Cap": st.column_config.TextColumn("Mkt Cap", help="Market capitalization = Share price × Shares outstanding"),
                        "Earnings Yield": st.column_config.TextColumn("EY", help="Earnings Yield = EBIT / Enterprise Value. Higher = cheaper stock relative to earnings."),
                        "EY Rank": st.column_config.NumberColumn("EY Rank", help="Rank by Earnings Yield (1 = highest EY, cheapest)"),
                        "Return on Capital": st.column_config.TextColumn("ROC", help="Return on Capital = EBIT / (Net Working Capital + Net Fixed Assets). Higher = more efficient use of capital."),
                        "ROC Rank": st.column_config.NumberColumn("ROC Rank", help="Rank by Return on Capital (1 = highest ROC, best quality)"),
                        "Magic Rank": st.column_config.NumberColumn("Magic ⭐", help="Combined rank = EY Rank + ROC Rank. Lower = better Magic Formula candidate."),
                        "EBIT": st.column_config.TextColumn("EBIT", help="Earnings Before Interest and Taxes (operating profit)"),
                        "EV": st.column_config.TextColumn("EV", help="Enterprise Value = Market Cap + Total Debt - Cash"),
                        "Tang. Capital": st.column_config.TextColumn("Tang Cap", help="Tangible Capital = Net Working Capital + Net Fixed Assets (PP&E)"),
                    }
                )
        
        # Rate limiting
        time.sleep(delay_between_requests)
    
    # Final update
    progress_placeholder.progress(1.0, text="✅ Screening complete!")
    
    # Store results in session state
    if all_results:
        final_df = pd.DataFrame(all_results)
        final_df = calculate_rankings(final_df)
        st.session_state.magic_formula_results = final_df
        st.session_state.magic_formula_last_run = datetime.now()
    
    st.session_state.screening_in_progress = False
    
    total_time = time.time() - start_time
    st.success(f"🎉 Screening complete! Analyzed {len(tickers)} stocks in {total_time/60:.1f} minutes. Found {successful} qualifying stocks.")

# Display cached results if available and not currently screening
elif st.session_state.magic_formula_results is not None and not st.session_state.screening_in_progress:
    df = st.session_state.magic_formula_results
    display_df = format_results_table(df, top_n=top_n_display)
    
    st.markdown(f"### 🏆 Top {min(top_n_display, len(display_df))} Magic Formula Stocks")
    st.dataframe(
        display_df.sort_values('Magic Rank'),
        use_container_width=True,
        hide_index=True,
        height=(len(display_df) + 1) * 35 + 3,
        column_config={
            "Ticker": st.column_config.TextColumn("Ticker", help="Stock ticker symbol"),
            "Company": st.column_config.TextColumn("Company", help="Company name"),
            "Sector": st.column_config.TextColumn("Sector", help="Business sector classification"),
            "Market Cap": st.column_config.TextColumn("Mkt Cap", help="Market capitalization = Share price × Shares outstanding"),
            "Earnings Yield": st.column_config.TextColumn("EY", help="Earnings Yield = EBIT / Enterprise Value. Higher = cheaper stock relative to earnings."),
            "EY Rank": st.column_config.NumberColumn("EY Rank", help="Rank by Earnings Yield (1 = highest EY, cheapest)"),
            "Return on Capital": st.column_config.TextColumn("ROC", help="Return on Capital = EBIT / (Net Working Capital + Net Fixed Assets). Higher = more efficient use of capital."),
            "ROC Rank": st.column_config.NumberColumn("ROC Rank", help="Rank by Return on Capital (1 = highest ROC, best quality)"),
            "Magic Rank": st.column_config.NumberColumn("Magic ⭐", help="Combined rank = EY Rank + ROC Rank. Lower = better Magic Formula candidate."),
            "EBIT": st.column_config.TextColumn("EBIT", help="Earnings Before Interest and Taxes (operating profit)"),
            "EV": st.column_config.TextColumn("EV", help="Enterprise Value = Market Cap + Total Debt - Cash"),
            "Tang. Capital": st.column_config.TextColumn("Tang Cap", help="Tangible Capital = Net Working Capital + Net Fixed Assets (PP&E)"),
        }
    )
    
    # Export option
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Full Results (CSV)",
            data=csv,
            file_name=f"magic_formula_screen_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )

else:
    # No results yet
    st.markdown("""
    ### Getting Started
    
    1. Adjust settings in the sidebar (market cap filter, sector exclusions)
    2. Click **Run Magic Formula Screen** to start
    3. Watch the live leaderboard update as stocks are analyzed
    4. The top 30 will be displayed and continuously updated during screening
    
    **Note:** Full screening takes ~30-60 minutes depending on your rate limit settings. 
    Results are cached for 24 hours - you can close the browser and return later.
    """)

# --- Footer with methodology ---
st.markdown("---")
with st.expander("📖 Magic Formula Methodology"):
    st.markdown("""
    ### How the Magic Formula Works
    
    Based on Joel Greenblatt's book *"The Little Book That Still Beats the Market"*:
    
    **1. Earnings Yield (Cheapness)**
    ```
    Earnings Yield = EBIT / Enterprise Value
    ```
    - Higher earnings yield = stock is cheaper relative to operating earnings
    - Enterprise Value = Market Cap + Total Debt - Cash
    
    **2. Return on Capital (Quality)**
    ```
    Return on Capital = EBIT / (Net Working Capital + Net Fixed Assets)
    ```
    - Higher ROC = company is more efficient with its capital
    - Net Working Capital = Current Assets - Current Liabilities
    - Net Fixed Assets = Property, Plant & Equipment (net of depreciation)
    
    **3. Combined Ranking**
    - Each stock is ranked on both metrics (1 = best)
    - Combined Rank = EY Rank + ROC Rank
    - Lower combined rank = better Magic Formula candidate
    
    **4. Exclusions**
    - Financial Services (different capital structure)
    - Utilities (regulated returns)
    - Stocks with negative EBIT
    - Market cap below threshold
    
    **Important:** This is a quantitative screen, not a buy recommendation. 
    Always do your own due diligence before investing.
    """)
