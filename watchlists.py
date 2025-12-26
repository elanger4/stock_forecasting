"""
Centralized Watchlist Configuration

Add new watchlists here. Each watchlist is a dictionary entry with:
- Key: Watchlist name (displayed in UI)
- Value: List of stock tickers

The DEFAULT_WATCHLIST determines which watchlist is selected on page load.
"""

WATCHLISTS = {
    "watchlist_0": [
        "ADBE", "AMD", "AMZN", "AXP", "AXTI", "BTI", "CELH", "FUBO",
        "HNST", "LNTH", "META", "MU", "NVDA", "PLTR", "PYPL", "RVLV", "TSLZ", "TSM", "VICE"
    ],
    "watchlist_1": [
        "PLTR",   # Palantir
        "NVDA",   # Nvidia
        "HOOD",   # Robinhood
        "TTWO",   # Take Two
    ],
    # Add more watchlists below as needed:
}

DEFAULT_WATCHLIST = "watchlist_0"


def get_watchlist_names() -> list:
    """Return list of all watchlist names."""
    return list(WATCHLISTS.keys())


def get_watchlist_stocks(watchlist_name: str) -> list:
    """Return list of stocks for a given watchlist name."""
    return WATCHLISTS.get(watchlist_name, WATCHLISTS[DEFAULT_WATCHLIST])


def get_default_watchlist() -> str:
    """Return the default watchlist name."""
    return DEFAULT_WATCHLIST
