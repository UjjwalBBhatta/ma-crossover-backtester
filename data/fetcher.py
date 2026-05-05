import yfinance as yf

def get_data(ticker, start, end):
    """
    Fetch historical OHLCV data for a ticker.

    Args:
        ticker: stock symbol (e.g. 'AAPL')
        start: start date string (e.g. '2020-01-01')
        end: end date string (e.g. '2024-01-01')

    Returns:
        DataFrame with OHLCV columns, single-level column index
    """
    df = yf.download(ticker, start=start, end=end, auto_adjust=True)
    df.columns = df.columns.droplevel(1) if df.columns.nlevels > 1 else df.columns
    return df