import pandas as pd

class MACrossoverStrategy:
    """
    Moving average crossover strategy.
    Signal = +1 when short MA > long MA, 0 otherwise.
    No lookahead: positions are shifted forward one period before backtest.
    """

    def __init__(self, short_window=20, long_window=200):
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Args:
            df: DataFrame with 'Close' column

        Returns:
            DataFrame with added columns:
                MA_short, MA_long, Position (0 or 1), Signal (crossover events)
        """
        df = df.copy()
        df["MA_short"] = df["Close"].rolling(self.short_window, min_periods=self.short_window).mean()
        df["MA_long"]  = df["Close"].rolling(self.long_window,  min_periods=self.long_window).mean()
        df["Position"] = (df["MA_short"] > df["MA_long"]).astype(int)
        df["Signal"]   = df["Position"].diff()
        return df