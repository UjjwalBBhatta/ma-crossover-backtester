import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.stats import *


def get_data(ticker, start, end):
    """
    Fetch historical price data for a ticker
    
    Args:
        ticker: stock symbol 
        start: start date string 
        end: end date string 
    
    Returns:
        DataFrame with OHLCV data
    """
    df = yf.download(ticker, start=start, end=end)
    df.columns = df.columns.droplevel(1)
    return df


def generate_signals(df, short_window=20, long_window=200):
    """
    Generate MA crossover buy/sell signals
    
    Args:
        df: DataFrame with price data
        short_window: fast MA period
        long_window: slow MA period
    
    Returns:
        DataFrame with signals added
    """
    pass


def backtest(df, initial_capital=100000):
    """
    Run backtest on generated signals
    
    Args:
        df: DataFrame with signals
        initial_capital: starting capital
    
    Returns:
        DataFrame with portfolio values
    """
    pass


def calculate_metrics(df, initial_capital=100000):
    """
    Calculate performance metrics
    
    Args:
        df: DataFrame with portfolio values
    
    Returns:
        dict of metrics
    """
    pass

if __name__ == "__main__":
    df = get_data("AAPL", "2020-01-01", "2024-01-01")
    print(df.head())
    print(df.columns)