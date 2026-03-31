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
    df = df.copy()
    df["MA_20"] = df["Close"].rolling(short_window).mean()
    df["MA_200"] = df["Close"].rolling(long_window).mean()
    df["Positions"] = (df["MA_20"]>df["MA_200"]).astype(int)
    df["Signals"] = df["Positions"].diff()
    return df

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

