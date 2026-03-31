import numpy as np
import pandas as pd

def daily_returns(prices):
    return prices.pct_change()

def moving_avg(prices, frame):
    return prices.rolling(frame).mean()

def rolling_volatility(prices, frame=20):
    returns = daily_returns(prices)
    return returns.rolling(frame).std()

def compute_variance(arr):
    """
    Compute sample variance from scratch.
    
    Formula: Var(X) = Σ(xᵢ - x̄)² / (n-1)
    
    Args:
        arr: list or array of numbers
        
    Returns:
        float: sample variance
    """
    n = len(arr)
    mean = sum(arr)/n

    sq_dv_sum = sum((x-mean)**2 for x in arr)
    
    return sq_dv_sum / (n - 1)

def compute_covariance(x,y):
    """Computes covariance.
    
    Formula: Cov(x,y) = Σ(xᵢ - x̄)(yᵢ - ȳ) / (n-1)
    
    Args:
        x: first array
        y: second array
        
    Returns:
        float covariance """
    assert len(x) == len(y)
    n = len(x)
    mean_x = sum(x)/n
    mean_y = sum(y)/n

    return sum((x[i]-mean_x)*(y[i]-mean_y) for i in range(n))/(n-1)

def compute_corelation(x,y):
    """Computes corelation between two data
    
    Formula: Correlation(x,y) = Cov(x/y)/(sqrt(var(x))*sqrt(var(y)))
    Args:
        x: first array
        y: second array
        
    Returns:
        float correlation"""
    return compute_covariance(x,y)/((compute_variance(x)*compute_variance(y))**0.5)
 
def log_returns(prices):
    """Takes a dataset as a parmeter
       Calculates the log return of a dataset
    """
    return np.log(prices/prices.shift(1)).dropna()

def calculate_maxdd(prices):
    """Computes the max drawdown of a ticker.
    First calculate the maximum for each
    After that calculates the drawdowns qand creates an array
    then the minimum of the drawdown(the most negative) is the max drawdown"""
    run_max = prices.cummax()
    dropdows = (prices-run_max)/run_max
    return dropdows.min()

def calculate_annual_sharpe(returns, rf = 0.0, periods_per_year = 252):
    """Computes the annualized sharpe ratio of a Ticker
    """
    daily_rf = rf/periods_per_year
    return ((returns.mean()-daily_rf)/returns.std())*np.sqrt(periods_per_year)