from dataclasses import dataclass, field
import pandas as pd
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.stats import calculate_annual_sharpe, calculate_maxdd


@dataclass
class EquityReport:
    """
    Output of a completed backtest run.

    Attributes:
        equity_curve: portfolio value at each timestep
        trades:       DataFrame of entry/exit events (date, type, price)
        metrics:      dict of performance statistics
    """
    equity_curve: pd.Series
    trades:       pd.DataFrame
    metrics:      dict


class BacktestEngine:
    """
    Generic single-asset backtester.

    Usage:
        engine = BacktestEngine(initial_capital=100_000)
        strategy = MACrossoverStrategy(short_window=20, long_window=200)
        df_with_signals = strategy.generate_signals(df)
        report = engine.run(df_with_signals)
    """

    def __init__(self, initial_capital: float = 100_000, fee_bps: float = 0.0, slippage_bps: float = 0.0):
        """
        Args:
            initial_capital: starting portfolio value in currency units
            fee_bps:         one-way commission in basis points (e.g. 10 = 0.10%)
            slippage_bps:    one-way slippage in basis points
        """
        self.initial_capital = initial_capital
        self.fee_bps         = fee_bps
        self.slippage_bps    = slippage_bps

    def _cost_per_trade(self) -> float:
        """Total round-trip cost as a fraction of trade value."""
        return 2 * (self.fee_bps + self.slippage_bps) / 10_000

    def run(self, df: pd.DataFrame) -> EquityReport:
        """
        Execute backtest on a DataFrame that already contains a 'Position' column.

        Position column must be 0 or 1 and must NOT be pre-shifted
        (shifting is handled here to make the no-lookahead guarantee explicit).

        Args:
            df: DataFrame with columns ['Close', 'Position', 'Signal']

        Returns:
            EquityReport with equity_curve, trades, and metrics
        """
        df = df.copy().dropna(subset=["Position"])

        # Shift positions by 1: we act on tomorrow's open using today's signal
        df["Position_lagged"] = df["Position"].shift(1)

        # Daily strategy return (gross)
        df["Returns"]          = df["Close"].pct_change()
        df["Strategy_Returns"] = df["Position_lagged"] * df["Returns"]

        # Deduct costs on each trade (Signal != 0 means a crossover happened)
        trade_mask = df["Signal"].shift(1).abs() == 1  # cost on the day we act
        df.loc[trade_mask, "Strategy_Returns"] -= self._cost_per_trade()

        # Equity curve
        df["Equity"] = (1 + df["Strategy_Returns"]).cumprod() * self.initial_capital

        # Trade log
        trades = df[df["Signal"] != 0][["Close", "Signal"]].copy()
        trades.columns = ["Price", "Type"]
        trades["Type"] = trades["Type"].map({1: "BUY", -1: "SELL"})

        metrics = self._compute_metrics(df)

        return EquityReport(
            equity_curve=df["Equity"],
            trades=trades,
            metrics=metrics,
        )

    def _compute_metrics(self, df: pd.DataFrame) -> dict:
        years        = len(df) / 252
        final_value  = df["Equity"].iloc[-1]
        total_return = (final_value - self.initial_capital) / self.initial_capital
        cagr         = (final_value / self.initial_capital) ** (1 / years) - 1
        sharpe       = calculate_annual_sharpe(df["Strategy_Returns"].dropna())
        max_dd       = calculate_maxdd(df["Equity"])
        win_rate     = (df["Strategy_Returns"] > 0).mean()

        return {
            "Total Return": f"{total_return * 100:.2f}%",
            "CAGR":         f"{cagr * 100:.2f}%",
            "Sharpe Ratio": f"{sharpe:.2f}",
            "Max Drawdown": f"{max_dd * 100:.2f}%",
            "Win Rate":     f"{win_rate * 100:.2f}%",
        }