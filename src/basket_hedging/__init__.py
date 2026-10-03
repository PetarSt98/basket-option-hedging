"""One-currency research hedge backtester. No broker-compliance claims."""

from .api import BacktestConfig, backtest_engine
from .engine import Engine, PolicyView, Result
from .policies import no_hedge, stock_delta_hedge

__all__ = ["BacktestConfig", "backtest_engine", "Engine", "PolicyView", "Result",
           "no_hedge", "stock_delta_hedge"]

__version__ = "0.2.0"
