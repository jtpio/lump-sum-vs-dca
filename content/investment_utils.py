"""Investment strategy utilities for Lump Sum vs DCA comparison."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd


@dataclass
class InvestmentResult:
    """Results from an investment strategy backtest."""

    strategy_name: str
    total_invested: float
    final_value: float
    total_return: float
    total_return_pct: float
    annualized_return: float
    max_drawdown: float
    volatility: float
    sharpe_ratio: float
    timeline: pd.DataFrame  # Date-indexed DataFrame with portfolio values


@dataclass
class BacktestParams:
    """Parameters for running a backtest."""

    total_amount: float
    start_date: str  # YYYY-MM-DD format
    end_date: str  # YYYY-MM-DD format
    dca_frequency: str = "monthly"  # 'weekly', 'monthly', 'quarterly'
    dca_periods: Optional[int] = (
        None  # Number of DCA periods (overrides end_date for DCA duration)
    )
    risk_free_rate: float = 0.02  # Annual risk-free rate for Sharpe ratio


def calculate_lump_sum(
    prices: pd.Series,
    params: BacktestParams,
) -> InvestmentResult:
    """
    Calculate returns for lump sum investment strategy.

    Invests the entire amount at the start date.

    Args:
        prices: Price series indexed by date
        params: Backtest parameters

    Returns:
        InvestmentResult with strategy performance
    """
    start_date = pd.to_datetime(params.start_date)
    end_date = pd.to_datetime(params.end_date)

    # Filter prices to the investment period
    mask = (prices.index >= start_date) & (prices.index <= end_date)
    period_prices = prices[mask].copy()

    if len(period_prices) < 2:
        raise ValueError("Not enough price data for the specified period")

    # Buy shares at the first available date
    buy_price = period_prices.iloc[0]
    shares = params.total_amount / buy_price

    # Calculate portfolio value over time
    portfolio_values = period_prices * shares

    # Create timeline DataFrame
    timeline = pd.DataFrame(
        {
            "date": portfolio_values.index,
            "price": period_prices.values,
            "shares": shares,
            "invested": params.total_amount,
            "value": portfolio_values.values,
            "return_pct": ((portfolio_values.values / params.total_amount) - 1) * 100,
        }
    ).set_index("date")

    # Calculate metrics
    final_value = portfolio_values.iloc[-1]
    total_return = final_value - params.total_amount
    total_return_pct = (total_return / params.total_amount) * 100

    # Calculate annualized return
    years = (period_prices.index[-1] - period_prices.index[0]).days / 365.25
    if years > 0:
        annualized_return = (
            (final_value / params.total_amount) ** (1 / years) - 1
        ) * 100
    else:
        annualized_return = 0

    # Calculate max drawdown
    max_drawdown = calculate_max_drawdown(portfolio_values)

    # Calculate volatility (annualized)
    daily_returns = portfolio_values.pct_change().dropna()
    volatility = daily_returns.std() * np.sqrt(252) * 100

    # Calculate Sharpe ratio
    excess_return = (annualized_return / 100) - params.risk_free_rate
    if volatility > 0:
        sharpe_ratio = excess_return / (volatility / 100)
    else:
        sharpe_ratio = 0

    return InvestmentResult(
        strategy_name="Lump Sum",
        total_invested=params.total_amount,
        final_value=final_value,
        total_return=total_return,
        total_return_pct=total_return_pct,
        annualized_return=annualized_return,
        max_drawdown=max_drawdown,
        volatility=volatility,
        sharpe_ratio=sharpe_ratio,
        timeline=timeline,
    )


def calculate_dca(
    prices: pd.Series,
    params: BacktestParams,
    *,
    dca_duration_months: Optional[int] = None,
) -> InvestmentResult:
    """Calculate returns for Dollar Cost Averaging (DCA).

    Invests equal amounts at regular intervals for an optional fixed
    *contribution period*, then holds until `params.end_date`.

    Args:
        prices: Price series indexed by date
        params: Backtest parameters
        dca_duration_months: Optional contribution period length (months). If
            provided, contributions stop after this duration and any remaining
            time (until `params.end_date`) is a holding period.

    Returns:
        InvestmentResult with strategy performance
    """
    start_date = pd.to_datetime(params.start_date)
    end_date = pd.to_datetime(params.end_date)

    # Filter prices to the investment period
    mask = (prices.index >= start_date) & (prices.index <= end_date)
    period_prices = prices[mask].copy()

    if len(period_prices) < 2:
        raise ValueError("Not enough price data for the specified period")

    # Determine DCA investment dates
    freq_map = {"weekly": "W", "monthly": "MS", "quarterly": "QS"}
    freq = freq_map.get(params.dca_frequency)
    if freq is None:
        raise ValueError(
            f"Unknown dca_frequency '{params.dca_frequency}'. "
            "Expected one of: weekly, monthly, quarterly."
        )

    if params.dca_periods is not None and dca_duration_months is not None:
        raise ValueError(
            "Use either params.dca_periods or dca_duration_months, not both"
        )

    if dca_duration_months is not None:
        if params.dca_frequency != "monthly":
            raise ValueError("dca_duration_months is only supported for monthly DCA")
        contribution_end = start_date + pd.DateOffset(months=dca_duration_months)
        investment_dates = pd.date_range(
            start=start_date, end=contribution_end, freq=freq
        )
    elif params.dca_periods is not None:
        # Use specified number of contribution periods
        if params.dca_periods <= 0:
            raise ValueError("dca_periods must be a positive integer")
        investment_dates = pd.date_range(
            start=start_date, periods=params.dca_periods, freq=freq
        )
    else:
        # Contribute throughout the full investment window
        investment_dates = pd.date_range(start=start_date, end=end_date, freq=freq)

    if len(investment_dates) == 0:
        investment_dates = pd.DatetimeIndex([start_date])

    # Amount per investment
    amount_per_investment = params.total_amount / len(investment_dates)

    # Track shares and investments over time
    total_shares = 0
    total_invested = 0
    investment_records = []

    for inv_date in investment_dates:
        # Find the closest available price date (on or after investment date)
        available_dates = period_prices.index[period_prices.index >= inv_date]
        if len(available_dates) == 0:
            continue

        actual_date = available_dates[0]
        price = period_prices.loc[actual_date]

        shares_bought = amount_per_investment / price
        total_shares += shares_bought
        total_invested += amount_per_investment

        investment_records.append(
            {
                "date": actual_date,
                "price": price,
                "amount_invested": amount_per_investment,
                "shares_bought": shares_bought,
                "total_shares": total_shares,
                "total_invested": total_invested,
            }
        )

    if not investment_records:
        raise ValueError("No investments could be made in the specified period")

    investments_df = pd.DataFrame(investment_records)

    # Calculate portfolio value for each day
    timeline_data = []
    running_shares = 0
    running_invested = 0
    inv_idx = 0

    for date in period_prices.index:
        # Check if there's an investment on this date
        while (
            inv_idx < len(investments_df)
            and investments_df.iloc[inv_idx]["date"] <= date
        ):
            running_shares = investments_df.iloc[inv_idx]["total_shares"]
            running_invested = investments_df.iloc[inv_idx]["total_invested"]
            inv_idx += 1

        price = period_prices.loc[date]
        value = running_shares * price

        timeline_data.append(
            {
                "date": date,
                "price": price,
                "shares": running_shares,
                "invested": running_invested,
                "value": value,
                "return_pct": ((value / running_invested) - 1) * 100
                if running_invested > 0
                else 0,
            }
        )

    timeline = pd.DataFrame(timeline_data).set_index("date")

    # Calculate metrics
    final_value = timeline["value"].iloc[-1]
    total_return = final_value - total_invested
    total_return_pct = (
        (total_return / total_invested) * 100 if total_invested > 0 else 0
    )

    # Calculate time-weighted annualized return
    # Use the period from first investment to end
    first_investment_date = investments_df["date"].iloc[0]
    years = (period_prices.index[-1] - first_investment_date).days / 365.25

    if years > 0 and total_invested > 0:
        # Modified Dietz method approximation for DCA
        annualized_return = ((final_value / total_invested) ** (1 / years) - 1) * 100
    else:
        annualized_return = 0

    # Calculate max drawdown
    portfolio_values = timeline["value"]
    max_drawdown = calculate_max_drawdown(portfolio_values[portfolio_values > 0])

    # Calculate volatility (annualized)
    portfolio_values_nonzero = timeline[timeline["value"] > 0]["value"]
    if len(portfolio_values_nonzero) > 1:
        daily_returns = portfolio_values_nonzero.pct_change().dropna()
        volatility = daily_returns.std() * np.sqrt(252) * 100
    else:
        volatility = 0

    # Calculate Sharpe ratio
    excess_return = (annualized_return / 100) - params.risk_free_rate
    if volatility > 0:
        sharpe_ratio = excess_return / (volatility / 100)
    else:
        sharpe_ratio = 0

    return InvestmentResult(
        strategy_name=f"DCA ({params.dca_frequency.capitalize()})",
        total_invested=total_invested,
        final_value=final_value,
        total_return=total_return,
        total_return_pct=total_return_pct,
        annualized_return=annualized_return,
        max_drawdown=max_drawdown,
        volatility=volatility,
        sharpe_ratio=sharpe_ratio,
        timeline=timeline,
    )


def calculate_max_drawdown(values: pd.Series) -> float:
    """Calculate the maximum drawdown of a series.

    Args:
        values: Series of portfolio values

    Returns:
        Maximum drawdown as a percentage (positive number)
    """
    if len(values) < 2:
        return 0.0

    peak = values.expanding(min_periods=1).max()
    drawdown = (values - peak) / peak
    max_drawdown = drawdown.min() * -100  # Convert to positive percentage

    return max_drawdown if not np.isnan(max_drawdown) else 0.0


def run_multiple_backtests(
    prices: pd.Series,
    total_amount: float,
    start_dates: list[str],
    investment_horizon_years: int,
    dca_frequency: str = "monthly",
    dca_duration_months: int = 12,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> pd.DataFrame:
    """
    Run multiple backtests starting from different dates.

    Useful for analyzing how starting date affects outcomes.

    Args:
        prices: Price series indexed by date
        total_amount: Total amount to invest
        start_dates: List of start dates (YYYY-MM-DD)
        investment_horizon_years: How long to hold the investment
        dca_frequency: Frequency of DCA investments
        dca_duration_months: How many months to spread DCA over
        progress_callback: Optional callback(current, total) for progress updates

    Returns:
        DataFrame with results for each start date
    """
    results = []
    total = len(start_dates)

    for i, start_date in enumerate(start_dates):
        if progress_callback:
            progress_callback(i, total)
        try:
            start_dt = pd.to_datetime(start_date)
            end_dt = start_dt + pd.DateOffset(years=int(investment_horizon_years))

            params_ls = BacktestParams(
                total_amount=total_amount,
                start_date=start_date,
                end_date=end_dt.strftime("%Y-%m-%d"),
            )

            params_dca = BacktestParams(
                total_amount=total_amount,
                start_date=start_date,
                end_date=end_dt.strftime("%Y-%m-%d"),
                dca_frequency=dca_frequency,
            )

            ls_result = calculate_lump_sum(prices, params_ls)
            dca_result = calculate_dca(
                prices,
                params_dca,
                dca_duration_months=dca_duration_months,
            )

            results.append(
                {
                    "start_date": start_date,
                    "end_date": end_dt.strftime("%Y-%m-%d"),
                    "ls_final_value": ls_result.final_value,
                    "ls_return_pct": ls_result.total_return_pct,
                    "ls_annualized": ls_result.annualized_return,
                    "ls_max_drawdown": ls_result.max_drawdown,
                    "dca_final_value": dca_result.final_value,
                    "dca_return_pct": dca_result.total_return_pct,
                    "dca_annualized": dca_result.annualized_return,
                    "dca_max_drawdown": dca_result.max_drawdown,
                    "ls_wins": ls_result.final_value > dca_result.final_value,
                    "difference_pct": ls_result.total_return_pct
                    - dca_result.total_return_pct,
                }
            )
        except (ValueError, IndexError):
            # Skip periods with insufficient data
            continue

    # Final progress update to show completion
    if progress_callback:
        progress_callback(total, total)

    return pd.DataFrame(results)


def calculate_rolling_comparison(
    prices: pd.Series,
    total_amount: float,
    investment_horizon_years: int,
    dca_frequency: str = "monthly",
    dca_duration_months: int = 12,
    rolling_start_interval: str = "MS",  # Monthly start dates
) -> pd.DataFrame:
    """
    Calculate rolling comparison of LS vs DCA over the entire price history.

    Args:
        prices: Price series indexed by date
        total_amount: Total amount to invest
        investment_horizon_years: Investment holding period
        dca_frequency: DCA investment frequency
        dca_duration_months: Duration of DCA investment period
        rolling_start_interval: How often to start a new comparison

    Returns:
        DataFrame with rolling comparison results
    """
    # Generate start dates
    min_date = prices.index.min()
    max_start_date = prices.index.max() - pd.DateOffset(
        years=int(investment_horizon_years)
    )

    if max_start_date <= min_date:
        raise ValueError("Price history too short for the specified investment horizon")

    start_dates = pd.date_range(
        start=min_date, end=max_start_date, freq=rolling_start_interval
    )
    start_dates_str = [d.strftime("%Y-%m-%d") for d in start_dates]

    return run_multiple_backtests(
        prices=prices,
        total_amount=total_amount,
        start_dates=start_dates_str,
        investment_horizon_years=investment_horizon_years,
        dca_frequency=dca_frequency,
        dca_duration_months=dca_duration_months,
    )


def format_currency(value: float, currency: str = "USD") -> str:
    """Format a value as currency."""
    symbols = {"USD": "$", "EUR": "€", "GBP": "£"}
    symbol = symbols.get(currency, currency)
    return f"{symbol}{value:,.0f}"


def format_percentage(value: float, decimals: int = 2) -> str:
    """Format a value as percentage."""
    return f"{value:,.{decimals}f}%"
