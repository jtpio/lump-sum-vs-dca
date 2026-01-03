"""
Market data module for backtesting investment strategies.

This module provides historical market data for analysis using
pre-bundled static JSON data (MSCI World 1972-2025).

The static data works offline and in browser environments (Pyodide/JupyterLite).
"""

import json
import pandas as pd
from pathlib import Path
from typing import Optional


# Path to bundled data files
_DATA_DIR = Path(__file__).parent / "data"

# Available static datasets
STATIC_DATASETS = {
    "MSCI World": "msci_world.json",
}


def load_msci_world(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.Series:
    """
    Load MSCI World Index data from the bundled JSON file.

    Data covers 1972-2025 (13,800+ trading days).

    Args:
        start_date: Optional start date filter (YYYY-MM-DD)
        end_date: Optional end date filter (YYYY-MM-DD)

    Returns:
        pd.Series with date index and closing prices

    Example:
        >>> prices = load_msci_world('2000-01-01', '2020-12-31')
        >>> print(f"Loaded {len(prices)} days of MSCI World data")
    """
    return load_static_data("MSCI World", start_date, end_date)


def load_static_data(
    dataset_name: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.Series:
    """
    Load data from a bundled static JSON file.

    Args:
        dataset_name: Name of the dataset (e.g., 'MSCI World')
        start_date: Optional start date filter (YYYY-MM-DD)
        end_date: Optional end date filter (YYYY-MM-DD)

    Returns:
        pd.Series with date index and closing prices
    """
    if dataset_name not in STATIC_DATASETS:
        available = ", ".join(STATIC_DATASETS.keys())
        raise ValueError(f"Unknown dataset '{dataset_name}'. Available: {available}")

    json_path = _DATA_DIR / STATIC_DATASETS[dataset_name]

    if not json_path.exists():
        raise FileNotFoundError(
            f"Data file not found: {json_path}. "
            f"Run the data export script to generate it."
        )

    with open(json_path, "r") as f:
        data = json.load(f)

    # Convert to pandas Series
    prices_dict = data["prices"]
    prices = pd.Series(prices_dict, name=data.get("ticker", dataset_name))
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()

    # Apply date filters
    if start_date:
        prices = prices[prices.index >= pd.to_datetime(start_date)]
    if end_date:
        prices = prices[prices.index <= pd.to_datetime(end_date)]

    return prices


def get_static_data_info() -> dict:
    """
    Get information about available static datasets.

    Returns:
        Dict with dataset names and their metadata
    """
    info = {}

    for name, filename in STATIC_DATASETS.items():
        json_path = _DATA_DIR / filename
        if json_path.exists():
            with open(json_path, "r") as f:
                data = json.load(f)
            info[name] = {
                "file": filename,
                "ticker": data.get("ticker", name),
                "symbol": data.get("symbol", "N/A"),
                "description": data.get("description", ""),
                "start_date": data.get("start_date", "N/A"),
                "end_date": data.get("end_date", "N/A"),
                "data_points": data.get("data_points", 0),
                "fetched_date": data.get("fetched_date", "N/A"),
            }
        else:
            info[name] = {"file": filename, "status": "not found"}

    return info
