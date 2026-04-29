"""CSV-mode parser for /problem.

Accepts a CSV string of daily prices, one row per trading day, one column per
asset. The first row is treated as the header (asset names). Optional first
column may be a date which we drop.

We deliberately keep the parser dependency-light (stdlib + numpy) so the
backend doesn't pull in pandas just to read 4-12 columns.
"""

from __future__ import annotations

import csv
import io
from typing import List

import numpy as np

from portfolio.data import AssetUniverse


_TRADING_DAYS = 252


def universe_from_price_csv(csv_text: str) -> AssetUniverse:
    reader = csv.reader(io.StringIO(csv_text))
    rows = [r for r in reader if any(cell.strip() for cell in r)]
    if len(rows) < 3:
        raise ValueError(
            "CSV must contain a header plus at least 2 price rows to compute returns."
        )

    header = [c.strip() for c in rows[0]]
    body = rows[1:]

    # Drop first column if it isn't numeric (treat as date).
    drop_first = not _looks_numeric(body[0][0]) if body[0] else False
    if drop_first:
        asset_names = header[1:]
        prices = np.array(
            [[_parse_float(c) for c in row[1:]] for row in body], dtype=float
        )
    else:
        asset_names = header
        prices = np.array([[_parse_float(c) for c in row] for row in body], dtype=float)

    if prices.ndim != 2 or prices.shape[1] != len(asset_names):
        raise ValueError("CSV: column count mismatch between header and data rows.")
    if (prices <= 0).any():
        raise ValueError("CSV: all prices must be strictly positive (log returns).")

    log_returns = np.diff(np.log(prices), axis=0)  # shape (T-1, N)
    if log_returns.shape[0] < 2:
        raise ValueError("CSV: need at least 3 price rows to estimate covariance.")

    mu_daily = log_returns.mean(axis=0)
    sigma_daily = np.cov(log_returns, rowvar=False, ddof=1)

    mu_ann = mu_daily * _TRADING_DAYS
    sigma_ann = sigma_daily * _TRADING_DAYS

    sigma_ann = 0.5 * (sigma_ann + sigma_ann.T)
    eigmin = np.linalg.eigvalsh(sigma_ann).min()
    if eigmin <= 0:
        sigma_ann = sigma_ann + (1e-8 - eigmin) * np.eye(len(asset_names))

    n_assets = len(asset_names)
    sectors = np.zeros(n_assets, dtype=int)  # CSV mode: no sector info

    return AssetUniverse(
        n_assets=n_assets,
        mu=mu_ann,
        sigma=sigma_ann,
        sectors=sectors,
        sector_names=["S0"],
        asset_names=list(asset_names),
    )


def _parse_float(s: str) -> float:
    s = s.strip().replace(",", "")
    if s == "":
        raise ValueError("CSV: empty cell in numeric region.")
    return float(s)


def _looks_numeric(s: str) -> bool:
    try:
        _parse_float(s)
        return True
    except ValueError:
        return False
